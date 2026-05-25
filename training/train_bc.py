import json
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from agents.dqn_agent import DQNAgent
from agents.epsilon_scheduler import EpsilonScheduler
from configs.dqn_configs import *
from enviornments.action_mapper import ActionMapper
from configs.env_config import EXPERT_DATASET, BC_CHECKPOINT, BC_CONFIG

FRAME_STACK = 4

class ExpertDataset(Dataset):

    def __init__(self, path):

        with open(path, "r") as f:
            raw = json.load(f)

        self.observations = torch.tensor([item["observation"] for item in raw], dtype=torch.float32)

        action = []

        for item in raw:
            discrete_action = self.continuous_to_discrete(item["action_steering"], item["action_throttle"])

            action.append(discrete_action)

        self.actions = torch.tensor(action, dtype=torch.long)

    def continuous_to_discrete(self, steering, throttle):
        
        best_action = 0
        best_distance = float("inf")
        action_map = ActionMapper().action

        for idx, action in action_map.items():

            ds = steering - action[0]
            dt = throttle - action[1]

            distance = ds * ds + dt * dt

            if distance < best_distance:
                best_distance = distance
                best_action = idx

        return best_action

    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, key):
        obs = self.observations[key]
        stacked = obs.repeat(FRAME_STACK)
        return stacked, self.actions[key]
    
class BCTrainer:

    def __init__(self, agent, config):

        self.agent = agent
        self.config = config
        
        self.optimizer = torch.optim.Adam(self.agent.online_net.parameters(), lr=config["lr"])

        self.criterion = nn.CrossEntropyLoss()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=2, factor=0.5)
    
    
    def train(self, train_loader, val_loader):

        cfg = self.config
        best_val = float("inf")
        no_improve = 0

        for epoch in range(1, cfg["epochs"] + 1):
            self.agent.online_net.train()
            train_loss = 0.0

            for obs_batch, action_batch in train_loader:
                logits = self.agent.online_net(obs_batch)

                loss = self.criterion(logits, action_batch)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.online_net.parameters(), cfg["clip_grad"])

                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.agent.online_net.eval()
            val_loss = 0.0

            with torch.no_grad():
                for obs_batch, action_batch in val_loader:
                    logits = self.agent.online_net(obs_batch)
                    val_loss += self.criterion(logits, action_batch).item()

            val_loss /= len(val_loader)
            self.scheduler.step(val_loss)

            print(
                f"Epoha [{epoch:03d}/{cfg['epochs']}] "
                f"train_loss: {train_loss:.5f}  val_loss: {val_loss:.5f}"
            )

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                no_improve = 0
                self._save_best()
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    break
        
        self._load_best()
    
    def _save_best(self):
        os.makedirs(os.path.dirname(BC_CHECKPOINT), exist_ok=True)
        torch.save(self.agent.online_net.state_dict(), BC_CHECKPOINT)

    def _load_best(self):
        self.agent.online_net.load_state_dict(torch.load(BC_CHECKPOINT, weights_only=True))
        self.agent.target_net.load_state_dict(self.agent.online_net.state_dict())

def main():
    if not os.path.exists(EXPERT_DATASET):
        raise FileNotFoundError
    
    full_dataset = ExpertDataset(EXPERT_DATASET)

    val_size = int(len(full_dataset) * BC_CONFIG["val_split"])
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size = BC_CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = BC_CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    epsilon_scheduler = EpsilonScheduler(start=0.0, end=0.0, decay=1, warmup_steps=0)
    print(f"BC obs_size: {full_dataset.observations.shape[1]}")

    obs_size = full_dataset.observations.shape[1] * FRAME_STACK

    try:
        from enviornments.metadrive_env import MetaDriveEnvWrapper
        from utils.env_randomizer import get_random_metadrive_config
        _tmp_env = MetaDriveEnvWrapper(get_random_metadrive_config())
        num_actions = _tmp_env.num_actions()
        _tmp_env.close()

    except Exception:

        num_actions = 11

    agent = DQNAgent(
        input_size=obs_size, 
        num_actions=num_actions,
        epsilon_scheduler=epsilon_scheduler
    )

    trainer = BCTrainer(agent, BC_CONFIG)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()