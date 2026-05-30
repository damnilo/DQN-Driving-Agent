import json
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from agents.dqn_agent import DQNAgent
from agents.epsilon_scheduler import EpsilonScheduler
from utils.action_discretizer import continuous_to_discrete
from configs.dqn_configs import *
from enviornments.action_mapper import ActionMapper
from configs.env_config import EXPERT_DATASET, BC_CHECKPOINT, BC_CONFIG, FRAME_STACK

STRAIGHT_MAPS = {"SSSS"}
CURVE_MAPS = {"SCSC", "CSCS", "CCCC", "4"}
class ExpertDataset(Dataset):

    def __init__(self, path, map_family=None):

        self.action_map = ActionMapper().action
        with open(path, "r") as f:
            raw = json.load(f)

        valid_observations = []
        valid_actions = []

        for item in raw:

            if map_family == "straight" and str(item.get("map")) not in STRAIGHT_MAPS:
                continue
            if map_family == "curve" and str(item.get("map")) not in {str(m) for m in CURVE_MAPS}:
                continue

            discrete_action = continuous_to_discrete(item["action_steering"], item["action_throttle"], self.action_map)

            if discrete_action == 2 and abs(item["action_throttle"]) < 1e-3:
                continue

            obs_array = np.array(item["observation"], dtype=np.float32)
            valid_observations.append(obs_array)
            valid_actions.append(discrete_action)

        if len(valid_actions) == 0:
            print("\n[UPOZORENJE] Filter akcija je izbacio previše frejmova. Vraćam sirove podatke.")
            for item in raw:
                obs_array = np.array(item["observation"], dtype=np.float32)
                valid_observations.append(obs_array)
                valid_actions.append(continuous_to_discrete(item["action_steering"], item["action_throttle"], self.action_map))

        self.observations = torch.tensor(np.array(valid_observations), dtype=torch.float32)
        self.actions = torch.tensor(valid_actions, dtype=torch.long)

        print(f"[DATASET INFO] Filtrirano stajanje preko akcija. Preostalo uzoraka za trening: {len(self.actions)}")

    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, key):
        return self.observations[key], self.actions[key]
    
class BCTrainer:

    def __init__(self, agent, config, obs_size):

        self.agent = agent
        self.config = config
        self.obs_size = obs_size

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def _make_optimizer(self, lr):
        return torch.optim.Adam(self.agent.online_net.parameters(), lr=lr)
    
    def train(self, train_loader, val_loader, lr=None, tag=""):

        cfg = self.config
        lr = lr or cfg["lr"]
        optimizer = self._make_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BC_CONFIG["epochs"], eta_min=1e-5)
        best_val = float("inf")
        no_improve = 0

        for epoch in range(1, cfg["epochs"] + 1):
            self.agent.online_net.train()
            train_loss = 0.0

            for obs_batch, action_batch in train_loader:
                logits = self.agent.online_net(obs_batch)

                loss = self.criterion(logits, action_batch)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.online_net.parameters(),
                                          cfg["clip_grad"])

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.agent.online_net.eval()
            val_loss = 0.0

            with torch.no_grad():
                for obs_batch, action_batch in val_loader:
                    logits = self.agent.online_net(obs_batch)
                    val_loss += self.criterion(logits, action_batch).item()

            val_loss /= len(val_loader)
            scheduler.step()

            print(
                f"[BC {tag}] Epoha [{epoch:03d}/{cfg['epochs']}] "
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
        self.agent.online_net.load_state_dict(torch.load(BC_CHECKPOINT, map_location="cpu", weights_only=True))
        self.agent.target_net.load_state_dict(self.agent.online_net.state_dict())

    
def make_loader(dataset_path, map_family=None, batch_size=256, val_split=0.15):
        ds = ExpertDataset(dataset_path, map_family=map_family)
        if len(ds) == 0:
            return None, None, ds

        val_size = int(len(ds) * val_split)
        train_size = len(ds) - val_size
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

        train_labels = [ds.actions[i].item() for i in train_ds.indices]
        num_actions = ActionMapper().num_actions()
        class_count = torch.bincount(torch.tensor(train_labels, dtype=torch.long), minlength=num_actions)
        class_weights = 1.0 / (class_count.float().sqrt() + 1e-5)
        class_weights[class_count == 0] = 0.0
        class_weights /= class_weights.sum()
        sample_weights = [class_weights[l].item() for l in train_labels]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_ds, 
            batch_size=batch_size, 
            sampler=sampler, 
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader, ds

def main():
    if not os.path.exists(EXPERT_DATASET):
        raise FileNotFoundError
    
    straight_train, straight_val, straight_ds = make_loader(EXPERT_DATASET, map_family="straight")

    obs_size = straight_ds.observations.shape[1]

    num_actions = ActionMapper().num_actions()

    epsilon_scheduler = EpsilonScheduler(start=0.0, end=0.0, decay=1, warmup_steps=0)
    
    agent = DQNAgent(
        input_size=obs_size, 
        num_actions=num_actions,
        epsilon_scheduler=epsilon_scheduler
    )

    trainer = BCTrainer(agent, BC_CONFIG, obs_size)

    if straight_train:
        trainer.train(straight_train, straight_val, lr=BC_CONFIG["lr"], tag="straight")
    else:
        print("[BC] No straight data found")

    curve_train, curve_val, curve_ds = make_loader(EXPERT_DATASET, map_family="curve")
    if curve_train:
        trainer.train(curve_train, curve_val, lr = BC_CONFIG["lr"] * 0.3, tag="curve")

if __name__ == "__main__":
    main()