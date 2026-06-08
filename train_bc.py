import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from agents.dqn_agent import DQNAgent
from agents.epsilon_scheduler import EpsilonScheduler
from utils.action_discretizer import discretize_action
from enviornments.action_mapper import ActionMapper
from configs.env_config import *

STRAIGHT_MAPS = {"SSSS"}
CURVE_MAPS = {"SCSC", "CSCS", "CCCC", "4"}
class ExpertDataset(Dataset):

    def __init__(self, path, map_family=None):
        data = np.load(path, allow_pickle=False)

        obs_all = data["obs"]
        actions_all = data["actions"]
        maps_all = data["maps"]

        if maps_all.dtype.kind in ("S", "V"):
            maps_all = np.array([m.decode("utf-8").rstrip("\x00") for m in maps_all])

        valid_observations = []
        valid_actions = []

        for i in range(len(obs_all)):
            map_str = str(maps_all[i])

            if map_family == "straight" and map_str not in STRAIGHT_MAPS:
                continue
            if map_family == "curve" and map_str not in {str(m) for m in CURVE_MAPS}:
                continue

            discrete_action = discretize_action(float(actions_all[i][0]), float(actions_all[i][1]))

            obs_array = np.array(obs_all[i], dtype=np.float32)
            valid_observations.append(obs_array)
            valid_actions.append(discrete_action)

        if len(valid_actions) == 0:
            print("\n[UPOZORENJE] Filter akcija je izbacio previše frejmova. Vraćam sirove podatke.")
            for i in range(len(obs_all)):
                obs_array = np.array(obs_all[i], dtype=np.float32)
                valid_observations.append(obs_array)
                valid_actions.append(discretize_action(float(actions_all[i][0]), float(actions_all[i][1])))

        self.observations = torch.tensor(np.array(valid_observations), dtype=torch.float32)
        self.actions = torch.tensor(valid_actions, dtype=torch.long)

        print(f"[DATASET INFO] Filtrirano stajanje preko akcija. Preostalo uzoraka za trening: {len(self.actions)}")

    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, key):
        return self.observations[key], self.actions[key]
    
class BCTrainer:

    def __init__(self, agent, config, obs_size, checkpoint_path):

        self.agent = agent
        self.config = config
        self.obs_size = obs_size
        self.checkpoint_path = checkpoint_path

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def _make_optimizer(self, lr):
        return torch.optim.Adam(self.agent.online_net.parameters(), lr=lr)
    
    def train(self, train_loader, val_loader, lr=None, tag=""):

        cfg = self.config
        lr = lr or cfg["lr"]
        optimizer = self._make_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-5)
        best_val = float("inf")
        no_improve = 0

        for epoch in range(1, cfg["epochs"] + 1):
            self.agent.online_net.train()
            train_loss = 0.0

            for obs_batch, action_batch in train_loader:
                action_logits = self.agent.online_net(obs_batch)

                loss = self.criterion(action_logits, action_batch)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.online_net.parameters(), cfg["clip_grad"])

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            self.agent.online_net.eval()
            val_loss = 0.0

            with torch.no_grad():
                for obs_batch, action_batch in val_loader:
                    action_logits = self.agent.online_net(obs_batch)
                    val_loss += self.criterion(action_logits, action_batch).item()

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
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(self.agent.online_net.state_dict(), self.checkpoint_path)

    def _load_best(self):
        self.agent.online_net.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu", weights_only=True))
        self.agent.target_net.load_state_dict(self.agent.online_net.state_dict())

    
def make_loader(dataset_path, map_family=None, batch_size=256, val_split=0.15):
        ds = ExpertDataset(dataset_path, map_family=map_family)
        if len(ds) == 0:
            return None, None, ds

        val_size = int(len(ds) * val_split)
        train_size = len(ds) - val_size
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

        # Build a WeightedRandomSampler to balance action classes with sqrt dampening
        try:
            # train_ds is a Subset; get original dataset and indices
            full_actions = ds.actions.numpy()
            train_indices = train_ds.indices
            train_actions = full_actions[train_indices]

            class_counts = np.bincount(train_actions, minlength=int(full_actions.max() + 1))
            freqs = class_counts / class_counts.sum()
            weights_per_class = 1.0 / (np.sqrt(freqs + 1e-8))
            sample_weights = weights_per_class[train_actions]

            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
        except Exception:
            # fallback to simple shuffle if something goes wrong
            train_loader = DataLoader(
                train_ds, 
                batch_size=batch_size, 
                shuffle=True, 
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
    
    data = np.load(EXPERT_DATASET, allow_pickle=False)
    lengths = [len(x) for x in data["obs"]]
    print("unique obs lengths:", np.unique(lengths))
    print("first obs:", data["obs"][0])
    
    straight_train, straight_val, straight_ds = make_loader(
        EXPERT_DATASET,
        map_family="straight", 
        batch_size=STRAIGHT_BC_CONFIG["batch_size"],
        val_split=STRAIGHT_BC_CONFIG["val_split"]
    )

    obs_size = straight_ds.observations.shape[1]

    num_actions = ActionMapper().num_actions()

    epsilon_scheduler = EpsilonScheduler(start=0.0, end=0.0, decay=1, warmup_steps=0)
    
    agent = DQNAgent(
        input_size=obs_size, 
        num_actions=num_actions,
        epsilon_scheduler=epsilon_scheduler
    )

    straight_trainer = BCTrainer(agent, STRAIGHT_BC_CONFIG, obs_size, checkpoint_path=BC_CHECKPOINT_STRAIGHT)

    if straight_train:
        straight_trainer.train(straight_train, straight_val, lr=STRAIGHT_BC_CONFIG["lr"], tag="straight")
    else:
        print("[BC] No straight data found")

    curve_train, curve_val, curve_ds = make_loader(EXPERT_DATASET, map_family="curve")

    curve_trainer = BCTrainer(agent, CURVE_BC_CONFIG, obs_size, checkpoint_path=BC_CHECKPOINT_CURVE)
    if curve_train:
        curve_trainer.train(curve_train, curve_val, lr = CURVE_BC_CONFIG["lr"] * CURVE_BC_CONFIG["lr_scale"], tag="curve")

if __name__ == "__main__":
    main()