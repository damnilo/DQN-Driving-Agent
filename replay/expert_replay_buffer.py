import json
import random
import numpy as np
from collections import deque
from utils.action_discretizer import continuous_to_discrete
from typing import Tuple, List
from enviornments.action_mapper import ActionMapper
from configs.env_config import FRAME_STACK

Transition = Tuple[
    np.ndarray, int, float, np.ndarray, bool
]


class ExpertReplayBuffer:

    def __init__(self, capacity: int, expert_dataset_path: str, num_actions: int, expert_ratio: float = 0.25):

        self.capacity = capacity
        self.num_actions = num_actions
        self.expert_ratio = expert_ratio

        self._agent_buffer: deque = deque(maxlen=capacity)

        self._expert_buffer: List[Transition] = []

        self._action_map = ActionMapper().action

        self._load_expert_data(expert_dataset_path)

    def _load_expert_data(self, path: str, map_filter=None) -> None:

        if not path:
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

        except FileNotFoundError:
            print(f"[ExpertReplayBuffer] Dataset nije pronadjen")

            return

        except json.JSONDecodeError as e:
            print(f"[ExpertReplayBuffer] Ostecen JSON: {path}")
            print(f"  {e}")
            print("  Pokreni: python salvage_dataset.py")
            print("  Ili ponovo: python collect_idm.py")
            return
        
        transition = []

        for i, item in enumerate(raw):
            
            if map_filter and str(item.get("map", "")) not in map_filter:
                continue

            obs = np.array(item["observation"], dtype=np.float32)
            next_obs = np.array(item["next_observation"], dtype=np.float32)

            steering = float(item["action_steering"])
            throttle = float(item["action_throttle"])
            action_idx = continuous_to_discrete(steering, throttle, self._action_map)

            reward = float(item.get("reward", 0.0))
            done = bool(item.get("done", False))

            transition.append((obs, action_idx, reward, next_obs, done))

        self._expert_buffer = transition
        print(f"[ExpertReplayBuffer] Ucitano {len(self._expert_buffer)} ekspertskih tranzicija")

    def push(self, obs, action, reward, next_obs, done):

        self._agent_buffer.append((obs.astype(np.float32), int(action), float(reward), next_obs.astype(np.float32), bool(done)))

    def sample(self, batch_size):

        n_expert = 0
        n_agent = batch_size

        has_expert = len(self._expert_buffer) > 0
        has_agent = len(self._agent_buffer) >= batch_size

        if has_expert and has_agent:
            n_expert = int(batch_size * self.expert_ratio)
            n_agent = batch_size - n_expert
        elif not has_agent and has_expert:
            n_expert = batch_size
            n_agent = 0

        samples: List[Transition] = []

        if n_expert > 0:
            samples += random.sample(self._expert_buffer, min(n_expert, len(self._expert_buffer)))

        if n_agent > 0:
            samples += random.sample(list(self._agent_buffer), min(n_agent, len(self._agent_buffer)))

        if not samples:
            raise RuntimeError("Replay Buffer je prazan. Ponovo pokreni collect_idm.py")

        obs_arr, actions_arr, rewards_arr, next_obs_arr, dones_arr = zip(*samples)

        return (np.array(obs_arr, dtype=np.float32), np.array(actions_arr, dtype=np.int64),
                np.array(rewards_arr, dtype=np.float32), np.array(next_obs_arr, dtype=np.float32),
                np.array(dones_arr, dtype=np.float32))
    
    def __len__(self):
        return len(self._agent_buffer)
    
    @property
    def agent_size(self):
        return len(self._agent_buffer)
    
    @property
    def expert_size(self):
        return len(self._expert_buffer)
    
    def is_ready(self, min_size):
        if len(self._expert_buffer) >= min_size:
            return True
        return self.agent_size >= min_size
    
    def clear_agent_buffer(self):
        self._agent_buffer.clear()