import json
import random
import numpy as np
from collections import deque
from utils.action_discretizer import discretize_action
from typing import Tuple, List
from enviornments.action_mapper import ActionMapper
from configs.env_config import FRAME_STACK

class Transition:

    def __init__(self, obs, action, reward, next_obs, done, priority=1.0):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done
        self.priority = priority

STRAIGHT_MAPS = {"SSSS"}
CURVE_MAPS = {"SCSC", "CSCS", "CCCC", "4"}

class ExpertReplayBuffer:
    ALPHA = 0.6
    BETA = 0.4
    BETA_INCREMENT = 1e-6
    EPS = 1e-5

    def __init__(self, capacity: int, expert_dataset_path: str, num_actions: int, expert_ratio: float = 0.25, map_filter=None):

        self.capacity = capacity
        self.num_actions = num_actions
        self.expert_ratio = expert_ratio

        self._agent_buffer: deque = deque(maxlen=capacity)
        self.max_priority = 1.0
        self._expert_buffer: List[Transition] = []

        self._load_expert_data(expert_dataset_path, map_filter)

    def _load_expert_data(self, path: str, map_filter=None) -> None:

        if not path:
            return
        
        try:
            data = np.load(path, allow_pickle=True)

        except FileNotFoundError:
            print(f"[ExpertReplayBuffer] Dataset nije pronadjen")

            return

        except json.JSONDecodeError as e:
            print(f"[ExpertReplayBuffer] Ostecen JSON: {path}")
            print(f"  {e}")
            print("  Pokreni: python salvage_dataset.py")
            print("  Ili ponovo: python collect_idm.py")
            return
        
        obs_all = data["obs"]
        next_obs_all = data["next_obs"]
        actions_all = data["actions"]
        rewards_all = data["rewards"]
        dones_all = data["dones"]
        maps_all = data["maps"]

        if maps_all.dtype.kind in {'U', 'S'}:
            maps_all = np.array([m.decode("utf-8").rstrip("\x00") for m in maps_all])

        transition = []

        for i in range(len(obs_all)):
            maps_str = str(maps_all[i])
            
            if map_filter and maps_str not in map_filter:
                continue
            

            discrete_action = discretize_action(float(actions_all[i][0]), float(actions_all[i][1]))
            transition.append((obs_all[i], int(discrete_action), float(rewards_all[i]), next_obs_all[i], bool(dones_all[i])   ))

        self._expert_buffer = transition
        print(f"[ExpertReplayBuffer] Ucitano {len(self._expert_buffer)} ekspertskih tranzicija")

    def push(self, obs, action, reward, next_obs, done):

        transition = Transition(obs, action, reward, next_obs, done, priority=self.max_priority)
        self._agent_buffer.append(transition)

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
            priorities = np.array([t.priority for t in self._agent_buffer])

            probs = priorities ** self.ALPHA
            probs /= probs.sum()

            indicies = np.random.choice(
                len(self._agent_buffer),
                size=min(n_agent, len(self._agent_buffer)),
                p=probs
            )

            agent_samples = [self._agent_buffer[i] for i in indicies]
            samples += agent_samples
        else:
            indicies = np.array([], dtype=np.int64)

        if not samples:
            raise RuntimeError("Replay Buffer je prazan. Ponovo pokreni collect_idm.py")
        
        if n_agent > 0:
            weights = (len(self._agent_buffer) * probs[indicies]) ** (-self.BETA)
            weights /= weights.max()
        else:
            weights = np.ones(len(samples), dtype=np.float32)

        weights = weights.astype(np.float32)

        self.BETA = min(1.0, self.BETA + self.BETA_INCREMENT)

        obs_arr = [t.obs for t in samples]
        actions_arr = [t.action for t in samples]
        rewards_arr = [t.reward for t in samples]
        next_obs_arr = [t.next_obs for t in samples]
        dones_arr = [t.done for t in samples]

        return (np.array(obs_arr, dtype=np.float32), np.array(actions_arr, dtype=np.int64),
                np.array(rewards_arr, dtype=np.float32), np.array(next_obs_arr, dtype=np.float32),
                np.array(dones_arr, dtype=np.float32), indicies, weights)
    
    def update_priorities(self, indicies, td_errors):
        
        for idx, td_error in zip(indicies, td_errors):
            td_error_arr = np.asarray(td_error)
            if td_error_arr.size != 1:
                td_error_arr = td_error_arr.reshape(-1)[0]

            priority = float(np.abs(td_error_arr)) + self.EPS
            self._agent_buffer[idx].priority = priority

            if priority > self.max_priority:
                self.max_priority = priority
    
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