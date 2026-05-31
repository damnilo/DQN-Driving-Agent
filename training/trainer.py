import random

import torch
import numpy as np

from utils.frame_stack import FrameStack
from configs.env_config import ENV_CONFIG, EXPERT_RATIO, EXPERT_RATIO_CURVE

class Trainer:

    MIN_STEPS_BEFORE_EXIT = 0
    EARLY_EXIT_THRESHOLD = -300

    def __init__(self, env, agent, replay_buffer, optimizer, config, logger, scheduler):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        self.frame_stack = FrameStack(stack_size=4)
        self.global_step = 0
        self.scheduler = scheduler
        self._last_map = None

        self.agent.online_net.to(self.device)
        self.agent.target_net.to(self.device)

    def _pick_map_for_episode(self, episode: int):
        if episode < 200:
            return "SSSS", 0.0
        if episode < 500:
            return random.choices(["SSSS", "SCSC", "CSCS"], weights=[0.50, 0.25, 0.25])[0], 0.05
        if episode < 900:
            return random.choices(["SSSS", "SCSC", "CSCS"], weights=[0.25, 0.40, 0.35])[0], 0.1
        if episode < 1400:
            return random.choices(["SSSS", "SCSC", "CSCS", "CCCC"], weights=[0.15, 0.25, 0.25, 0.35])[0], 0.15
        if episode < 2000:
            return random.choices(["SSSS", "CSCS", "CCCC", 4], weights=[0.10, 0.25, 0.30, 0.35])[0], 0.2
        return random.choices(["SSSS", "SCSC", "CSCS", "CCCC", 4], weights=[0.10, 0.15, 0.15, 0.20, 0.40])[0], 0.25 

    def _horizon_for_map(self, target_map) -> int:
        if target_map == "SSSS":
            return 800
        if target_map in ("SCSC", "CSCS"):
            return 1000
        if target_map == "CCCC":
            return 1200
        return 1600

    @staticmethod
    def _map_family(target_map) -> str:
        return "straight" if target_map == "SSSS" else "curve"

    def set_map(self, episode):
        target_map, density = self._pick_map_for_episode(episode)
        horizon = self._horizon_for_map(target_map)

        if self._last_map == target_map:
            return False

        old_family = self._map_family(self._last_map) if self._last_map is not None else None
        new_family = self._map_family(target_map)

        current_map = self.env.env.config.get("map") if self._last_map is not None else None
        print(f"[Curriculum] Ep {episode}: {current_map} -> {target_map}")

        self.env.close()

        current_config = dict(ENV_CONFIG)
        current_config["map"] = target_map
        current_config["traffic_density"] = density
        current_config["horizon"] = horizon
        current_config["start_seed"] = 0
        current_config["num_scenarios"] = 50

        from enviornments.metadrive_env import MetaDriveEnvWrapper

        self.env = MetaDriveEnvWrapper(current_config)
        self.frame_stack = FrameStack(stack_size=4)
        self._last_map = target_map

        if target_map == "SSSS":
            self.replay_buffer.expert_ratio = EXPERT_RATIO
            self.env.reward_function.use_soft_out_of_road = False
        else:
            self.replay_buffer.expert_ratio = EXPERT_RATIO_CURVE
            self.env.reward_function.use_soft_out_of_road = True

        return True

    def train(self, num_episodes):

        for episode in range(num_episodes):

            self.run_episode(episode)

    def run_episode(self, episode):
        start_step = self.global_step
        state, _ = self.env.reset()
        state = self.frame_stack.reset(state)
        
        done = False

        episode_reward = 0.0

        losses = []
        steps = 0

        while not done:

            action = self.agent.select_action(
                state,
                self.global_step,
                training=True
            )

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self.frame_stack.step(next_state)

            done = terminated or truncated

            self.replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done 
            )

            state = next_state

            episode_reward += reward

            self.global_step += 1

            if self.replay_buffer.is_ready(self.config["min_replay_size"]):

                batch = self.replay_buffer.sample(
                    self.config["batch_size"]
                )

                loss = self.train_step(batch)
                losses.append(loss)

            tau = 0.005
            for target_param, online_param in zip(self.agent.target_net.parameters(), self.agent.online_net.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

            if steps >= self.MIN_STEPS_BEFORE_EXIT and episode_reward < self.EARLY_EXIT_THRESHOLD and self.global_step > 300:
                print("[Trainer] Epizoda zavrsena ranije zbog loseg ucenja...")
                done = True

            steps += 1
        
        avg_loss = np.mean(losses) if losses else 0.0

        self.logger.log_episode(
            episode = episode,
            reward = episode_reward,
            epsilon = self.agent.epsilon_scheduler.get_epsilon(start_step),
            avg_loss = avg_loss,
            global_step = self.global_step,
            steps = steps
        )

        return episode_reward
    
    def train_step(self, batch):

        states, actions, rewards, next_states, dones = batch

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            was_training = self.agent.online_net.training
            self.agent.online_net.eval()

            next_action = self.agent.online_net(next_states_t).argmax(dim=1, keepdim=True)
            target_q_values = self.agent.target_net(next_states_t)
            max_target_q_values = target_q_values.gather(1, next_action)

            self.agent.online_net.train(was_training)

            targets = rewards_t + (self.config["gamma"] * (1 - dones_t) * max_target_q_values)

            targets = torch.clamp(targets, min=-500.0, max=500.0)

        q_values = self.agent.online_net(states_t)

        if torch.isnan(q_values).any():
            print("NaN Q Values")
            return 0.0

        actions_q_values = torch.gather(q_values, dim=1, index=actions_t)

        loss = torch.nn.functional.smooth_l1_loss(actions_q_values, targets)

        if torch.isnan(loss).any():
            print("Nan Loss Detected!")
            return 0.0

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.agent.online_net.parameters(), 1.0)
        
        self.optimizer.step()
        return float(loss.item())
