import torch
import numpy as np

from replay.transition import Transition
from utils.logger import Logger
from utils.frame_stack import FrameStack

class Trainer:

    def __init__(self, env, agent, replay_buffer, optimizer, config, logger, scheduler):

        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        self.frame_stack = FrameStack(stack_size=4)
        self.global_step = 0
        self.scheduler = scheduler

    def set_map(self, episode):
        if episode < 300:
            self.env.env.config["map"] = "SSSS"
            self.env.env.config["traffic_density"] = 0.0
            self.env.env.config["horizon"] = 500
        elif episode < 600:
            self.env.env.config["map"] = "SCSC"
            self.env.env.config["traffic_density"] = 0.03
            self.env.env.config["horizon"] = 600
        elif episode < 900:
            self.env.env.config["map"] = "SCSCS"
            self.env.env.config["traffic_density"] = 0.05
            self.env.env.config["horizon"] = 700
        elif episode < 1300:
            self.env.env.config["map"] = 3
            self.env.env.config["traffic_density"] = 0.08
            self.env.env.config["horizon"] = 800
        elif episode < 1800:
            self.env.env.config["map"] = 4
            self.env.env.config["traffic_density"] = 0.12
            self.env.env.config["horizon"] = 900
        else:
            self.env.env.config["map"] = 5
            self.env.env.config["traffic_density"] = 0.15
            self.env.env.config["horizon"] = 1000

    def train(self, num_episodes):

        for episode in range(num_episodes):

            self.run_episode(episode)

    def run_episode(self, episode):

        state, _ = self.env.reset()
        state = self.frame_stack.reset(state)
        
        done = False

        episode_reward = 0.0

        losses = []

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


            TAU = 0.0001
            for online_p, target_p in zip(
                self.agent.online_net.parameters(), self.agent.target_net.parameters()
            ):
                target_p.data.copy_(TAU * online_p.data + (1.0 - TAU) * target_p.data
            )
        
        avg_loss = np.mean(losses) if losses else 0.0

        self.logger.log_episode(
            episode = episode,
            reward = episode_reward,
            epsilon = self.agent.epsilon_scheduler.get_epsilon(self.global_step),
            avg_loss = avg_loss,
            global_step = self.global_step
        )

        return episode_reward
    
    def train_step(self, batch):
        # PRIJE — očekivao Transition objekte:
        # states = np.array([t.state for t in batch])

        # POSLIJE — ExpertReplayBuffer.sample() vraća direktno arraye:
        states, actions, rewards, next_states, dones = batch

        states_t = torch.as_tensor(states, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32)

        with torch.no_grad():
            next_action = self.agent.online_net(next_states_t).argmax(dim=1, keepdim=True)
            target_q_values = self.agent.target_net(next_states_t)
            max_target_q_values = target_q_values.gather(1, next_action)[0]
            targets = rewards_t + (self.config["gamma"] * (1 - dones_t) * max_target_q_values)

            targets = torch.clamp(targets, min=-50.0, max=50.0)

        q_values = self.agent.online_net(states_t)

        if torch.isnan(q_values).any():
            print("NaN Q Values")
            return

        actions_q_values = torch.gather(q_values, dim=1, index=actions_t)

        loss = torch.nn.functional.smooth_l1_loss(actions_q_values, targets)

        if torch.isnan(loss):
            print("Nan Loss Detected!")
            return

        self.optimizer.zero_grad()
        loss.backward()

        total_norm = 0.0

        for p in self.agent.online_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.agent.online_net.parameters(), 5.0)
        self.optimizer.step()
        return loss.item()
