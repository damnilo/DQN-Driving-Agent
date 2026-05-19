import torch
import numpy as np

from replay.transition import Transition
from utils.logger import Logger

class Trainer:

    def __init__(self, env, agent, replay_buffer, optimizer, config, logger):

        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.config = config
        self.logger = logger

        self.global_step = 0

    def set_map(self, episode):
        if episode < 300:
            self.env.config["map"] = "SSSS"
        elif episode < 600:
            self.env.config["map"] = 3
        else:
            self.env.config["map"] = 5

    def train(self, num_episodes):

        for episode in range(num_episodes):

            self.run_episode(episode)

    def run_episode(self, episode):

        state, _ = self.env.reset()

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

            done = terminated or truncated

            self.replay_buffer.push(Transition(
                state = state,
                action = action,
                reward = reward,
                done = done,
                next_state = next_state
            ))

            state = next_state

            episode_reward += reward

            self.global_step += 1

            if len(self.replay_buffer) >= self.config["batch_size"]:

                batch = self.replay_buffer.sample(
                    self.config["batch_size"]
                )

                loss = self.train_step(batch)
                losses.append(loss)

            if self.global_step % self.config["target_update_freq"] == 0:

                self.agent.update_target_network()
        
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

        states = np.array([t.state for t in batch])

        actions = np.array([t.action for t in batch])

        rewards = np.array([t.reward for t in batch])

        dones = np.array([t.done for t in batch])

        next_states = np.array([t.next_state for t in batch])

        states_t = torch.as_tensor(
            states, 
            dtype=torch.float32
        )

        actions_t = torch.as_tensor(
            actions, 
            dtype=torch.int64
        ).unsqueeze(1)

        rewards_t = torch.as_tensor(
            rewards, 
            dtype=torch.float32
        ).unsqueeze(1)

        dones_t = torch.as_tensor(
            dones, 
            dtype=torch.float32
        ).unsqueeze(1)

        next_states_t = torch.as_tensor(
            next_states, 
            dtype=torch.float32
        )

        with torch.no_grad():

            target_q_values = self.agent.target_net(next_states_t)

            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

            targets = rewards_t + (self.config["gamma"] * (1 - dones_t) * target_q_values)

        q_values = self.agent.online_net(states_t)

        actions_q_values = torch.gather(
            q_values,
            dim=1,
            index=actions_t
        )

        loss = torch.nn.functional.smooth_l1_loss(
            actions_q_values,
            targets
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.agent.online_net.parameters(), 10.0)

        self.optimizer.step()

        return loss.item()
