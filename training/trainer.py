import torch
import numpy as np

from utils.frame_stack import FrameStack

class Trainer:

    MIN_STEPS_BEFORE_EXIT = 0
    EARLY_EXIT_THRESHOLD = -600

    def __init__(self, env, agent, replay_buffer, optimizer, config, logger, scheduler):
        """Sets up all training components, moves both networks to the available device,
        and initialises the frame stack and global step counter."""

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


    def train(self, num_episodes):

        for episode in range(num_episodes):

            self.run_episode(episode)

    def run_episode(self, episode):
        """Collects one full episode of transitions, pushes them to the replay buffer,
        runs a train_step whenever the buffer is ready, and performs soft target-network
        updates every environment step."""

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

            next_state, reward, terminated, truncated, info = self.env.step(action)
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

            tau = self.config.get("tau", 0.005)
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
        """Executes one Double-DQN gradient update weighted by PER importance-sampling
        weights, clips gradients to norm 1, refreshes transition priorities with the
        resulting TD errors, and returns the scalar loss."""

        states, actions, rewards, next_states, dones, indices, weights = batch

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = self.agent.online_net(next_states_t).argmax(dim=1, keepdim=True)
            target_q = self.agent.target_net(next_states_t)
            max_target_q = target_q.gather(1, next_action).squeeze(1)

            targets = rewards_t + (1.0 - dones_t) * self.config["gamma"] * max_target_q

        action_q = self.agent.online_net(states_t)

        if torch.isnan(action_q).any():
            print("NaN Q Values")
            return 0.0

        action_q = action_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        elementwise_loss = torch.nn.functional.smooth_l1_loss(action_q, targets, reduction='none')
        loss = (weights_t * elementwise_loss).mean()

        if torch.isnan(loss).any():
            print("Nan Loss Detected!")
            return 0.0

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.agent.online_net.parameters(), 1.0)
        
        self.optimizer.step()
        td_errors = (action_q - targets).detach().cpu().numpy()
        if len(indices) > 0:
            agent_td_errors = td_errors[-len(indices):]
            self.replay_buffer.update_priorities(indices, agent_td_errors)
        return float(loss.item())
