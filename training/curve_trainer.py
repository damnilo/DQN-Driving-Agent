import numpy as np
from training.trainer import Trainer

class CurveTrainer(Trainer):
    MIN_STEPS_BEFORE_EXIT = 150
    EARLY_EXIT_THRESHOLD = -200

    def __init__(self, env, agent, optimizer, replay_buffer, config, logger):
        super().__init__(env=env, agent=agent, optimizer=optimizer, replay_buffer=replay_buffer, config=config, logger=logger, scheduler=None)

    def run_episode(self, episode):
        start_step = self.global_step
        state, _ = self.env.reset()
        state = self.frame_stack.reset()

        done = False
        episode_reward = 0.0
        losses = []
        steps = 0

        while not done:
            action = self.agent.select_action(state, self.global_step, training=True)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self.frame_stack.step(next_state)

            done = terminated or truncated

            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            self.global_step += 1

            if self.replay_buffer.is_ready(self.config["min_replay_size"]):
                batch = self.replay_buffer.sample(self.config["batch_size"])
                loss = self.train_step(batch)
                losses.append(loss)

            tau = 0.005
            for tp, op in zip(self.agent.target_net.parameters(), self.agent.online_net.parameters()):
                tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

            if steps >= self.MIN_STEPS_BEFORE_EXIT and episode_reward < self.EARLY_EXIT_THRESHOLD:
                done = True

            steps += 1

        avg_loss = np.mean(losses) if losses else 0.0
        self.logger.log_episode(
            episode=episode,
            reward=episode_reward,
            epsilon=self.agent.epsilon_scheduler.get_epsilon(start_step),
            avg_loss = avg_loss,
            global_step = self.global_step,
            steps = steps
        )

        return episode_reward