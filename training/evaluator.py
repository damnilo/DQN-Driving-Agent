from utils.frame_stack import FrameStack

class Evaluator:

    def __init__(self, env, agent, logger, stack_size=4):

        self.env = env
        self.agent = agent
        self.logger = logger
        self.frame_stack = FrameStack(stack_size=stack_size)

    def evaluate(self, num_episodes):

        for episode in range(num_episodes):

            state, _ = self.env.reset()
            state = self.frame_stack.reset(state)

            done = False

            total_reward = 0

            info = {}

            while not done:

                action = self.agent.select_action(state, step=0, training=False)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = self.frame_stack.step(next_state)

                done = terminated or truncated

                total_reward += reward

            self.logger.log_episode_result(
                episode = episode,
                success = info.get("arrive_dest", False),
                collision = info.get("crash", False),
                out_of_road = info.get("out_of_road", False),
                reward = total_reward
            )