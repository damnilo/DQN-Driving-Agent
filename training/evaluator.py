from utils.frame_stack import FrameStack

class Evaluator:

    def __init__(self, env, agent, logger):

        self.env = env
        self.agent = agent
        self.logger = logger

    def evaluate(self, num_episodes):

        for episode in range(num_episodes):
            frame_stack = FrameStack(stack_size=4)

            state, _ = self.env.reset()
            state = frame_stack.reset(state)

            done = False

            total_reward = 0

            info = {}

            while not done:

                action = self.agent.select_action(state, step=0, training=False)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state = frame_stack.step(next_state)
                state = next_state

                done = terminated or truncated

                total_reward += reward
            
            is_success = info.get("arrive_dest", False)
            is_collision = info.get("crash", False) or info.get("crash_vehicle", False) or info.get("crash_object", False)
            is_out_of_road = info.get("out_of_road", False)

            if is_collision or is_out_of_road:
                is_success = False

            self.logger.log_episode_result(
                episode = episode,
                success = is_success,
                collision = is_collision,
                out_of_road = is_out_of_road,
                reward = total_reward
            )