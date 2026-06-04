from utils.frame_stack import FrameStack
from configs.env_config import ENV_CONFIG
from enviornments.metadrive_env import MetaDriveEnvWrapper


class Evaluator:

    def __init__(self, env, agent, logger):

        self.env = env
        self.agent = agent
        self.logger = logger

    def _horizon_for_map(self, map_name) -> int:
        if map_name == "SSSS":
            return 800
        if map_name in ("SCSC", "CSCS"):
            return 1500
        if map_name == "CCCC":
            return 1800
        return 2000

    def _make_env_for_map(self, map_name):
        config = dict(ENV_CONFIG)
        config["map"] = map_name
        config["horizon"] = self._horizon_for_map(map_name)
        config["num_scenarios"] = 20
        config["start_seed"] = 0
        config["traffic_density"] = 0.0
        env = MetaDriveEnvWrapper(config)
        if map_name == "SSSS":
            env.reward_function.use_soft_out_of_road = False
        else:
            env.reward_function.use_soft_out_of_road = True
        return env

    def evaluate_on_map(self, map_name, num_episodes=10):
        eval_env = self._make_env_for_map(map_name)

        success_count = 0
        total_eval_reward = 0.0

        try:
            for episode in range(num_episodes):
                frame_stack = FrameStack(stack_size=4)

                state, _ = eval_env.reset()
                state = frame_stack.reset(state)

                next_obs, _, term, trunc, info = eval_env.step_continuous([0.0, 0.3])
                if not (term or trunc):
                    state = frame_stack.step(next_obs)
                else:
                    state = frame_stack.reset(next_obs)

                done = False
                total_reward = 0.0
                info = {}

                while not done:
                    action = self.agent.select_action(state, step=0, training=False)

                    next_state, reward, terminated, truncated, info = eval_env.step(action)
                    next_state = frame_stack.step(next_state)
                    state = next_state

                    done = terminated or truncated
                    total_reward += reward

                reason = "unknown"
                if info.get("arrive_dest", False):
                    reason = "SUCCESS"
                elif info.get("out_of_road", False):
                    reason = "OUT_OF_ROAD"
                elif info.get("crash", False):
                    reason = "CRASH"
                elif info.get("max_step", False):
                    reason = "TIMEOUT"

                is_success = reason == "SUCCESS"
                is_collision = reason == "CRASH"
                is_out_of_road = reason == "OUT_OF_ROAD"

                self.logger.log_episode_result(
                    episode=episode,
                    reward=total_reward,
                    success=is_success,
                    collision=is_collision,
                    out_of_road=is_out_of_road,
                    map_name=map_name,
                )

                total_eval_reward += total_reward
                if is_success:
                    success_count += 1

                print(
                    f"[Eval] map={map_name} ep={episode:4d} | "
                    f"Reward {total_reward:8.2f} | {reason}"
                )
        finally:
            eval_env.close()

        success_rate = success_count / num_episodes
        avg_reward = total_eval_reward / num_episodes

        return {
            "map": map_name,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "success_count": success_count,
            "num_episodes": num_episodes,
        }

    def evaluate_maps(self, maps, episodes_per_map=10):
        """
        Evaluacija na vise mapa. Pre poziva mora biti zatvoren
        training MetaDrive env (trainer.env.close()), inace MetaDrive baca
        AssertionError: Can not call this API after engine initialization!
        """
        results = {}
        for map_name in maps:
            results[map_name] = self.evaluate_on_map(map_name, num_episodes=episodes_per_map)
        return results

    def evaluate(self, num_episodes):
        """Evaluacija na trenutnoj mapi (kompatibilnost sa starim kodom)."""
        current_map = self.env.env.config.get("map", "SSSS")
        result = self.evaluate_on_map(current_map, num_episodes=num_episodes)
        return result["success_rate"], result["avg_reward"]
