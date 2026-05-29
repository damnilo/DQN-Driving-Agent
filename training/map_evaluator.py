from enviornments.metadrive_env import MetaDriveEnvWrapper
from utils.frame_stack import FrameStack
from configs.env_config import ENV_CONFIG

class MapEvaluator:

    def __init__(self, agent, logger=None):
        self.agent = agent
        self.logger = logger

    def evaluate_maps(self, maps, seeds, episodes_per_map=5):
        results = []

        for map_name in maps:
            for seed in seeds:
                config = dict(ENV_CONFIG)
                config["use_render"] = False
                config["map"] = map_name
                config["start_seed"] = seed
                config["num_scenarios"] = episodes_per_map
                config["traffic_density"] = 0.0
                config["horizon"] = 1000

                env = MetaDriveEnvWrapper(config)

                for episode in range(episodes_per_map):
                    result = self._run_episode(env, map_name, seed, episode)
                    results.append(result)

                    print(
                        f"[MapEval] map={map_name} seed={seed} "
                        f"ep={episode} success={result['success']} "
                        f"reason={result['reason']} reward={result['reward']:.2f} "
                        f"steps={result['steps']}"
                    )

                env.close()

        return results
    
    def _run_episode(self, env, map_name, seed, episode):
        frame_stack = FrameStack(stack_size=4)

        state, _ = env.reset()
        state = frame_stack.reset(state)

        done = False
        total_reward = 0.0
        steps = 0
        info = {}

        while not done:
            action = self.agent.select_action(state, step=0, training=False)

            next_state, reward, terminated, truncated, info = env.step(action)
            state = frame_stack.step(next_state)

            done = terminated or truncated
            total_reward += reward
            steps += 1

        reason = "UNKNOWN"
        if info.get("arrive_dest", False):
            reason = "SUCCESS"
        elif info.get("out_of_road", False):
            reason = "OUT_OF_ROAD"
        elif info.get("crash", False):
            reason = "CRASH"
        elif info.get("max_step", False):
            reason = "TIMEOUT"

        return {
            "map": map_name,
            "seed": seed,
            "episode": episode,
            "reward": total_reward,
            "steps": steps,
            "reason": reason,
            "success": reason == "SUCCESS",
        }