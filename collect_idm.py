import os
import numpy as np
from enviornments.metadrive_env import MetaDriveEnvWrapper
from metadrive.policy.expert_policy import ExpertPolicy
from utils.frame_stack import FrameStack
from configs.env_config import *

NUM_EPISODES = 300

RESTART_EVERY = 20

def normalize_action(steering, throttle):
    steering = float(np.clip(steering, -1.0, 1.0))
    throttle = float(np.clip(throttle / 1.0, -0.5, 1.0))
    return steering, throttle

def reset_with_timeout(env):
    # MetaDrive / Panda3D must be reset from the main interpreter thread.
    # The previous threaded approach caused `signal only works in main thread`.
    try:
        return env.reset()
    except Exception as e:
        # If reset gets stuck or fails, let the caller handle restart.
        print(f"env.reset() failed: {e}")
        return None
    
def map_for_episode(episode, num_episodes):
    # target distribution: ~22% straight (SSSS), rest mixed
    ratio = episode / max(1, num_episodes)

    if ratio < 0.22:
        return "SSSS"
    elif ratio < 0.48:
        return "SCSC"
    elif ratio < 0.74:
        return "CSCS"
    elif ratio < 0.92:
        return "CCCC"
    return "SCSC"

def round_obs(arr, decimals=5):
    return np.round(arr, decimals).tolist()

def main():

    os.makedirs("dataset", exist_ok=True)

    base_config = dict(ENV_CONFIG)
    base_config["agent_policy"] = ExpertPolicy
    base_config["traffic_density"] = 0.0
    base_config["accident_prob"] = 0.0
    
    frame_stack = FrameStack(stack_size=4)
    
    episodes = []

    curr_map = None
    env = None

    for episode in  range(NUM_EPISODES):
        target_map = map_for_episode(episode, NUM_EPISODES)

        if env is None or target_map != curr_map:
            if env is not None:
                env.close()

            collect_config = dict(base_config)
            collect_config["map"] = target_map
            collect_config["start_seed"] = episode * 20
            collect_config["num_scenarios"] = 20

            env = MetaDriveEnvWrapper(collect_config)
            curr_map = target_map

        if episode % RESTART_EVERY == 0 and episode > 0:
            print(f"Preventivni restart env-a na epizodi {episode}")
            env.close()
            env = MetaDriveEnvWrapper(collect_config)

        result = reset_with_timeout(env);

        if result is None:
            env.close()
            env = MetaDriveEnvWrapper(collect_config)
            result = reset_with_timeout(env)

            if result is None:
                raise RuntimeError("Neuspesan reset nakon restartovanja env-a")
            
        raw_obs, info = result
        stacked_obs = frame_stack.reset(raw_obs)

        done = False

        episode_steps = 0

        total_reward = 0.0

        obs_list = [round_obs(stacked_obs, decimals=4)]
        actions = []
        rewards = []
        dones = []
        infos = []

        while not done:

            action = env.engine.get_policy(env.agent.id).act()
            norm_steering, norm_throttle = normalize_action(action[0], action[1])

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_state = frame_stack.step(next_obs)
            done = terminated or truncated

            obs_list.append(round_obs(next_state, decimals=4))
            actions.append([round(norm_steering, 5), round(norm_throttle, 5)])
            rewards.append(round(float(reward), 5))
            dones.append(bool(done))
            infos.append({
                "velocity": round(float(info.get("velocity", 0.0)), 5),
                "arrive_dest": bool(info.get("arrive_dest", False)),
                "crash": bool(info.get("crash", False)),
                "out_of_road": bool(info.get("out_of_road", False)),
            })

            stacked_obs = next_state
            total_reward += reward
            episode_steps += 1

        episodes.append({
            "episode_id": episode,
            "map": curr_map,
            "total_reward": round(float(total_reward), 5),
            "obs": obs_list,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "infos": infos
        })

        print(f"[Dataset]"
              f"Episode {episode+1:03d}"
              f" Steps {episode_steps:04d}"
              f" Reward {total_reward:.2f}",
              flush=True)
        
    
    print(episodes[0].keys())
    all_obs = np.array([obs for ep in episodes for obs in ep["obs"][:-1]], dtype=np.float32)
    all_next_obs = np.array([obs for ep in episodes for obs in ep["obs"][1:]], dtype=np.float32)
    all_actions = np.array([a for ep in episodes for a in ep["actions"]], dtype=np.float32)
    all_rewards = np.array([r for ep in episodes for r in ep["rewards"]], dtype=np.float32)
    all_dones = np.array([d for ep in episodes for d in ep["dones"]], dtype=bool)
    all_maps = np.array([ep["map"] for ep in episodes for _ in ep["actions"]], dtype="U4")

    np.savez_compressed(EXPERT_DATASET,
        obs=all_obs,
        next_obs=all_next_obs,
        actions=all_actions,
        rewards=all_rewards,
        dones=all_dones,
        maps=all_maps,
    )

if __name__ == "__main__":
    main()