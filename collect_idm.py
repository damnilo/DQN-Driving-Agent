import json
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

def reset_with_timeout(env, timeout=30):
    # MetaDrive / Panda3D must be reset from the main interpreter thread.
    # The previous threaded approach caused `signal only works in main thread`.
    try:
        return env.reset()
    except Exception as e:
        # If reset gets stuck or fails, let the caller handle restart.
        print(f"env.reset() failed: {e}")
        return None
    
def map_for_episode(episode, num_episodes):
    ratio = episode / max(1, num_episodes)

    if ratio < 0.1:
        return "SSSS"
    elif ratio < 0.35:
        return "SCSC"
    elif ratio < 0.60:
        return "CSCS"
    elif ratio < 0.80:
        return "CCCC"
    return 4

def make_json_safe(obj):

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    if isinstance(obj, dict):
        return {
            k: make_json_safe(v)
            for k, v in obj.items()
        }
    
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    
    return obj

def main():

    os.makedirs("dataset", exist_ok=True)

    base_config = dict(ENV_CONFIG)
    base_config["agent_policy"] = ExpertPolicy
    base_config["traffic_density"] = 0.0
    base_config["accident_prob"] = 0.0
    
    frame_stack = FrameStack(stack_size=4)
    dataset = []

    curr_map = None
    env = None

    for episode in  range(NUM_EPISODES):
        target_map = map_for_episode(episode, NUM_EPISODES)

        if env is None or target_map != curr_map:
            if env is not None:
                env.close()

            collect_config = dict(base_config)
            collect_config["map"] = target_map
            collect_config["start_seed"] = episode * 50
            collect_config["num_scenarios"] = 50

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

        while not done:

            action = env.engine.get_policy(env.agent.id).act()
            norm_steering, norm_throttle = normalize_action(action[0], action[1])

            next_obs, reward, terminated, truncated, next_info = env.step_continuous(action)
            next_state = frame_stack.step(next_obs)
            done = terminated or truncated


            dataset.append({
                "episode_id": episode,
                "step": episode_steps,
                "map": curr_map,
                "observation": stacked_obs.tolist(),
                "next_observation": next_state.tolist(),
                "action_steering": norm_steering,
                "action_throttle": norm_throttle,
                "reward": float(reward),
                "done": bool(done),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "velocity": float(next_info.get("velocity", 0.0)),
                "heading_error": float(next_info.get("heading_error", 0.0)),
                "lateral_offset": float(next_info.get("lateral_offset", 0.0)),
                "navigation_command": next_info.get("navigation_command", "IDLE"),
                "arrive_dest": bool(next_info.get("arrive_dest", False)),
                "crash": bool(next_info.get("crash", False)),
                "out_of_road": bool(next_info.get("out_of_road", False)),
            })

            stacked_obs = next_state
            total_reward += reward
            episode_steps += 1

        print(f"[Dataset]"
              f"Episode {episode+1:03d}"
              f" Steps {episode_steps:04d}"
              f" Reward {total_reward:.2f}",
              flush=True)
        
    
    tmp_path = EXPERT_DATASET + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(dataset), f)

    os.replace(tmp_path, EXPERT_DATASET)
    print(f"[Dataset] Sacuvano {len(dataset)} tranzicija u {EXPERT_DATASET}")

    env.close()

if __name__ == "__main__":
    main()