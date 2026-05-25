import json
import os
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy
from enviornments.observation_builder import ObservationBuilder
from utils.frame_stack import FrameStack
from configs.env_config import *

NUM_EPISODES = int(TRAIN_CONFIG["num_episodes"] / 5)

RESTART_EVERY = 20

def reset_with_timeout(env, timeout=30):
    # MetaDrive / Panda3D must be reset from the main interpreter thread.
    # The previous threaded approach caused `signal only works in main thread`.
    try:
        return env.reset()
    except Exception as e:
        # If reset gets stuck or fails, let the caller handle restart.
        print(f"env.reset() failed: {e}")
        return None

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

    ENV_CONFIG["agent_policy"] = ExpertPolicy

    env = MetaDriveEnv(ENV_CONFIG)
    observation_builder = ObservationBuilder()

    frame_stack = FrameStack(stack_size=4)

    dataset = []

    for episode in  range(NUM_EPISODES):

        if episode % RESTART_EVERY == 0 and episode > 0:
            print(f"Preventivni restart env-a na epizodi {episode}")
            env.close()
            env = MetaDriveEnv(ENV_CONFIG)

        result = reset_with_timeout(env);

        if result is None:
            env.close()
            env = MetaDriveEnv(ENV_CONFIG)
            result = reset_with_timeout(env)

            if result is None:
                raise RuntimeError("Neuspesan reset nakon restartovanja env-a")
            
        raw_obs, info = result
        first_obs = observation_builder.build(env=env, raw_obs=raw_obs, info=info)
        stacked_obs = frame_stack.reset(first_obs)

        done = False

        episode_steps = 0

        total_reward = 0.0

        while not done:

            processed_obs = (observation_builder.build(
                env=env, raw_obs=raw_obs, info=info
            ))
            stacked_obs = frame_stack.step(processed_obs)

            action = env.engine.get_policy(env.agent.id).act()

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            dataset.append({
                "observation": processed_obs.tolist(),

                "action_steering": float(action[0]),

                "action_throttle": float(action[1])
            })

            raw_obs = next_obs
            total_reward += reward
            episode_steps += 1

        print(f"[Dataset]"
              f"Episode {episode+1:03d}"
              f" Steps {episode_steps:04d}"
              f" Reward {total_reward:.2f}")
        
    with open(EXPERT_DATASET, "w") as f:

        json.dump(make_json_safe(dataset), f)

    print()

    env.close()

if __name__ == "__main__":
    main()