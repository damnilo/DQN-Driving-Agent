import torch
import os
from enviornments.metadrive_env import MetaDriveEnvWrapper
from enviornments.action_mapper import ActionMapper
from agents.dqn_agent import DQNAgent
from utils.frame_stack import FrameStack
from agents.epsilon_scheduler import EpsilonScheduler
from configs.env_config import ENV_CONFIG, FRAME_STACK

MAPS_TEST = ["SSSS","SCSC", "CSCS", "CCCC", 4]
EPISODES_PER_MAP = 3

def get_checkpoint_for_map(map_name):
    if map_name == "SSSS":
        return "checkpoints/best_straight.pt"
    elif map_name == 4:
        return "checkpoints/best_random.pt"
    else:
        return "checkpoints/best_curve.pt"

def run_episode(env, agent, render=True):
    frame_stack = FrameStack(stack_size=FRAME_STACK)

    obs, info = env.reset()
    obs = frame_stack.reset(obs)

    am = ActionMapper()
    done = False
    total_reward = 0.0
    step = 0
    max_heading_err = 0.0
    jerk_sum = 0.0
    prev_action = None

    while not done:
        action = agent.select_action(obs, step=0, training=False)

        next_obs, reward, term, trunc, info = env.step(action)
        obs = frame_stack.step(next_obs)
        done = term or trunc
        total_reward += reward
        step += 1

        heading_err = abs(info.get("heading_error", 0.0))
        max_heading_err = max(heading_err, max_heading_err)

        if prev_action is not None:
            prev_s = am.map(prev_action)[0]
            curr_s = am.map(action)[0]
            jerk_sum += abs(curr_s - prev_s)
        prev_action = action

        if step % 100 == 0:
            print(
                f"Step {step:4d} | "
                f"Reward={reward:6.2f} | "
                f"Heading Error={heading_err:.3f} | "
                f"Lateral={info.get('lateral_offset', 0.0):.3f} | "
                f"Speed={info.get('velocity', 0.0):.1f} | "
                f"Action={action}"
            )

    reason = "UNKNOWN"
    if info.get("arrive_dest", False): reason = "SUCCESS"
    elif info.get("crash", False): reason = "CRASH"
    elif info.get("out_of_road", False): reason = "OUT_OF_ROAD"
    elif info.get("max_step", False): reason = "MAX_STEP"

    avg_jerk = jerk_sum / max(step, 1)

    return {
        "reason": reason,
        "total_reward": total_reward,
        "steps": step,
        "max_heading_err": max_heading_err,
        "avg_steering_jerk": avg_jerk
    }

def main():

    temp_config = dict(ENV_CONFIG)
    temp_env = MetaDriveEnvWrapper(temp_config)
    obs, _ = temp_env.reset()
    obs_size = len(obs) * FRAME_STACK
    action_dim = temp_env.num_actions()
    temp_env.close()

    epsilon_scheduler = EpsilonScheduler(start=0.0, end=0.0, decay=1, warmup_steps=0)
    agent = DQNAgent(obs_size, action_dim, epsilon_scheduler)

    curr_checkpoint = None

    for map_name in MAPS_TEST:
        print(f"\n{'='*50}")
        print(f"MAP: {map_name}")
        print(f"{'='*50}")

        checkpoint_path = get_checkpoint_for_map(map_name)

        if checkpoint_path != curr_checkpoint:
            if not os.path.exists(checkpoint_path):
                print(f"[WARN] Checkpoint not found: {checkpoint_path}, skipping map.")
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            agent.online_net.load_state_dict(checkpoint["online_net"])
            agent.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["online_net"]))
            agent.online_net.eval()
            curr_checkpoint = checkpoint_path
            print(f"Loaded: {checkpoint_path}")

        successes = 0

        for ep in range(1):
            config = dict(ENV_CONFIG)
            config["use_render"] = True
            config["map"] = map_name
            config["start_seed"] = ep * 10
            config["traffic_density"] = 0.0
            env = MetaDriveEnvWrapper(config)

            print(f"\n Episode {ep+1}/{EPISODES_PER_MAP}")
            result = run_episode(env, agent)
            env.close()

            if "SUCCESS" == result["reason"]:
                successes += 1

            print(
                f" -> {result['reason']} | "
                f"Reward = {result['total_reward']:.1f} | "
                f"Steps = {result['steps']} | "
                f"Max_heading_error = {result['max_heading_err']:.3f}"
                f"Avg_steer_jerk = {result['avg_steering_jerk']:.4f}"
            )

        print(f"\n [{map_name}] Success rate: {successes} / {EPISODES_PER_MAP}")

if __name__ == "__main__":
    main()