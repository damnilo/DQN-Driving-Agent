import torch
from enviornments.metadrive_env import MetaDriveEnvWrapper
from agents.dqn_agent import DQNAgent
from utils.frame_stack import FrameStack
from agents.epsilon_scheduler import EpsilonScheduler
from configs.env_config import ENV_CONFIG

def main():

    config = dict(ENV_CONFIG)
    config["use_render"] = True
    config["map"] = "SCSC"
    config["start_seed"] = 0
    config["traffic_density"] = 0.1
    env = MetaDriveEnvWrapper(config)
    frame_stack = FrameStack(stack_size=4)
    obs, info = env.reset()
    obs = frame_stack.reset(obs)
    base_state_dim = env.obs_size if env.obs_size is not None else len(obs)
    state_dim = base_state_dim * 4
    action_dim = env.num_actions()
    epsilon_scheduler = EpsilonScheduler(start=0.0, end=0.0, decay=1, warmup_steps=0)
    agent = DQNAgent(state_dim, action_dim, epsilon_scheduler)
    checkpoint_path = "checkpoints/final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    agent.online_net.load_state_dict(checkpoint["online_net"])

    if "target_net" in checkpoint:
        agent.target_net.load_state_dict(checkpoint["target_net"])
    else:
        agent.target_net.load_state_dict(agent.online_net.state_dict())
    agent.online_net.eval()
    done = False
    total_reward = 0
    global_step = 0

    while not done:

        with torch.no_grad():

            action = agent.select_action(
                obs, global_step, training=False
            )

        next_obs, reward, terminated, truncated, info = env.step(action)
        obs = frame_stack.step(next_obs)
        done = terminated or truncated
        total_reward += reward

        if done:
            print(f"Voznja zavrsena! Uspesnost: {info.get('arrive_dest', False)}")
            print(f"Ukupna nagrada: {total_reward}")
            break

    env.close()

if __name__ == "__main__":
    main()