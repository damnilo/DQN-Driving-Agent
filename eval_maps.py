import torch
import os
from agents.dqn_agent import DQNAgent
from agents.epsilon_scheduler import EpsilonScheduler
from enviornments.metadrive_env import MetaDriveEnvWrapper
from configs.env_config import ENV_CONFIG, FRAME_STACK
from training.map_evaluator import MapEvaluator

def main():
    temp_env = MetaDriveEnvWrapper(dict(ENV_CONFIG))
    obs, _ = temp_env.reset()
    state_dim = len(obs) * FRAME_STACK
    action_dim = temp_env.num_actions()
    temp_env.close()

    epsilon_scheduler = EpsilonScheduler(start=0.0, end=0.0, decay=1, warmup_steps=0)

    agent = DQNAgent(input_size=state_dim, num_actions=action_dim, epsilon_scheduler=epsilon_scheduler)

    checkpoint_path = "checkpoints/final.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Nema checkpoints/final.pt. Prvo pokreni train.py do kraja."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    agent.online_net.load_state_dict(checkpoint["online_net"])
    agent.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["online_net"]))

    maps = ["SSSS", "SCSC", "CSCS", "SCSCS", "CSCSC", 3, 4]
    seeds = [0, 10, 20, 30, 40]

    evaluator = MapEvaluator(agent)
    evaluator.evaluate_maps(maps=maps, seeds=seeds, episodes_per_map=5)

if __name__ == "__main__":
    main()