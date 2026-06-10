import torch
import os
import random
import numpy as np
from enviornments.metadrive_env import MetaDriveEnvWrapper
from enviornments.action_mapper import ActionMapper 
from agents.dqn_agent import DQNAgent
from agents.epsilon_scheduler import EpsilonScheduler
from replay.expert_replay_buffer import ExpertReplayBuffer
from training.curve_trainer import CurveTrainer
from training.evaluator import Evaluator
from training.checkpoint_manager import CheckpointManager
from utils.logger import Logger
from configs.env_config import *

MAX_EPISODES = 4000
NUM_ACTIONS = ActionMapper().num_actions()

def main():
    env = MetaDriveEnvWrapper(dict(ENV_CONFIG))

    env.reset()
    obs_size = env.obs_size * FRAME_STACK
    env.close()

    epsilon_scheduler = EpsilonScheduler(**CURVE_EPSILON_CONFIG)

    agent = DQNAgent(
        input_size=obs_size,
        num_actions=NUM_ACTIONS,
        epsilon_scheduler=epsilon_scheduler
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.online_net.to(device)
    agent.target_net.to(device)

    optimizer = torch.optim.Adam(agent.online_net.parameters(), lr = CURVE_TRAIN_CONFIG["lr"])

    straight_checkpoint = "checkpoints/best_curve.pt"
    if not os.path.exists(straight_checkpoint):
        raise FileNotFoundError("Nema best_curve.pt. Prvo pokreni train_straight.py")

    ckpt = torch.load(straight_checkpoint, map_location="cpu", weights_only=True)
    agent.online_net.load_state_dict(ckpt["online_net"])
    agent.target_net.load_state_dict(ckpt["target_net"])

    print("[Phase 2] Ucitan best_curve.pt kao polazna tacka")
    
    checkpoint_manager = CheckpointManager()

    replay_buffer = ExpertReplayBuffer(
        capacity=CURVE_TRAIN_CONFIG["replay_capacity"],
        expert_dataset_path=EXPERT_DATASET,
        num_actions=NUM_ACTIONS,
        expert_ratio=EXPERT_RATIO_CURVE,
        map_filter={"4"}
    )

    logger = Logger(log_dir="logs")

    initial_config = dict(ENV_CONFIG)
    initial_config["map"] = 4
    initial_config["traffic_density"] = 0.0
    initial_config["horizon"] = 1200
    initial_config["num_scenarios"] = 50

    train_env=MetaDriveEnvWrapper(initial_config)

    trainer = CurveTrainer(
        env=train_env, agent=agent, replay_buffer=replay_buffer, optimizer=optimizer, config=CURVE_TRAIN_CONFIG, logger=logger
    )
    trainer._last_map = 4

    evaluator = Evaluator(env, agent, logger)

    best_curve_score = 0.0
    episode = 0

    try:

        for episode in range(MAX_EPISODES):

            trainer.run_episode(episode)

            if (episode + 1) % EVAL_FREQ == 0:
                trainer.env.close()
                trainer._last_map = None

                results = evaluator.evaluate_maps(
                    maps=4,
                    episodes_per_map=EVAL_EPISODES_PER_MAP,
                )

                curve_score = results["4"]["success_rate"]

                if curve_score > best_curve_score:
                    best_curve_score = curve_score
                    checkpoint_manager.save(
                        "checkpoints/best_random.pt",
                        agent, optimizer, trainer.global_step, episode
                    )
                    print(f"[Checkpoint] Novi best_random.pt: {curve_score:.3f}")

    except KeyboardInterrupt:
        print("Prekid treninga od strane korisnika.")
    
    except Exception as e:
        import traceback
        print(f"Greska na epizodi {episode + 1}:")
        traceback.print_exc()

    finally:

        checkpoint_manager.save(
            "checkpoints/final.pt", agent, optimizer, trainer.global_step, episode
        )

        trainer.env.close()
        logger.close()

if __name__ == "__main__":
    main()
