import torch
import os

from enviornments.metadrive_env import MetaDriveEnvWrapper
from enviornments.action_mapper import ActionMapper
from agents.dqn_agent import DQNAgent
from agents.epsilon_scheduler import EpsilonScheduler
from replay.expert_replay_buffer import ExpertReplayBuffer
from training.trainer import Trainer
from training.evaluator import Evaluator
from training.checkpoint_manager import CheckpointManager
from utils.logger import Logger
from configs.env_config import *

TARGET_SUCCESS = 0.90
EVAL_FREQ = 40
MAX_EPISODES = 2000

def main():
    env = MetaDriveEnvWrapper(dict(ENV_CONFIG))
    env.reset()
    obs_size = env.obs_size * FRAME_STACK

    epsilon_scheduler = EpsilonScheduler(**EPSILON_CONFIG)
    num_actions = ActionMapper().num_actions()
    agent = DQNAgent(
        input_size = obs_size,
        num_actions = num_actions,
        epsilon_scheduler=epsilon_scheduler
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.online_net.to(device)
    agent.target_net.to(device)

    optimizer = torch.optim.Adam(agent.online_net.parameters(), lr=TRAIN_CONFIG["lr"])

    if os.path.exists(BC_CHECKPOINT_STRAIGHT):
        bc_state = torch.load(BC_CHECKPOINT_STRAIGHT, map_location="cpu", weights_only=True)
        agent.online_net.load_state_dict(bc_state)
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        print(f"[BC] Ucitan checkpoint: {BC_CHECKPOINT_STRAIGHT}")

    checkpoint_manager = CheckpointManager()

    replay_buffer = ExpertReplayBuffer(
        capacity = TRAIN_CONFIG["replay_capacity"],
        expert_dataset_path = EXPERT_DATASET,
        num_actions = num_actions,
        expert_ratio = EXPERT_RATIO,
        map_filter={"SSSS"}
    )

    logger = Logger(log_dir="logs")

    trainer = Trainer(
        env=env, optimizer=optimizer, agent=agent,
        replay_buffer=replay_buffer, config=TRAIN_CONFIG,
        logger=logger, scheduler=None
    )
    evaluator = Evaluator(env, agent, logger)

    trainer._last_map = "SSSS"
    straight_config = dict(ENV_CONFIG)
    straight_config["map"] = "SSSS"
    straight_config["horizon"] = 500
    straight_config["num_scenarios"] = 20
    straight_config["traffic_density"] = 0.0
    trainer.env.close()
    trainer.env = MetaDriveEnvWrapper(straight_config)
    replay_buffer.expert_ratio = EXPERT_RATIO

    best_success = 0.0
    episode = 0

    try:
        for episode in range(MAX_EPISODES):
            trainer.run_episode(episode)

            if (episode+1) % EVAL_FREQ == 0:
                trainer.env.close()
                trainer._last_map = None

                results = evaluator.evaluate_maps(
                    maps=["SSSS"],
                    episodes_per_map=10
                )

                success = results["SSSS"]["success_rate"]
                print(f"[EVAL] SSSS Success = {success:.2f}")

                trainer.env = MetaDriveEnvWrapper(straight_config)
                trainer._last_map = "SSSS"
                evaluator.env = trainer.env

                if success > best_success:
                    best_success = success
                    checkpoint_manager.save(
                        "checkpoints/best_straight.pt",
                        agent, optimizer,
                        trainer.global_step, episode
                    )

                    print(f"[Checkpoint] Novi best_straight.pt: {success:.2f}")

                if success >= TARGET_SUCCESS:
                    print(f"[Phase 1] Cilj dostignut ({success:.2f}). Zaustavljam")
                    break

    except KeyboardInterrupt:
        print("Prekid od strane korisnika")

    finally:
        checkpoint_manager.save(
            "checkpoints/straight_final.pt",
            agent, optimizer, trainer.global_step, episode
        )

        trainer.env.close()
        logger.close()

if __name__ == "__main__":
    main()