import torch
import os
from enviornments.metadrive_env import MetaDriveEnvWrapper  
from agents.dqn_agent import DQNAgent
from agents.epsilon_scheduler import EpsilonScheduler
from replay.expert_replay_buffer import ExpertReplayBuffer
from training.trainer import Trainer
from training.evaluator import Evaluator
from training.checkpoint_manager import CheckpointManager
from utils.logger import Logger
from configs.dqn_configs import *
from configs.env_config import *

def main():
    env = MetaDriveEnvWrapper(ENV_CONFIG)

    env.reset()
    obs_size = env.obs_size * FRAME_STACK

    epsilon_scheduler = EpsilonScheduler(**EPSILON_CONFIG)

    agent = DQNAgent(
        input_size=obs_size, num_actions=env.num_actions(), 
        epsilon_scheduler=epsilon_scheduler
    )

    if os.path.exists(BC_CHECKPOINT):
        agent.online_net.load_state_dict(
            torch.load(
                BC_CHECKPOINT, map_location=torch.device("cpu"), weights_only=True
            )
        )

        agent.target_net.load_state_dict(agent.online_net.state_dict())

    optimizer = torch.optim.Adam(agent.online_net.parameters(), lr = TRAIN_CONFIG["lr"])
    scheduler = None

    replay_buffer = ExpertReplayBuffer(
        capacity=TRAIN_CONFIG["replay_capacity"],
        expert_dataset_path=EXPERT_DATASET,
        num_actions=env.num_actions(),
        expert_ratio=EXPERT_RATIO
    )

    logger = Logger(log_dir="logs")

    checkpoint_manager = CheckpointManager()

    if RESUME_PATH and os.path.exists(RESUME_PATH):
        checkpoint_manager.load(RESUME_PATH, agent, optimizer)

    trainer = Trainer(
        env=env, agent=agent, replay_buffer=replay_buffer, optimizer=optimizer, config=TRAIN_CONFIG, logger=logger, scheduler=scheduler
    )

    eval_env = env
    evaluator = Evaluator(eval_env, agent, logger)

    try: 
        for episode in range(TRAIN_CONFIG["num_episodes"]):
            trainer.set_map(episode);
            trainer.run_episode(episode)

            if (episode+1) % EVAL_FREQ == 0:
                evaluator.evaluate(num_episodes=20)

            if(episode+1) % CHECKPOINT_FREQ == 0:
                path = os.path.join("checkpoints", f"ep_{episode+1}.pt")

                checkpoint_manager.save(path, agent, optimizer, trainer.global_step, episode+1)
    except KeyboardInterrupt:
        print("Prekid treninga od strane korisnika.")
    
    except Exception as e:
        import traceback
        print(f"Greska na epizodi {episode+1}:")
        traceback.print_exc()

    finally:

        checkpoint_manager.save(
            "checkpoints/final.pt", agent, optimizer, trainer.global_step, episode
        )

        env.close()
        logger.close()

if __name__ == "__main__":
    main()