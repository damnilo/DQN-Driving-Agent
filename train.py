import torch
import os
from enviornments.metadrive_env import MetaDriveEnvWrapper  
from agents.dqn_agent import DQNAgent
from agents.epsilon_scheduler import EpsilonScheduler
from replay.replay_buffer import ReplayBuffer
from replay.expert_replay_buffer import ExpertReplayBuffer
from training.trainer import Trainer
from training.evaluator import Evaluator
from training.checkpoint_manager import CheckpointManager
from utils.logger import Logger
from configs.dqn_configs import *

FRAME_STACK = 4
ENV_CONFIG = {
        "use_render": False,
        "manual_control": False,
        "traffic_density": 0.1,
        "num_scenarios": 100,
        "start_seed": 0,
        "map": 4,
        # "daytime": random.choice(["08:00", "12:00", "17:30", "20:00"]),
        "accident_prob": 0.0,
        "vehicle_config": {
            "show_lidar": True,
            # "vehicle_model": "default", # Neki MetaDrive verzije zahtevaju specifične modele, ostavi default ako pravi problem
        },
        # ISPRAVLJENI KLJUČEVI:
        "on_continuous_line_done": True, 
        "crash_vehicle_done": True,      # Sudar sa drugim vozilom
        "crash_object_done": True,       # Sudar sa objektom (ogradom, čunjem)
        "out_of_road_done": True,  
}

TRAIN_CONFIG = {
    "num_episodes": 500,
    "batch_size": 64,
    "gamma" : 0.99,
    "lr": 3e-5,
    "replay_capacity": 150_000,
    "min_replay_size": 2_000,
    "target_update_freq": 300
}

EXPERT_RATIO = 0.3
EXPERT_DATASET = "dataset/expert_dataset.json"

EPSILON_CONFIG = {
    "start": 1.0,
    "end": 0.02,
    "decay": 350_000,
    "warmup_steps": 500
}

CHECKPOINT_FREQ = 100
EVAL_FREQ = 100
RESUME_PATH = None
BC_CHECKPOINT = "checkpoints/bc_pretrain.pt"

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
                evaluator.evaluate(num_episodes=5)

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
        evaluator.evaluate(num_episodes=30)
        eval_env.close()
        logger.close()

if __name__ == "__main__":
    main()