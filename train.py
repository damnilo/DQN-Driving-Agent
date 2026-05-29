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
    env = MetaDriveEnvWrapper(dict(ENV_CONFIG))

    env.reset()
    obs_size = env.obs_size * FRAME_STACK

    epsilon_scheduler = EpsilonScheduler(**EPSILON_CONFIG)

    agent = DQNAgent(
        input_size=obs_size, num_actions=env.num_actions(), 
        epsilon_scheduler=epsilon_scheduler
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.online_net.to(device)
    agent.target_net.to(device)

    optimizer = torch.optim.Adam(agent.online_net.parameters(), lr = TRAIN_CONFIG["lr"])

    if os.path.exists(BC_CHECKPOINT):
        try:
            bc_state = torch.load(
                BC_CHECKPOINT,
                map_location=torch.device("cpu"),
                weights_only=True
            )
            agent.online_net.load_state_dict(bc_state)
            agent.target_net.load_state_dict(agent.online_net.state_dict())
            print(f"[BC] Ucitan checkpoint: {BC_CHECKPOINT}")
        except RuntimeError as e:
            print(f"[BC] Preskacem nekompatibilan checkpoint: {BC_CHECKPOINT}")
            print(e)
    
    checkpoint_manager = CheckpointManager()

    replay_buffer = ExpertReplayBuffer(
        capacity=TRAIN_CONFIG["replay_capacity"],
        expert_dataset_path=EXPERT_DATASET,
        num_actions=env.num_actions(),
        expert_ratio=EXPERT_RATIO
    )

    logger = Logger(log_dir="logs")

    trainer = Trainer(
        env=env, agent=agent, replay_buffer=replay_buffer, optimizer=optimizer, config=TRAIN_CONFIG, logger=logger, scheduler=None
    )

    evaluator = Evaluator(env, agent, logger)

    episode = 0
    try: 
        if PRETRAIN_DQN_STEPS > 0:
            print(f"[Pre-training] {PRETRAIN_DQN_STEPS} DQN koraka...")
            for _ in range(PRETRAIN_DQN_STEPS):
                batch = replay_buffer.sample(TRAIN_CONFIG["batch_size"])
                trainer.train_step(batch)
            print("[Pre-training] Zavrseno.")
        else:
            print("[Pre-training] Preskoceno (PRETRAIN_DQN_STEPS=0, cuva BC)")

        best_composite_score = -float("inf")
        best_curve_success = 0.0

        for episode in range(TRAIN_CONFIG["num_episodes"]):
            changed = trainer.set_map(episode)
            if changed: 
                evaluator.env = trainer.env

            trainer.run_episode(episode)

            if episode > 0 and (episode + 1) % EVAL_FREQ == 0:
                # MetaDrive dozvoljava samo jedan engine — zatvori trening env pre eval-a
                trainer.env.close()
                trainer._last_map = None

                results = evaluator.evaluate_maps(
                    maps=EVAL_MAPS,
                    episodes_per_map=EVAL_EPISODES_PER_MAP,
                )

                # Sledeća epizoda ponovo kreira env kroz set_map()

                curr_map = results[env.get_map()]

                composite_score = (
                    CHECKPOINT_STRAIGHT_WEIGHT * curr_map["success_rate"]
                )

                print(
                    f"[Eval] Current Map success={curr_map['success_rate']:.2f} "
                    f"reward={curr_map['avg_reward']:.2f}"
                )

                should_save = composite_score > best_composite_score

                if should_save:
                    best_composite_score = composite_score

                    checkpoint_manager.save(
                        "checkpoints/best_model.pt",
                        agent,
                        optimizer,
                        trainer.global_step,
                        episode + 1,
                    )

            if (episode + 1) % CHECKPOINT_FREQ == 0:
                path = os.path.join("checkpoints", f"ep_{episode + 1}.pt")
                checkpoint_manager.save(path, agent, optimizer, trainer.global_step, episode + 1)

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
