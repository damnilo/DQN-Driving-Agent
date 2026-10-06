import os

import torch

from environment.metadrive_env import MetaDriveEnvWrapper
from environment.action_mapper import ActionMapper
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
TARGET_SUCCESS = 0.90
TRAIN_SEEDS = 50
EVAL_START_SEED = 50


def _load_curve_weights(path, agent):
    """Loads online and target weights from a CheckpointManager file or a raw state dict."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Nema {path}. Prvo pokreni train_curve.py")

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "online_net" in ckpt:
        online = ckpt["online_net"]
        target = ckpt.get("target_net", online)
    else:
        online = ckpt
        target = ckpt

    agent.online_net.load_state_dict(online)
    agent.target_net.load_state_dict(target)


def _stage_for(episode):
    stage = RANDOM_CURRICULUM[0]
    for candidate in RANDOM_CURRICULUM:
        if episode >= candidate[0]:
            stage = candidate
    return stage


def _make_random_env(block_dist):
    config = dict(ENV_CONFIG)
    config["map"] = 4
    config["traffic_density"] = 0.0
    config["horizon"] = 2000
    config["num_scenarios"] = TRAIN_SEEDS
    config["start_seed"] = 0
    if block_dist is not None:
        config["block_dist_config"] = block_dist
    return MetaDriveEnvWrapper(config)


def main():
    """Loads best_curve.pt, fine-tunes on procedural 4-block maps with a block curriculum,
    and evaluates on held-out seeds every EVAL_FREQ episodes."""

    best_score = 0.0
    episode = 0

    probe = MetaDriveEnvWrapper(dict(ENV_CONFIG))
    probe.reset()
    obs_size = probe.obs_size * FRAME_STACK
    probe.close()

    epsilon_scheduler = EpsilonScheduler(**RANDOM_EPSILON_CONFIG)

    agent = DQNAgent(
        input_size=obs_size,
        num_actions=NUM_ACTIONS,
        epsilon_scheduler=epsilon_scheduler
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.online_net.to(device)
    agent.target_net.to(device)

    optimizer = torch.optim.Adam(agent.online_net.parameters(), lr=CURVE_TRAIN_CONFIG["lr"])
    _load_curve_weights("checkpoints/best_curve.pt", agent)
    print("[Phase 3] Ucitan best_curve.pt kao polazna tacka")

    checkpoint_manager = CheckpointManager()

    expert_path = EXPERT_DATASET if EXPERT_RATIO_RANDOM > 0.0 else ""
    replay_buffer = ExpertReplayBuffer(
        capacity=CURVE_TRAIN_CONFIG["replay_capacity"],
        expert_dataset_path=expert_path,
        num_actions=NUM_ACTIONS,
        expert_ratio=EXPERT_RATIO_RANDOM,
        map_filter={"4"},
        expected_obs_size=obs_size,
    )

    logger = Logger(log_dir="logs")

    train_env = _make_random_env(RANDOM_CURRICULUM[0][1])
    trainer = CurveTrainer(
        env=train_env, agent=agent, replay_buffer=replay_buffer,
        optimizer=optimizer, config=CURVE_TRAIN_CONFIG, logger=logger
    )
    trainer._block_dist = RANDOM_CURRICULUM[0][1]
    print(f"[Curriculum] Ep 0: blokovi -> {RANDOM_CURRICULUM[0][1].__name__}")

    evaluator = Evaluator(probe, agent, logger)

    def use_stage(ep):
        _, block_dist = _stage_for(ep)
        changed = trainer._block_dist is not block_dist
        if trainer.env is not None and not changed:
            return
        if trainer.env is not None:
            trainer.env.close()
        trainer.env = _make_random_env(block_dist)
        evaluator.env = trainer.env
        if changed:
            trainer._epsilon_origin = trainer.global_step
            label = "default" if block_dist is None else block_dist.__name__
            print(f"[Curriculum] Ep {ep}: blokovi -> {label}")
        trainer._block_dist = block_dist

    try:
        for episode in range(MAX_EPISODES):
            use_stage(episode)
            trainer.run_episode(episode)

            if (episode + 1) % EVAL_FREQ == 0:
                trainer.env.close()
                trainer.env = None

                results = evaluator.evaluate_maps(
                    maps=[4],
                    episodes_per_map=EVAL_EPISODES_PER_MAP,
                )

                score = results[4]["success_rate"]
                print(
                    f"[Eval] held-out seeds {EVAL_START_SEED}+ "
                    f"success = {score:.3f}"
                )

                use_stage(episode + 1)

                if score > best_score:
                    best_score = score
                    checkpoint_manager.save(
                        "checkpoints/best_random.pt",
                        agent, optimizer, trainer.global_step, episode
                    )
                    print(f"[Checkpoint] Novi best_random.pt: {score:.3f}")

                if best_score >= TARGET_SUCCESS:
                    print("Dostignut zeljeni uspeh. Prekidam program")
                    break

    except KeyboardInterrupt:
        print("Prekid treninga od strane korisnika.")

    except Exception:
        import traceback
        print(f"Greska na epizodi {episode + 1}:")
        traceback.print_exc()

    finally:
        checkpoint_manager.save(
            "checkpoints/final.pt", agent, optimizer, trainer.global_step, episode
        )
        if trainer.env is not None:
            trainer.env.close()
        logger.close()


if __name__ == "__main__":
    main()
