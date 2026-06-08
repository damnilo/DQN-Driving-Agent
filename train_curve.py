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

CURVE_MAPS = ["SCSC", "CSCS", "CCCC"]
EVAL_FREQ = 40
MAX_EPISODES = 4000

CURRICULUM_STAGES = [
    (0, ["SCSC", "CSCS"], [0.50, 0.50], 0.0),
    (800, ["CCCC", "SCSC", "CSCS"], [0.33, 0.33, 0.34], 0.0),
]

# Straight retention parameters
STRAIGHT_RETENTION_THRESHOLD = 0.85
STRAIGHT_BOOST_AMOUNT = 0.12
STRAIGHT_BOOST_DURATION = 500
NUM_ACTIONS = ActionMapper().num_actions()

def pick_curve_map(episode, straight_boost=0.0):
    stage = CURRICULUM_STAGES[0]
    for s in CURRICULUM_STAGES:
        if episode >= s[0]:
            stage = s
    _, maps, weights, density = stage

    # if we have a temporary straight boost, add it to SSSS weight and renormalize
    weights = list(weights)
    if straight_boost > 0.0 and "SSSS" in maps:
        idx = maps.index("SSSS")
        weights[idx] = max(0.0, weights[idx] + straight_boost)
        # renormalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

    chosen = random.choices(maps, weights=weights)[0]
    return chosen, density

def _horizon_for_map(map):
    if map in ["SCSC", "CSCS"]:
        return 1000
    if map == "CCCC":
        return 1400
    
    return 800

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

    straight_checkpoint = "checkpoints/best_straight.pt"
    if not os.path.exists(straight_checkpoint):
        raise FileNotFoundError("Nema best_straight.pt. Prvo pokreni train_straight.py")

    ckpt = torch.load(straight_checkpoint, map_location="cpu", weights_only=True)
    agent.online_net.load_state_dict(ckpt["online_net"])
    agent.target_net.load_state_dict(ckpt["target_net"])

    print("[Phase 2] Ucitan best_straight.pt kao polazna tacka")
    
    checkpoint_manager = CheckpointManager()

    replay_buffer = ExpertReplayBuffer(
        capacity=CURVE_TRAIN_CONFIG["replay_capacity"],
        expert_dataset_path=EXPERT_DATASET,
        num_actions=NUM_ACTIONS,
        expert_ratio=EXPERT_RATIO_CURVE,
        map_filter={"SCSC", "CSCS", "CCCC", "4"}
    )

    logger = Logger(log_dir="logs")

    # dynamic straight boost state
    straight_boost_until = 0
    straight_boost = 0.0

    initial_map, initial_density = pick_curve_map(0, straight_boost)
    initial_config = dict(ENV_CONFIG)
    initial_config["map"] = initial_map
    initial_config["traffic_density"] = initial_density
    initial_config["horizon"] = _horizon_for_map(initial_map)
    initial_config["num_scenarios"] = 20

    train_env=MetaDriveEnvWrapper(initial_config)

    trainer = CurveTrainer(
        env=train_env, agent=agent, replay_buffer=replay_buffer, optimizer=optimizer, config=CURVE_TRAIN_CONFIG, logger=logger
    )
    trainer._last_map = initial_map

    evaluator = Evaluator(env, agent, logger)

    best_curve_score = 0.0
    episode = 0

    try:

        for episode in range(MAX_EPISODES):
            # apply temporary straight boost if active
            if episode < straight_boost_until:
                cur_boost = straight_boost
            else:
                cur_boost = 0.0

            target_map, density = pick_curve_map(episode, cur_boost)

            if trainer._last_map != target_map:
                trainer.env.close()
                curve_config = dict(ENV_CONFIG)
                curve_config["map"] = target_map
                curve_config["traffic_density"] = density
                curve_config["horizon"] = _horizon_for_map(target_map)
                curve_config["num_scenarios"] = 20
                trainer.env = MetaDriveEnvWrapper(curve_config)
                trainer._last_map = target_map
                evaluator.env = trainer.env
                print(f"[Curriculum] Ep {episode}: -> {target_map}")

            trainer.run_episode(episode)

            if (episode + 1) % EVAL_FREQ == 0:
                trainer.env.close()
                trainer._last_map = None

                results = evaluator.evaluate_maps(
                    maps=CURVE_MAPS,
                    episodes_per_map=EVAL_EPISODES_PER_MAP,
                )

                curve_score = sum(results.get(m, {}).get("success_rate", 0.0) for m in CURVE_MAPS) / len(CURVE_MAPS)

                print(
                    f"Curve composite = {curve_score:.3f} | "
                    + " | ".join(f"{m}={results.get(m, {}).get('success_rate', 0.0):.2f}" for m in CURVE_MAPS)
                )

                if curve_score > best_curve_score:
                    best_curve_score = curve_score
                    checkpoint_manager.save(
                        "checkpoints/best_curve.pt",
                        agent, optimizer, trainer.global_step, episode
                    )
                    print(f"[Checkpoint] Novi best_curve.pt: {curve_score:.3f}")

                next_map, next_density = pick_curve_map(episode+1, cur_boost)
                curve_config = dict(ENV_CONFIG)
                curve_config["map"] = next_map
                curve_config["traffic_density"] = next_density
                curve_config["horizon"] = _horizon_for_map(next_map)
                curve_config["num_scenarios"] = 20
                trainer.env = MetaDriveEnvWrapper(curve_config)
                trainer._last_map = next_map
                evaluator.env = trainer.env

    except KeyboardInterrupt:
        print("Prekid treninga od strane korisnika.")
    
    except Exception as e:
        import traceback
        print(f"Greska na epizodi {episode + 1}:")
        traceback.print_exc()

    finally:

        checkpoint_manager.save(
            "checkpoints/curve_final.pt", agent, optimizer, trainer.global_step, episode
        )

        trainer.env.close()
        logger.close()

if __name__ == "__main__":
    main()
