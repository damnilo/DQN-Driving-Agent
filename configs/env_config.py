ENV_CONFIG = {
        "use_render": False,
        "manual_control": False,
        "traffic_density": 0.0,
        "num_scenarios": 20,
        "start_seed": 0,
        "map": "SSSS",
        "image_observation": False,
        # "daytime": random.choice(["08:00", "12:00", "17:30", "20:00"]),
        "accident_prob": 0.0,   
        "on_continuous_line_done": True, 
        "crash_vehicle_done": True,      # Sudar sa drugim vozilom
        "crash_object_done": True,       # Sudar sa objektom (ogradom, čunjem)
        "out_of_road_done": True,  
}

FRAME_STACK = 4

TRAIN_CONFIG = {
    "batch_size": 64,
    "tau": 0.005,
    "gamma" : 0.99,
    "lr": 1e-4,
    "replay_capacity": 150_000,
    "min_replay_size": 1_000,
    "target_update_freq": None
}

CURVE_TRAIN_CONFIG = {
    "batch_size": 32,
    "tau": 0.005,
    "gamma": 0.99,
    "lr": 5e-5,
    "replay_capacity": 150_000,
    "min_replay_size": 2_000,
    "target_update_freq": None
}

EXPERT_RATIO = 0.0
EXPERT_RATIO_CURVE = 0.10
# Existing expert_dataset.npz was labeled with the lane-local reward. Leave this at 0
# until collect_idm.py is run again, then raise it (0.10-0.20) to mix those transitions.
EXPERT_RATIO_RANDOM = 0.0
EXPERT_DATASET = "dataset/expert_dataset.npz"

RANDOM_EPSILON_CONFIG = {
    "start": 0.08,
    "end": 0.02,
    "decay": 300_000,
    "warmup_steps": 2_000
}

EVAL_EPISODES_PER_MAP = 10

CURVE_EPSILON_CONFIG = {
    "start": 0.45,
    "end": 0.03,
    "decay": 450_000,
    "warmup_steps": 1_000
}

EPSILON_CONFIG = {
    "start": 0.50,
    "end": 0.05,
    "decay": 200_000,
    "warmup_steps": 5_000
}

CHECKPOINT_FREQ = 100
EVAL_FREQ = 40
BC_CHECKPOINT_STRAIGHT = "checkpoints/bc_pretrain_straight.pt"

CURVE_BC_CONFIG = {
    "epochs": 80,
    "batch_size": 256,
    "lr_scale": 0.3,
    "lr": 2e-4,
    "val_split": 0.15,
    "patience": 20,
    "clip_grad": 0.5
}

STRAIGHT_BC_CONFIG = {
    "epochs": 80,
    "batch_size": 64,
    "lr": 2e-4,
    "val_split": 0.15,
    "patience": 20,
    "clip_grad": 0.5
}


class _RouteBlockDist:
    """MetaDrive block distribution used while `map` is an integer block count."""

    MIN_LANE_NUM = 1
    MAX_LANE_NUM = 5
    DISTRIBUTION = {}

    @classmethod
    def all_blocks(cls, version="v2"):
        return list(cls.DISTRIBUTION)

    @classmethod
    def block_probability(cls, version="v2"):
        weights = list(cls.DISTRIBUTION.values())
        total = sum(weights) or 1.0
        return [weight / total for weight in weights]

    @classmethod
    def get_block(cls, block_id, version="v2"):
        from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
        return PGBlockDistConfig.get_block(block_id, version)


class StraightCurveBlockDist(_RouteBlockDist):
    DISTRIBUTION = {"Straight": 0.35, "Curve": 0.65}


class JunctionBlockDist(_RouteBlockDist):
    DISTRIBUTION = {
        "Straight": 0.15,
        "Curve": 0.35,
        "StdInterSection": 0.25,
        "StdTInterSection": 0.25,
    }


# (start_episode, block distribution or None for MetaDrive's default mix)
RANDOM_CURRICULUM = [
    (0, StraightCurveBlockDist),
    (400, JunctionBlockDist),
    (1000, None),
]