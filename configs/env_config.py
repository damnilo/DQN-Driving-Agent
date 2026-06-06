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
    "gamma" : 0.99,
    "lr": 1e-4,
    "replay_capacity": 150_000,
    "min_replay_size": 1_000,
    "target_update_freq": None
}

CURVE_TRAIN_CONFIG = {
    "batch_size": 32,
    "gamma": 0.97,
    "lr": 5e-5,
    "replay_capacity": 150_000,
    "min_replay_size": 10_000,
    "target_update_freq": None
}

EXPERT_RATIO = 0.3
EXPERT_RATIO_CURVE = 0.35
PRETRAIN_DQN_STEPS = 0
EXPERT_DATASET = "dataset/expert_dataset.json"

EVAL_EPISODES_PER_MAP = 10

CURVE_EPSILON_CONFIG = {
    "start": 0.80,
    "end": 0.03,
    "decay": 800_000,
    "warmup_steps": 1_000
}

EPSILON_CONFIG = {
    "start": 0.50,
    "end": 0.05,
    "decay": 200_000,
    "warmup_steps": 5_000
}

CHECKPOINT_FREQ = 100
EVAL_FREQ = 200
BC_CHECKPOINT_STRAIGHT = "checkpoints/bc_pretrain_straight.pt"
BC_CHECKPOINT_CURVE = "checkpoints/bc_pretrain_curve.pt"

CURVE_BC_CONFIG = {
    "epochs": 80,
    "batch_size": 256,
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