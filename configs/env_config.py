ENV_CONFIG = {
        "use_render": False,
        "manual_control": False,
        "traffic_density": 0.0,
        "num_scenarios": 50,
        "start_seed": 0,
        "map": "SSSS",
        "image_observation": False,
        # "daytime": random.choice(["08:00", "12:00", "17:30", "20:00"]),
        "accident_prob": 0.0,
        # ISPRAVLJENI KLJUČEVI:
        "vehicle_config": {
            "show_lidar": False,
            "show_side_detector": False,
            "show_lane_line_detector": False,

            "lidar": dict(
                num_lasers=240,
                distance=50.0,
                num_others=0,
                gaussian_noise=0.0,
                dropout_prob=0.0,
            ),

            "side_detector": dict(
                num_lasers = 0,
                distance = 50.0,
            ),

            "lane_line_detector": dict(
                num_lasers = 0,
                distance = 50.0,
            ),

        },
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
    "lr": 5e-6,
    "replay_capacity": 150_000,
    "min_replay_size": 1_000,
    "target_update_freq": None
}

EXPERT_RATIO = 0.3
EXPERT_RATIO_CURVE = 0.35
PRETRAIN_DQN_STEPS = 0
EXPERT_DATASET = "dataset/expert_dataset.json"

EVAL_EPISODES_PER_MAP = 10

CURVE_EPSILON_CONFIG = {
    "start": 0.60,
    "end": 0.05,
    "decay": 400_000,
    "warmup_steps": 1_000
}

EPSILON_CONFIG = {
    "start": 0.50,
    "end": 0.02,
    "decay": 400_000,
    "warmup_steps": 5_000
}

CHECKPOINT_FREQ = 100
EVAL_FREQ = 200
BC_CHECKPOINT = "checkpoints/bc_pretrain.pt"

BC_CONFIG = {
    "epochs": 120,
    "batch_size": 256,
    "lr": 2e-4,
    "val_split": 0.15,
    "patience": 20,
    "clip_grad": 0.5
}