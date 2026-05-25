ENV_CONFIG = {
        "use_render": False,
        "manual_control": False,
        "traffic_density": 0.1,
        "num_scenarios": 100,
        "start_seed": 0,
        "map": "SSSS",
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

FRAME_STACK = 4

TRAIN_CONFIG = {
    "num_episodes": 2500,
    "batch_size": 64,
    "gamma" : 0.99,
    "lr": 1e-5,
    "replay_capacity": 150_000,
    "min_replay_size": 2_000,
    "target_update_freq": 300
}

EXPERT_RATIO = 0.3
EXPERT_DATASET = "dataset/expert_dataset.json"

EPSILON_CONFIG = {
    "start": 1.0,
    "end": 0.05,
    "decay": 400_000,
    "warmup_steps": 500
}

CHECKPOINT_FREQ = 100
EVAL_FREQ = 100
RESUME_PATH = None
BC_CHECKPOINT = "checkpoints/bc_pretrain.pt"

BC_CONFIG = {
    "epochs": 50,
    "batch_size": 128,
    "lr": 5e-4,
    "val_split": 0.15,
    "patience": 8,
    "clip_grad": 1.0
}