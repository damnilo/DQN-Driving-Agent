import torch
from enviornments.metadrive_env import MetaDriveEnvWrapper
from agents.dqn_agent import DQNAgent
from agents.epsilon_scheduler import EpsilonScheduler

ENV_CONFIG = {
        "use_render": True,
        "manual_control": False,
        "traffic_density": 0.1,
        "num_scenarios": 500,
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

def main():

    config = ENV_CONFIG
    env = MetaDriveEnvWrapper(config)

    obs, info = env.reset()
    state_dim = env.obs_size if env.obs_size is not None else len(obs)
    action_dim = env.num_actions()
    epsilon_scheduler = EpsilonScheduler(start=0.0, end=0.0, decay=1, warmup_steps=0)
    agent = DQNAgent(state_dim, action_dim, epsilon_scheduler)

    checkpoint_path = "checkpoints/final.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    agent.online_net.load_state_dict(checkpoint["online_net"])
    agent.online_net.eval()
    done = False
    total_reward = 0
    global_step = 0

    while not done:

        with torch.no_grad():

            action = agent.select_action(
                obs, global_step, training=False
            )

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if done:
            print(f"Voznja zavrsena! Uspesnost: {info.get('arrive_dest', False)}")
            print(f"Ukupna nagrada: {total_reward}")
            break

    env.close()

if __name__ == "__main__":
    main()