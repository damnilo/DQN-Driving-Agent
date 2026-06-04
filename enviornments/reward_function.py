import numpy as np

class RewardFunction:
    LANE_WIDTH = 2.0
    MAX_HEADING_ERR = np.pi

    def __init__(self):
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.low_speed_steps = 0
        self.episode_steps = 0
        self.use_soft_out_of_road = False

    def compute(self, info, env_reward=0.0):
        self.episode_steps += 1
        if info.get("crash", False):
            return -100.0 if self.use_soft_out_of_road else -300.0

        if info.get("out_of_road", False):
            return -80.0 if self.use_soft_out_of_road else -300.0
        
        if info.get("arrive_dest", False):
            return 100.0
        
        steering = float(info.get("steering", 0.0))

        speed_val = float(info.get("velocity", 0.0))
        heading_err = abs(float(info.get("heading_error", 0.0)))
        lateral = abs(float(info.get("lateral_offset", 0.0)))

        reward = 0.0

        reward += (speed_val * np.cos(heading_err)) / 40.0
        reward += (1.0 - lateral) * 0.2
        reward -= 0.02 * abs(steering - self.prev_steering)

        self.prev_steering = steering

        return reward

        
    def reset(self):
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.low_speed_steps = 0
        self.episode_steps = 0