import numpy as np

class RewardFunction:

    def __init__(self):
        self.prev_steering = 0.0
        self.prev_longitudinal = None

    def compute(self, info):
        if info.get("crash", False):
            return -100.0

        if info.get("out_of_road", False):
            return -100.0
        
        if info.get("arrive_dest", False):
            return 1000.0
        
        if info.get("max_step", False):
            return -50.0
        
        if info.get("stuck", False):
            return -40.0
        
        steering = float(info.get("steering", 0.0))
        long = float(info.get("longitudinal", 0.0))
        speed_val = float(info.get("velocity", 0.0))
        heading_err = abs(float(info.get("heading_error", 0.0)))
        lateral = abs(float(info.get("lateral_offset", 0.0)))
        nav_cmd = info.get("navigation_command", "STRAIGHT")

        reward = 0.1

        if self.prev_longitudinal is None:
            progress_delta = 0.0
        else:
            progress_delta = long - self.prev_longitudinal

        reward += progress_delta * 2.5

        reward += np.cos(heading_err) * 0.6
        reward -= abs(lateral) * 0.5
        reward -= 0.25 * abs(steering - self.prev_steering)
        reward -= 0.03 * abs(steering)
        reward -= abs(heading_err) * speed_val * 0.04

        if nav_cmd in ("LEFT", "RIGHT", "left", "right") and speed_val < 25:
            reward += 0.3

        if abs(heading_err) > 0.5:
            reward -= 2.0

        self.prev_steering = steering
        self.prev_longitudinal = long

        return reward

        
    def reset(self):
        self.prev_steering = 0.0
        self.prev_longitudinal = None