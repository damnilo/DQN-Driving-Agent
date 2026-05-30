class RewardFunction:
    LANE_WIDTH = 2.0
    MAX_HEADING_ERR = 1.0

    def __init__(self):
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.low_speed_steps = 0
        self.use_soft_out_of_road = False

    def compute(self, info, env_reward=0.0, action=None):
        if info.get("crash", False):
            return -150.0 if self.use_soft_out_of_road else -300.0

        if info.get("out_of_road", False):
            return -60.0 if self.use_soft_out_of_road else -300.0
        
        if info.get("arrive_dest", False):
            return 200.0

        step_reward = float(info.get("step_reward", env_reward))  # MetaDrive ugrađeni reward
        
        steering = float(info.get("steering", 0.0))
        throttle = float(info.get("acceleration", 0.0))
        
        if action is not None:
            steering = float(action[0])
            throttle = float(action[1])

        speed_val = float(info.get("velocity", 0.0))
        heading_err = abs(float(info.get("heading_error", 0.0)))
        lateral = abs(float(info.get("lateral_offset", 0.0)))
        nav_cmd = float(info.get("navigation_command_float", 0.0))

        penalty = 0.0

        jerk_scale = 0.01 if self.use_soft_out_of_road else 0.02
        penalty += abs(throttle - self.prev_throttle) * 0.02

        base_steer_jerk = abs(steering - self.prev_steering)
        if heading_err > 0.25:
            steer_w = 0.03 if base_steer_jerk > 10 else 0.005
        else:
            steer_w = 0.04

        penalty += base_steer_jerk * steer_w

        step_reward += 0.4 * max(0.0, 1.0 - heading_err / self.MAX_HEADING_ERR)
        step_reward += 0.3 * max(0.0, 1.0 - lateral / (self.LANE_WIDTH / 2))

        if nav_cmd != 0.0 and steering * nav_cmd > 0:
            turn_bonus = 0.3 * abs(steering) * abs(nav_cmd)
            step_reward += turn_bonus

        target_speed = max(6.0, 40.0 * (1.0 - heading_err))
        moving_bonus = min(speed_val, target_speed) * 0.01

        if speed_val < 1.0 and heading_err < 0.2:
            self.low_speed_steps += 1
            penalty += 1.0 + min(self.low_speed_steps, 50) * 0.03
        else:
            self.low_speed_steps = 0

        self.prev_steering = steering
        self.prev_throttle = throttle 

        return float(step_reward + moving_bonus - penalty)
    
    def reset(self):
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.low_speed_steps = 0