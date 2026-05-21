class RewardFunction:
    LANE_WIDTH = 2.0
    MAX_HEADING_ERR = 1.0

    def __init__(self):
        self.prev_steering = 0.0

    def compute(self, info):
        reward = 0
        
        if info.get("out_of_road", False):
            return -50.0
        
        if info.get("crash", False):
            return -50.0
        
        speed_penalty = 0.0
        speed = info.get("speed", 0.0)
        if speed < 5.0:
            speed_penalty -= 0.5
        
        # 1. Add a small 'Existence' bonus. 
        # This makes every step alive worth something.
        reward += 0.1

        reward += self.forward_reward(info) * 2.0
        reward += self.lane_reward(info) * 0.5
        reward += self.goal_reward(info)
        reward += self.heading_penalty(info) * 0.3
        reward += self.action_smoothing_penalty(info) * 0.2
        reward += self.lateral_penalty(info) * 0.2
        reward += speed_penalty
        reward += self.idle_penalty(info) * 0.3

        return reward
    
    def idle_penalty(self, info):
        # Reduced this so it doesn't completely cancel out the lane reward
        if info.get("speed", 0.00) < 0.5:
            return -0.5 
        return 0.0
    
    def forward_reward(self, info):
        speed = info.get("speed", 0.0)
        heading_err = abs(info.get("heading_diff", 0.0))
        haeding_factor = max(0.0, 1.0 - heading_err)
        # Boosted slightly to reward progress more than just "sitting centered"
        return speed * haeding_factor * 0.015
    
    def lane_reward(self, info):
        lane_offset = abs(info.get("lateral", 0.0))
        normalised = min(lane_offset / self.LANE_WIDTH, 1.0)
        # We give up to +0.8 here
        return (1.0 - (normalised)) * 0.7
    
    def action_smoothing_penalty(self, info):
        steering = info.get("steering", 0.0)
        heading_err = abs(info.get("heading_diff", 0.0))
        
        scale = max(0.05, 0.35 * (1.0 - heading_err * 1.5))
        
        penalty = abs(steering - self.prev_steering)
        self.prev_steering = steering
        return float(-scale * penalty)
    
    def heading_penalty(self, info):
        heading_err = abs(info.get("heading_diff", 0.0))
        normilised = min(heading_err / self.MAX_HEADING_ERR, 1.0)

        return -0.5 * normilised
    
    def lateral_penalty(self, info):
        lateral = info.get("lateral", 0.0)
        lateral_velocity = info.get("lateral_velocity", 0.0)

        if (lateral * lateral_velocity) > 0:
            return -0.2 * abs(lateral_velocity)
        
        return 0.0

    def goal_reward(self, info):

        if info.get("arrive_dest", False):
            return 150.0
        
        return 0.0
