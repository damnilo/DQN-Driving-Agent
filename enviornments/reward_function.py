class RewardFunction:
    LANE_WIDTH = 2.0
    MAX_HEADING_ERR = 1.0

    def __init__(self):
        self.prev_steering = 0.0

    def compute(self, info):
        reward = 0
        
        if info.get("out_of_road", False):
            return -300.0
        
        if info.get("crash", False):
            return -300.0
        # 1. Add a small 'Existence' bonus. 
        # This makes every step alive worth something.
        reward += 0.1

        reward += self.forward_reward(info)
        reward += self.idle_penalty(info) # Renamed for clarity
        reward += self.lane_reward(info)
        reward += self.goal_reward(info)
        reward += self.action_smoothing_penalty(info)
        reward += self.heading_penalty(info)
        reward += self.lateral_penalty(info)

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
        return speed * haeding_factor * 0.03
    
    def lane_reward(self, info):
        lane_offset = abs(info.get("lateral", 0.0))
        normalised = min(lane_offset / self.LANE_WIDTH, 1.0)
        # We give up to +0.8 here
        return (1.0 - (normalised)) * 1.5
    
    def action_smoothing_penalty(self, info):
        return float(-0.3 * abs(info.get("steering", 0.0)))
    
    def heading_penalty(self, info):
        heading_err = abs(info.get("heading_diff", 0.0))
        normilised = min(heading_err / self.MAX_HEADING_ERR, 1.0)

        return -2.0 * normilised
    
    def lateral_penalty(self, info):
        lateral = info.get("lateral", 0.0)
        lateral_velocity = info.get("lateral_velocity", 0.0)

        if (lateral * lateral_velocity) > 0:
            return -1.0 * abs(lateral_velocity)
        
        return 0.0

    def goal_reward(self, info):

        if info.get("arrive_dest", False):
            return 150.0
        
        return 0.0
