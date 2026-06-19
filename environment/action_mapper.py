import numpy as np

class ActionMapper:
    def __init__(self):
        """Builds the discrete action space as the Cartesian product of 7 steering
        values × 4 throttle values, yielding 28 actions total."""

        self.steering_actions = [-0.30, -0.18, -0.09, 0.0, 0.09, 0.18, 0.30]
        self.throttle_actions = [-0.30, -0.05, 0.25, 0.60]

        self.action_space = [
            (self.steering_actions[steer_idx], self.throttle_actions[throttle_idx])
            for steer_idx in range(len(self.steering_actions))
            for throttle_idx in range(len(self.throttle_actions))
        ]

    def map(self, action):
        """Converts either a flat integer index or a (steer_idx, throttle_idx) pair into
        the corresponding continuous (steering, throttle) tuple."""
        
        if isinstance(action, (int, np.integer)):
            return self.action_space[int(action)]

        if isinstance(action, (list, tuple, np.ndarray)):
            if len(action) != 2:
                raise ValueError("Expected discrete action pair [steer_idx, throttle_idx]")
            action_space_idx = int(action[0]) * len(self.throttle_actions) + int(action[1])
            return self.action_space[action_space_idx]

        raise TypeError("Action must be an int index or a 2-element discrete action pair")
    
    def num_actions(self):
        return len(self.action_space)
    
    def get_steering_actions(self):
        return np.array(self.steering_actions, dtype=np.float32)
    
    def get_throttle_actions(self):
        return np.array(self.throttle_actions, dtype=np.float32)