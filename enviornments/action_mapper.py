import numpy as np

class ActionMapper:
    def __init__(self):
        # Use ordered sequences (lists) so indexing is stable and deterministic
        self.steering_actions = [
            -0.50, -0.36, -0.26, -0.18, -0.12, -0.06, 0.0,
            0.06, 0.12, 0.18, 0.26, 0.36, 0.50
        ]

        self.throttle_actions = [
            -0.30, -0.15, 0.10, 0.35, 0.50, 0.65
        ]

        self.action_space = [
            (steer_idx, throttle_idx)
            for steer_idx in range(len(self.steering_actions))
            for throttle_idx in range(len(self.throttle_actions))
        ]

    def map(self, action):
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