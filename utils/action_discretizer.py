import numpy as np
from environment.action_mapper import ActionMapper

action_mapper = ActionMapper()

def _discretize_steering(steering):
    return int(np.argmin(np.abs(action_mapper.get_steering_actions() - steering)))

def _discretize_throttle(throttle):
    return int(np.argmin(np.abs(action_mapper.get_throttle_actions() - throttle)))

def discretize_action(steering, throttle):
    """Snaps continuous (steering, throttle) values to their nearest bin indices and
    encodes them as a single flat index (steer_idx × num_throttle + throttle_idx)."""

    discrete_steering = _discretize_steering(steering)
    discrete_throttle = _discretize_throttle(throttle)
    return discrete_steering * len(action_mapper.throttle_actions) + discrete_throttle