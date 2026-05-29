import numpy as np

STEERING_WEIGHT = 3.0

def continuous_to_discrete(steering, throttle, action_map):

    best_action, best_distance = 0, float("inf")

    for idx, (s, t) in action_map.items():
        ds, dt = steering - s, throttle - t
        distance = (ds * STEERING_WEIGHT) ** 2 + dt ** 2

        if distance < best_distance:
            best_distance, best_action = distance, idx

    return best_action