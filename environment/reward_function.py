import numpy as np

class RewardFunction:

    def __init__(self):
        self.prev_steering = 0.0
        self._prev_route_travelled = None

    def compute(self, info):
        """Returns the shaped step reward. Terminal outcomes are handled first.
        Live reward follows metres travelled along the navigation route, so a lane
        change does not cancel earlier progress."""

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
        heading_err = abs(float(info.get("heading_error", 0.0)))
        lateral = abs(float(info.get("lateral_offset", 0.0)))
        travelled = float(info.get("route_travelled", 0.0))

        if self._prev_route_travelled is None:
            route_delta = 0.0
        else:
            route_delta = travelled - self._prev_route_travelled
        route_delta = float(np.clip(route_delta, -1.0, 20.0))

        reward = 0.05
        reward += route_delta * 2.0
        reward += np.cos(heading_err) * 0.35
        reward -= min(lateral, 3.0) * 0.25
        reward -= 0.15 * abs(steering - self.prev_steering)
        reward -= 0.02 * abs(steering)

        self.prev_steering = steering
        self._prev_route_travelled = travelled

        return reward

    def reset(self):
        self.prev_steering = 0.0
        self._prev_route_travelled = None