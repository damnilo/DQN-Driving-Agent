import numpy as np

class ObservationBuilder:

    EGO_SLICE = slice(0, 9)
    NAV_SLICE = slice(9, 19)
    LIDAR_SLICE = slice(19, 259)

    def __init__(self):
        self._action_map = None

    def reset(self):
        self.prev_action_idx = 0

    def build(self, env, raw_obs, info, prev_action_idx=0):
        vec = self._to_vector(raw_obs)

        if len(vec) >= 259:
            ego = np.clip(vec[self.EGO_SLICE], -1.0, 1.0)
            nav = np.clip(vec[self.NAV_SLICE], -1.0, 1.0)
            lidar = np.clip(vec[self.LIDAR_SLICE], 0.0, 1.0)
        else:
            ego, nav, lidar = self._fallback_from_agent(env, info)

        sensor_extras = self._sensor_extras(env)
        nav_extras = self._nav_extras(info)
        angles_extras = self._future_waypoints(env)

        # Prev action may be a scalar index or a pair [steering_idx, throttle_idx].
        if isinstance(prev_action_idx, (list, tuple, np.ndarray)):
            arr = np.asarray(prev_action_idx, dtype=np.float32).flatten()
            if arr.size == 1:
                prev_action_feat = np.asarray([float(arr[0]), 0.0], dtype=np.float32)
            else:
                # ensure exactly two elements
                prev_action_feat = np.asarray([float(arr[0]), float(arr[1])], dtype=np.float32)
        else:
            # keep two-element feature for compatibility
            prev_action_feat = np.asarray([float(prev_action_idx), 0.0], dtype=np.float32)

        # Ensure all pieces are 1-D arrays before concatenation
        ego = np.atleast_1d(ego)
        nav = np.atleast_1d(nav)
        lidar = np.atleast_1d(lidar)
        sensor_extras = np.atleast_1d(sensor_extras)
        nav_extras = np.atleast_1d(nav_extras)
        prev_action_feat = np.atleast_1d(prev_action_feat)
        angles_extras = np.atleast_1d(angles_extras)

        return np.concatenate([ego, nav, lidar, sensor_extras, nav_extras, prev_action_feat, angles_extras]).astype(np.float32)
    
    def _nav_extras(self, info):
        nav_cmd = float(info.get("navigation_command_float", 0.0))
        dist_left = np.clip(float(info.get("dist_left", 0.0)) / 10.0, 0.0, 1.0)
        dist_right = np.clip(float(info.get("dist_right", 0.0)) / 10.0, 0.0, 1.0)
        lane_center = np.clip(float(info.get("lane_center_ratio", 0.0)), -1.0, 1.0)

        return np.array([nav_cmd, dist_left, dist_right, lane_center], dtype=np.float32)

    def _to_vector(self, raw_obs):
        if isinstance(raw_obs, dict):
            parts = []
            for key in ("ego_state", "navigation", "lidar"):
                if key in raw_obs:
                    parts.append(np.asarray(raw_obs[key], dtype=np.float32).flatten())
            
            if parts:
                return np.concatenate(parts)

            return np.asarray(raw_obs.get("lidar", []), dtype=np.float32).flatten()
        
        return np.asarray(raw_obs, dtype=np.float32).flatten()

    def _fallback_from_agent(self, env, info):
        agent = env.agent
        speed = float(info.get("velocity", getattr(agent, "speed", 0.0))) / 120.0
        heading_err = float(info.get("heading_error", 0.0)) / np.pi
        lat = float(info.get("lateral_offset", 0.0)) / 5.0
        ego = np.array([
            heading_err,
            float(getattr(agent, "steering", 0.0)),
            speed,
            float(info.get("dist_left", 0.0)) / 5.0,
            float(info.get("dist_right", 0.0)) / 5.0,
            0, 0, 0, 0
        ], dtype=np.float32)[:9]
        nav = np.zeros(10, dtype=np.float32)
        try:
            ni = np.asarray(env.agent.navigation.get_navi_info(), dtype=np.float32)
            nav[: min(10, len(ni))] = ni[:10]
        except Exception:
            pass

        lidar = self._read_lidar_sensor(env)
        return ego, nav, lidar

    def _read_lidar_sensor(self, env):
        try:
            lidar = env.engine.get_sensor("lidar")
            data = np.asarray(lidar.perceive(env.agent, num_lasers=240), dtype=np.float32).flatten()
            return np.clip(data[:240], 0.0, 1.0)
        except Exception:
            return np.zeros(240, dtype=np.float32)

    def _sensor_extras(self, env):
        parts = []
        for name in ("side_detector", "lane_line_detector"):
            try:
                sensor = env.engine.get_sensor(name)
                data = np.asarray(sensor.perceive(env.agent), dtype=np.float32)
                parts.append(np.clip(data, 0.0, 1.0))
            except Exception:
                pass

        if not parts:
            return np.array([], dtype=np.float32)
        
        return np.concatenate(parts)
    
    def _future_waypoints(self, env):

        future_angles = []

        try:
            checkpoints = env.agent.navigation.checkpoints

            ego_pos = env.agent.position
            ego_heading = env.agent.heading_theta

            for checkpoint in checkpoints[:5]:
                dx = checkpoint[0] - ego_pos[0]
                dy = checkpoint[1] - ego_pos[1]

                waypoint_heading = np.arctan2(dy, dx)

                angle = waypoint_heading - ego_heading
                angle = np.arctan2(np.sin(angle), np.cos(angle))

                angle = np.clip(angle / np.pi, -1.0, 1.0)

                future_angles.append(angle)
        except Exception:
            future_angles = []

        while len(future_angles) < 5:
            future_angles.append(0.0)

        curvature = abs(future_angles[-1] - future_angles[0])

        future_angles.append(curvature)

        return np.asarray(future_angles, dtype=np.float32)