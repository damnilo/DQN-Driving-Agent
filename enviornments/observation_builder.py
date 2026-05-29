import numpy as np

class ObservationBuilder:

    EGO_SLICE = slice(0, 9)
    NAV_SLICE = slice(9, 19)
    LIDAR_SLICE = slice(19, 259)
    OBS_DIM = 259

    def build(self, env, raw_obs, info):
        vec = self._to_vector(raw_obs)

        if len(vec) >= 259:
            ego = np.clip(vec[self.EGO_SLICE], -1.0, 1.0)
            nav = np.clip(vec[self.NAV_SLICE], -1.0, 1.0)
            lidar = np.clip(vec[self.LIDAR_SLICE], 0.0, 1.0)
        else:
            ego, nav, lidar = self._fallback_from_agent(env, info)

        extras = self._sensor_extras(env)

        return np.concatenate([ego, nav, lidar, extras]).astype(np.float32)
    
    def _to_vector(self, raw_obs):
        if isinstance(raw_obs, dict):
            parts = []
            for key in ("lidar", "ego_state", "navigation"):
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
            data = np.asarray(lidar.perceive(env.agent, num_lasers=240), dtype=np.float32)
            return np.clip(data.flatten()[:240], 0.0, 1.0)
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