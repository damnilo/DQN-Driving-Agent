import numpy as np

class ObservationBuilder:

    def build(self, env, raw_obs, info):

        lidar = self.extract_lidar(raw_obs) / 50.0

        speed = np.clip(np.array([info.get("speed", 0.0)]) / 120.0, 0.0, 1.0)

        heading = np.clip(np.array([self.compute_heading_error(info)]) / np.pi, -1.0, 1.0)

        lane_offset = np.clip(np.array([self.compute_lane_offset(info)]) / 5.0, -1.0, -1.0)

        waypoints = self.extract_waypoints(env)  # NOVO

        return np.concatenate([
            lidar, speed, heading, lane_offset, waypoints  # NOVO
        ]).astype(np.float32)
    
    def extract_lidar(self, raw_obs):

        if isinstance(raw_obs, dict):
            lidar = raw_obs.get("lidar", raw_obs.get("cloud_points", np.array([])))
            return np.array(lidar, dtype=np.float32).flatten()

        return np.array(raw_obs, dtype=np.float32).flatten()
    
    def compute_heading_error(self, info):
        return info.get("heading_diff", 0.0)
    
    def compute_lane_offset(self, info):
        return info.get("lateral", 0.0)

    def extract_waypoints(self, env):
        try:
            navi = env.agent.navigation
            checkpoints = navi.checkpoints
            ego_pos = env.agent.position
            ego_heading = env.agent.heading_theta

            waypoints = []
            for i in range(min(6, len(checkpoints))):
                wp = checkpoints[i]
                dx = wp[0] - ego_pos[0]
                dy = wp[1] - ego_pos[1]

                # Pretvori u lokalni koordinatni sistem vozila
                cos_h = np.cos(-ego_heading)
                sin_h = np.sin(-ego_heading)
                local_x = dx * cos_h - dy * sin_h
                local_y = dx * sin_h + dy * cos_h

                # Normalizuj
                waypoints.extend([local_x / 50.0, local_y / 50.0])

            # Dopuni nulama ako ima manje od 3 checkpointa
            while len(waypoints) < 12:
                waypoints.append(0.0)

            return np.array(waypoints, dtype=np.float32)

        except Exception:
            return np.zeros(12, dtype=np.float32)

    def obs_size(self, env, raw_obs: np.ndarray, info: dict) -> int:
        return len(self.build(env, raw_obs, info))