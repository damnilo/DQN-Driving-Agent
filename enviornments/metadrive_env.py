import numpy as np
from metadrive import MetaDriveEnv
from enviornments.action_mapper import ActionMapper
from enviornments.observation_builder import ObservationBuilder
from enviornments.reward_function import RewardFunction
from enviornments.info_builder import InfoBuilder

class MetaDriveEnvWrapper:

    def __init__(self, env_config):

        self.env = MetaDriveEnv(env_config)

        self.action_mapper = ActionMapper()

        self.observation_builder = ObservationBuilder()

        self.reward_function = RewardFunction()

        self.info_builder = InfoBuilder()
        self.prev_longitudinal = None
        self.stuck_step = 0
        
        self.obs_size: int = None

        self._last_discrete_action = 0

    def reset(self):
        self._last_discrete_action = 0
        self.prev_longitudinal = None
        self.stuck_step = 0
        self.reward_function.reset()
        self.observation_builder.reset()
        raw_obs, info = self.env.reset()
        info = self._enrich_info(info)
        processed_obs = self.observation_builder.build(
            self.env, raw_obs, info, prev_action_idx = 0
        )
        future_features = self.get_future_waypoint_features()

        processed_obs = np.concatenate([processed_obs, future_features]).astype(np.float32)
        if self.obs_size is None:
            self.obs_size = len(processed_obs)

        return processed_obs, info

    def step(self, discrete_action):
        self._last_discrete_action = discrete_action

        continuous_action = self.action_mapper.map(
            discrete_action
        )

        return self._step_inner(continuous_action)
    
    def close(self):

        self.env.close()

    def num_actions(self):
        return self.action_mapper.num_actions()

    @property
    def agent(self):
        return self.env.agent

    @property
    def engine(self):
        return self.env.engine

    def step_continuous(self, continuous_action):
        return self._step_inner(continuous_action)

    def _step_inner(self, continuous_action):
        raw_obs, env_reward, terminated, truncated, info = self.env.step(continuous_action)
        info = self._enrich_info(info)
        long = info.get("longitudinal", 0.0)
        if self.prev_longitudinal is not None:
            progress_delta = long - self.prev_longitudinal

            if progress_delta < 0.05:
                self.stuck_step += 1
            else:
                self.stuck_step = 0
        
        if self.stuck_step > 120:
            terminated = True
            info["stuck"] = True
        
        self.prev_longitudinal = long
        reward = self.reward_function.compute(info)
        processed_obs = self.observation_builder.build(self.env, raw_obs, info, prev_action_idx=self._last_discrete_action)
        future_features = self.get_future_waypoint_features()
        processed_obs = np.concatenate([processed_obs, future_features]).astype(np.float32)
        return processed_obs, reward, terminated, truncated, info

    def _enrich_info(self, info):
        return self.info_builder.build(self.env, info)

    def get_map(self):
        return self.env.config["map"]
    
    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_future_waypoint_features(self):
        vehicle = self.env.agent
        lane = vehicle.lane

        long, _ = lane.local_coordinates(vehicle.position)

        future_distances = [5, 10, 20, 35]
        features = []

        vehicle_heading = vehicle.heading_theta
        vehicle_pos = np.array(vehicle.position)

        cos_h = np.cos(-vehicle_heading)
        sin_h = np.sin(-vehicle_heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        for d in future_distances:
            future_long = long + d

            try:
                future_world_pos = np.array(lane.position(future_long, 0))
                future_heading = lane.heading_theta_at(future_long)

                heading_diff = self.normalize_angle(future_heading - vehicle_heading) / np.pi

                curvature = np.clip((heading_diff / max(d, 1.0)) * 10, -1, 1)

                relative_world = (future_world_pos - vehicle_pos)
                relative_local = rotation @ relative_world

                local_x = np.clip(relative_local[0] / 40.0, -1.0, 1.0)
                local_y = np.clip(relative_local[1] / 5.0, -1.0, 1.0)

                features.extend([heading_diff, curvature, local_x, local_y])
            except Exception:
                features.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)