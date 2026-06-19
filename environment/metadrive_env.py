import numpy as np
from metadrive import MetaDriveEnv
from environment.action_mapper import ActionMapper
from environment.observation_builder import ObservationBuilder
from environment.reward_function import RewardFunction
from environment.info_builder import InfoBuilder

class MetaDriveEnvWrapper:

    def __init__(self, env_config):
        """Instantiates MetaDriveEnv together with all helper components (action mapper,
        observation builder, reward function, info builder) and initialises per-episode
        tracking state."""

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
        """Resets the environment and all stateful helpers, constructs the first
        processed observation including future waypoint features, and caches the
        observation size on the first call."""

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

        processed_obs = np.concatenate([future_features, processed_obs]).astype(np.float32)
        if self.obs_size is None:
            self.obs_size = len(processed_obs)

        return processed_obs, info

    def step(self, discrete_action):
        """Records the action index, maps it to a continuous pair, and delegates
        execution to _step_inner."""

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
        """Executes a raw env step, detects stuck episodes through longitudinal-progress
        tracking, applies the custom reward function, and builds the processed
        observation with waypoint features appended."""
        
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
        processed_obs = np.concatenate([future_features, processed_obs]).astype(np.float32)
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
        """Walks the lane chain ahead at [5, 10, 20, 35] m and returns 16 features:
        per-waypoint heading difference, curvature, and vehicle-relative (x, y)
        position."""
         
        vehicle = self.env.agent
        lane = vehicle.lane

        try:
            long, _ = lane.local_coordinates(vehicle.position)
        except Exception:
            long = 0.0

        future_distances = [5, 10, 20, 35]
        features = []

        vehicle_heading = vehicle.heading_theta
        vehicle_pos = np.array(vehicle.position)

        cos_h = np.cos(-vehicle_heading)
        sin_h = np.sin(-vehicle_heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        for d in future_distances:
            remaining = d
            current_lane = lane
            cur_long = long
            found = False

            try:
                while current_lane is not None and remaining >= 0:
                    lane_length = float(getattr(current_lane, "length", 0.0) or 0.0)

                    # distance available on this lane from cur_long to end
                    avail = max(0.0, lane_length - cur_long)

                    if remaining <= avail:
                        # point lies on this lane
                        future_world_pos = np.array(current_lane.position(cur_long + remaining, 0))
                        future_heading = current_lane.heading_theta_at(cur_long + remaining)
                        found = True
                        break
                    else:
                        # advance to next lane
                        remaining -= avail
                        cur_long = 0.0
                        # choose the next lane that best matches current heading
                        next_lanes = getattr(current_lane, "next_lanes", None) or []
                        if not next_lanes:
                            current_lane = None
                            break
                        current_lane = self._select_best_lane(next_lanes, vehicle.heading_theta)

                if not found:
                    raise RuntimeError("Could not sample future point on lane chain")

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
    
    def _select_best_lane(self, lanes, heading):
        """Picks the next-lane candidate whose initial heading most closely matches
        the given heading, used to trace the most plausible route through junctions."""

        best, best_diff = lanes[0], float("inf")
        
        for lane in lanes:
            try:
                diff = abs(self.normalize_angle(lane.heading_theta_at(0) - heading))
                if diff < best_diff:
                    best, best_diff = lane, diff
            except Exception:
                pass

        return best