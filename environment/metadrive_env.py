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
        self._prev_position = None
        self.stuck_step = 0
        
        self.obs_size: int = None

        self._last_discrete_action = 0

    def reset(self):
        """Resets the environment and all stateful helpers, constructs the first
        processed observation including future waypoint features, and caches the
        observation size on the first call."""

        self._last_discrete_action = 0
        self._prev_position = None
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
        position = np.asarray(self.env.agent.position, dtype=np.float64)[:2]
        if self._prev_position is not None:
            moved = float(np.linalg.norm(position - self._prev_position))
            if moved < 0.2:
                self.stuck_step += 1
            else:
                self.stuck_step = 0

        if self.stuck_step > 120:
            terminated = True
            info["stuck"] = True

        self._prev_position = position
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
        """Samples the navigation route at [5, 10, 20, 35] m and returns 16 features:
        per-waypoint heading difference, curvature, and vehicle-relative (x, y)."""

        vehicle = self.env.agent
        future_distances = [5, 10, 20, 35]
        features = []

        vehicle_heading = vehicle.heading_theta
        vehicle_pos = np.array(vehicle.position)[:2]

        cos_h = np.cos(-vehicle_heading)
        sin_h = np.sin(-vehicle_heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        route = self._route_lane_sequence(vehicle)

        for d in future_distances:
            try:
                future_world_pos, future_heading = self._sample_along_lanes(route, vehicle, d)
                heading_diff = self.normalize_angle(future_heading - vehicle_heading) / np.pi
                curvature = np.clip((heading_diff / max(d, 1.0)) * 10, -1, 1)

                relative_world = future_world_pos - vehicle_pos
                relative_local = rotation @ relative_world

                local_x = np.clip(relative_local[0] / 40.0, -1.0, 1.0)
                local_y = np.clip(relative_local[1] / 5.0, -1.0, 1.0)

                features.extend([heading_diff, curvature, local_x, local_y])
            except Exception:
                features.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)

    def _route_lane_sequence(self, vehicle):
        """Ordered lanes along the navigation checkpoints, starting from the road
        the vehicle is on. Falls back to the vehicle lane when navigation is missing."""

        nav = getattr(vehicle, "navigation", None)
        checkpoints = getattr(nav, "checkpoints", None) if nav is not None else None
        if not checkpoints or len(checkpoints) < 2:
            return [vehicle.lane]

        graph = nav.map.road_network.graph
        start = int(nav._target_checkpoints_index[0])
        sequence = []
        previous = None

        for i in range(start, len(checkpoints) - 1):
            road_lanes = list(graph[checkpoints[i]][checkpoints[i + 1]])
            if not road_lanes:
                break
            if previous is None:
                chosen = self._closest_lane(vehicle, road_lanes)
            else:
                chosen = self._connected_lane(previous, road_lanes) or road_lanes[0]
            sequence.append(chosen)
            previous = chosen

        return sequence or [vehicle.lane]

    def _sample_along_lanes(self, lanes, vehicle, distance):
        """Walks `distance` metres forward along `lanes` and returns world position
        and lane heading at that point."""

        current = lanes[0]
        try:
            long, _ = current.local_coordinates(vehicle.position)
        except Exception:
            long = 0.0
        long = float(np.clip(long, 0.0, float(getattr(current, "length", 0.0) or 0.0)))

        remaining = distance
        index = 0
        hops = 0
        while current is not None and remaining >= 0 and hops < 16:
            hops += 1
            lane_length = float(getattr(current, "length", 0.0) or 0.0)
            available = max(0.0, lane_length - long)
            if remaining <= available:
                point = np.asarray(current.position(long + remaining, 0), dtype=np.float64)[:2]
                heading = float(current.heading_theta_at(long + remaining))
                return point, heading

            remaining -= available
            long = 0.0
            index += 1
            if index < len(lanes):
                current = lanes[index]
                continue

            nxt = list(getattr(current, "next_lanes", None) or [])
            current = self._connected_lane(current, nxt) if nxt else None
            if current is None and nxt:
                current = nxt[0]

        raise RuntimeError("Could not sample future point on the route")

    def _closest_lane(self, vehicle, lanes):
        best, best_lat = lanes[0], float("inf")
        for lane in lanes:
            try:
                _, lat = lane.local_coordinates(vehicle.position)
                if abs(lat) < best_lat:
                    best, best_lat = lane, abs(lat)
            except Exception:
                pass
        return best

    def _connected_lane(self, lane, candidates):
        """Returns the candidate that continues `lane` along the road graph."""

        if lane is None or not candidates:
            return None
        nxt = list(getattr(lane, "next_lanes", None) or [])
        nxt_keys = {self._lane_key(item) for item in nxt}
        for candidate in candidates:
            if candidate in nxt or self._lane_key(candidate) in nxt_keys:
                return candidate
        return None

    def _lane_key(self, lane):
        index = getattr(lane, "index", None)
        if index is None:
            return id(lane)
        return tuple(index)