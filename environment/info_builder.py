import numpy as np

NAV_CMD_TO_FLOAT = {
    "LEFT": -1.0, "left": -1.0,
    "RIGHT": 1.0, "right": 1.0,
    "STRAIGHT": 0.0, "forward": 0.0,
    "IDLE": 0.0,
}

class InfoBuilder:
    def build(self, env, info:dict) -> dict:
        """Extends the raw MetaDrive info dict with computed fields: normalised speed,
        heading error, lateral offset, lane-center ratio, navigation command as a float,
        and the agent's longitudinal position on the current lane."""
        
        agent = env.agent
        nav = agent.navigation
        out = dict(info)

        out["velocity"] = float(
            info.get("velocity", getattr(agent, "speed", 0.0) or 0.0)
        )

        out["speed"] = out["velocity"]
        out["steering"] = float(info.get("steering", getattr(agent, "steering", 0.0) or 0.0))

        out["acceleration"] = float(info.get("acceleration", 0.0))

        raw = info.get("raw_action")
        if raw is not None:
            out["raw_action"] = tuple(raw)

        out["crash"] = bool(info.get("crash", False))
        out["out_of_road"] = bool(info.get("out_of_road", False))
        out["arrive_dest"] = bool(info.get("arrive_dest", False))
        out["max_step"] = bool(info.get("max_step", False))

        cmd = str(info.get("navigation_command", "IDLE"))
        out["navigation_command"] = cmd
        out["navigation_command_float"] = NAV_CMD_TO_FLOAT.get(cmd, 0.0)
        
        out["lateral_offset"] = self._lateral_offset(agent, nav)
        out["heading_error"] = self._heading_error(agent, nav)

        out["dist_left"] = float(getattr(agent, "dist_to_left_side", 0.0) or 0.0)
        out["dist_right"] = float(getattr(agent, "dist_to_right_side", 0.0) or 0.0)

        lane_w = float(getattr(agent, "lane_width", 3.5) or 3.5)
        out["lane_width"] = lane_w
        out["lane_center_ratio"] = (out["dist_left"] - out["dist_right"]) / max(lane_w, 0.1)

        out["step_reward"] = float(info.get("step_reward", 0.0))
        out["episode_reward"] = float(info.get("episode_reward", 0.0))

        lane = agent.lane
        long, _ = lane.local_coordinates(agent.position)
        out["longitudinal"] = long

        return out

    def _lateral_offset(self, agent, nav) -> float:
        """Returns the signed lateral distance from the lane centre, preferring the
        navigation object's attribute if present and falling back to lane local
        coordinates."""
        
        try:
            if hasattr(nav, "current_lateral"):
                return float(nav.current_lateral)
            
            lane = nav.current_lane
            _, lat = lane.local_coordinates(agent.position)
            return float(lat)
        except Exception:
            return 0.0

    def _heading_error(self, agent, nav) -> float:
        """Computes the signed difference between the lane heading at the agent's
        longitudinal position and the agent's own heading, wrapped to [-π, π]."""
        
        try:
            lane = nav.current_lane
            long, _ = lane.local_coordinates(agent.position)
            lane_heading = lane.heading_theta_at(long)
            agent_heading = float(agent.heading_theta)

            return float(np.arctan2(
                np.sin(lane_heading - agent_heading),
                np.cos(lane_heading - agent_heading)
            ))
        except Exception:
            return 0.0