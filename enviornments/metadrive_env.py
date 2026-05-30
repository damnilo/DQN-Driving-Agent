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

        self.obs_size: int = None

        self._last_discrete_action = 0

    def reset(self):
        self._last_discrete_action = 0
        self.reward_function.reset()
        self.observation_builder.reset()
        raw_obs, info = self.env.reset()
        info = self._enrich_info(info)
        processed_obs = self.observation_builder.build(
            self.env, raw_obs, info, prev_action_idx = 0
        )

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

    def num_actions(self) -> int:
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
        reward = self.reward_function.compute(info, env_reward, continuous_action)
        processed_obs = self.observation_builder.build(self.env, raw_obs, info, prev_action_idx=self._last_discrete_action)
        return processed_obs, reward, terminated, truncated, info

    def _enrich_info(self, info):
        return self.info_builder.build(self.env, info)

    def get_map(self):
        return self.env.config["map"]