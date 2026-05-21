from collections import deque
import numpy as np

class FrameStack:

    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, initial_observation):
        self.frames.clear()

        for _ in range(self.stack_size):
            self.frames.append(initial_observation)

        return self._get_stacked_state()
    
    def step(self, observation):
        self.frames.append(observation)

        return self._get_stacked_state()
    
    def _get_stacked_state(self):
        return np.concatenate(list(self.frames),axis=0)