from collections import deque
import numpy as np

class FrameStack:

    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, initial_observation):
        """Fills the buffer with (stack_size - 1) zero frames followed by the initial
        observation and returns the first stacked state."""

        self.frames.clear()

        zero_frame = np.zeros_like(initial_observation, dtype=np.float32)
        for _ in range(self.stack_size - 1):
            self.frames.append(zero_frame)
        
        self.frames.append(initial_observation.astype(np.float32))

        return self._get_stacked_state()
    
    def step(self, observation):
        """Appends the new observation to the rolling buffer, dropping the oldest frame,
        and returns the updated stacked state."""
        
        self.frames.append(observation.astype(np.float32))

        return self._get_stacked_state()
    
    def _get_stacked_state(self):
        return np.concatenate(list(self.frames),axis=0)