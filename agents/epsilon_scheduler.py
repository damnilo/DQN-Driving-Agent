import numpy as np

class EpsilonScheduler:

    def __init__(self, start, end, decay, warmup_steps=1_000):

        self.start = start
        self.end = end
        self.decay = decay
        self.warmup_steps = warmup_steps

    def get_epsilon(self, step):
        """Holds ε at `start` during the warmup window, then linearly interpolates it
        down to `end` over `decay` steps."""
        
        if step < self.warmup_steps:
            return self.start

        return float(np.interp(step, [self.warmup_steps, self.warmup_steps + self.decay], [self.start, self.end]))