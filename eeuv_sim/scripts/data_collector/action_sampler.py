import numpy as np

class SmoothActionSampler:
    def __init__(self, thrust_limit=20.0, alpha=0.9):
        self.alpha = alpha
        self.thrust_limit = thrust_limit
        self.last_action = np.zeros(8)

    def smooth_sample(self):
        noise = np.random.uniform(-1, 1, size=8)
        new_action = self.alpha * self.last_action + (1 - self.alpha) * noise * self.thrust_limit
        self.last_action = new_action
        return new_action

    def random_sample(self):
        return np.random.uniform(-self.thrust_limit, self.thrust_limit, size=8)