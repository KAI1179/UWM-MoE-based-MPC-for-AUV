# import numpy as np
#
# class SmoothActionSampler:
#     def __init__(self, thrust_limit=20.0, alpha=0.9):
#         self.alpha = alpha
#         self.thrust_limit = thrust_limit
#         self.last_action = np.zeros(8)
#
#     def sample(self):
#         noise = np.random.uniform(-1, 1, size=8)
#         new_action = self.alpha * self.last_action + (1 - self.alpha) * noise * self.thrust_limit
#         self.last_action = new_action
#         return new_action


import numpy as np

class SmoothActionSampler:
    def __init__(self, thrust_limit=20.0, alpha=0.8, noise_shift_rate=0.01):  ## 0.9 0.01
        self.alpha = alpha
        self.thrust_limit = thrust_limit
        self.noise_shift_rate = noise_shift_rate  # 控制目标扰动的更新频率
        self.last_action = np.zeros(8)
        self.target_noise = np.random.uniform(-1, 1, size=8)  # 初始化目标扰动方向

    def sample(self):
        # 慢慢更新目标噪声方向，保留其趋势
        # self.target_noise = (
        #     (1 - self.noise_shift_rate) * self.target_noise +
        #     self.noise_shift_rate * np.random.uniform(-1, 1, size=8)
        # )
        #
        # # 当前扰动是目标噪声乘以推力限制
        # noise = self.target_noise * self.thrust_limit

        # 融合历史动作和平滑扰动，生成新动作
        # new_action = self.alpha * self.last_action + (1 - self.alpha) * noise
        # self.last_action = new_action

        new_action = np.random.uniform(-1, 1, size=8) * self.thrust_limit

        return new_action

