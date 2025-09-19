import h5py
import numpy as np

log_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_4.h5"

with h5py.File(log_path, "r") as f:
    lin_vel = np.array(f["linear_velocity"])  # shape: (N, 3)

# 每个时刻的速度模长（瞬时速度）
speed = np.linalg.norm(lin_vel, axis=1)  # 单位 m/s

# 平均速度（标量）
avg_speed = np.mean(speed)

print(f"ROV 平均速度: {avg_speed:.3f} m/s")
