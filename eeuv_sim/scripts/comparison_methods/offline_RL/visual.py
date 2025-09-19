#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
import h5py

def generate_smooth_3d_trajectory(waypoints, num_points=500):
    waypoints = np.array(waypoints)
    t = np.linspace(0, 1, len(waypoints))
    cs_x = CubicSpline(t, waypoints[:, 0])
    cs_y = CubicSpline(t, waypoints[:, 1])
    cs_z = CubicSpline(t, waypoints[:, 2])
    t_smooth = np.linspace(0, 1, num_points)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)
    z_smooth = cs_z(t_smooth)
    return np.vstack((x_smooth, y_smooth, z_smooth)).T

def read_actual_path(h5_file):
    with h5py.File(h5_file, 'r') as f:
        # 优先兼容 /run/position（与评估节点保存一致）；否则回退到根 /position
        if "run" in f and "position" in f["run"]:
            return np.array(f["run"]["position"])
        return np.array(f["position"])

# === 第一条轨迹：参考轨迹 ===
waypoints = [
    [5,   0,  -10],
    [12,  10, -20],
    [20, -10, -5],
    [28,   5, -18],
    [35,   0,  -8]
]

trajectory_ref = generate_smooth_3d_trajectory(waypoints)
way_x, way_y, way_z = zip(*waypoints)

# === 第二条轨迹：仿真采集实际轨迹 ===
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/offline_RL/eval_logs/rov_eval_20250821_120455.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/offline_RL/eval_logs/rov_eval_20250821_143859.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/offline_RL/eval_logs/rov_eval_20250910_214321.h5"  ## 使用两个hdf5文件训练的，
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/offline_RL/eval_logs/rov_eval_20250910_221327.h5"  ## 使用两个hdf5文件训练的，
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/offline_RL/eval_logs/rov_eval_20250918_163028.h5"  ##
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/offline_RL/eval_logs/rov_eval_20250918_212111.h5"  ##
h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/offline_RL/eval_logs/rov_eval_20250918_213610.h5"  ##
trajectory_actual = read_actual_path(h5_path)

# === 绘图 ===
fig = go.Figure()

# 参考轨迹
fig.add_trace(go.Scatter3d(
    x=trajectory_ref[:, 0], y=trajectory_ref[:, 1], z=trajectory_ref[:, 2],
    mode='lines',
    name='Reference Trajectory',
    line=dict(color='blue', width=4)
))
# 参考轨迹关键点
fig.add_trace(go.Scatter3d(
    x=way_x, y=way_y, z=way_z,
    mode='markers+text',
    name='Waypoints',
    marker=dict(size=6, color='red'),
    text=[f'P{i}' for i in range(len(waypoints))],
    textposition="top center"
))

# 实际轨迹
fig.add_trace(go.Scatter3d(
    x=trajectory_actual[:, 0], y=trajectory_actual[:, 1], z=trajectory_actual[:, 2],
    mode='lines+markers',
    name='Actual Trajectory',
    line=dict(color='green', width=3),
    marker=dict(size=3, color='green')
))

fig.update_layout(
    title='Reference vs Actual ROV Trajectory',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=True
)

fig.show()
