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
        # 假设数据是平铺保存的，不分 episode
        return np.array(f["position"])

# === 第一条轨迹：参考轨迹 ===
waypoints = [
    [5,   0,  -10],
    [12,  10, -20],
    [20, -10, -5],
    [28,   5, -18],
    [35,   0,  -8]
]
# waypoints = [
#     [5,   0,  10],
#     [12,  -10, 20],
#     [20, 10, 5],
#     [28,   -5, 18],
#     [35,   0,  8]
# ]


# waypoints = [
#     [2,  -3,  -5],
#     [6,  5, -10],
#     [10, -5, -2],
#     [14,   2, -9],
#     [18,   0,  -4]
# ]

# waypoints = [
#     [34.5,  -0.0, -15.0],     # 0°
#     [30.25, -10.25, -15.0],   # 45°
#     [20.0,  -14.5, -15.0],    # 90°
#     [9.75,  -10.25, -15.0],   # 135°
#     [5.5,   0.0, -15.0],     # 180°
#     [9.75, 10.25, -15.0],   # 225°
#     [20.0, 14.5, -15.0],    # 270°
#     [30.25, 10.25, -15.0]   # 315°
# ]

trajectory_ref = generate_smooth_3d_trajectory(waypoints)
way_x, way_y, way_z = zip(*waypoints)

# === 第二条轨迹：仿真采集实际轨迹 ===
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_5.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_0815_1300.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_0815_1332.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0818_2134.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0818_2142.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0818_2147.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0819_1942.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0819_2001.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0819_2005.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0819_2008.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0819_2011.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0825_1944.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0825_1950.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0825_1950.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_MoE_rov_log_0902_2050.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_MoE_rov_log_0902_2100.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_MoE_rov_log_0903_1130.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_MoE_rov_log_0903_1140.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_MoE_rov_log_0903_1150.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_MoE_rov_log_0904_1320.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_MoE_rov_log_0904_1330.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_MoE_rov_log_0904_1340.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/RL/logs/ppo_rov_log_20250825_211604_ep00.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/RL/logs/ppo_rov_log_20250825_212627_ep00.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/RL/logs/ppo_rov_log_20250825_212627_ep00.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/RL/logs/ppo_rov_log_20250828_085239_ep00.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_5_1.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_6.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_doMPC.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_LOS.h5"
# h5_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_ecos.h5'
h5_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/PPO_RL/logs/eval_runs/eval_ep001.h5'
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
