import numpy as np
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go

def generate_smooth_3d_trajectory(waypoints, num_points=500):
    """
    使用三次样条在给定3D waypoint上生成平滑轨迹
    """
    waypoints = np.array(waypoints, dtype=float)
    t = np.linspace(0, 1, len(waypoints))  # 参数化每个航点

    cs_x = CubicSpline(t, waypoints[:, 0])
    cs_y = CubicSpline(t, waypoints[:, 1])
    cs_z = CubicSpline(t, waypoints[:, 2])

    t_smooth = np.linspace(0, 1, num_points)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)
    z_smooth = cs_z(t_smooth)

    trajectory = np.vstack((x_smooth, y_smooth, z_smooth)).T
    return trajectory

# --- 字母航点（不连续） ---
# letters_waypoints = {
#     "V": [
#         [6.0,  10.0, 5.0],
#         [8.0,   0.0, 5.0],
#         [10.0, 10.0, 5.0]
#     ],
#     "I": [
#         [13.0, 10.0, 5.0],
#         [13.0, -10.0, 5.0]
#     ],
#     "P": [
#         [16.0, -10.0, 5.0],
#         [16.0,  10.0, 5.0],
#         [22.0,  10.0, 5.0],
#         [22.0,   0.0, 5.0],
#         [16.0,   0.0, 5.0]
#     ],
#     "L": [
#         [25.0, 10.0, 5.0],
#         [25.0,-10.0, 5.0],
#         [31.0,-10.0, 5.0]
#     ]
# }
letters_waypoints = {
    # V: 底部两个点 -> 顶部中点
    "V": [
        [6.0,  10.0, 5.0],
        [9.0, -10.0, 5.0],
        [12.0, 10.0, 5.0]
    ],
    # I: 顶到底
    "I": [
        [16.0, 10.0, 5.0],
        [16.0,-10.0, 5.0]
    ],
    # P: 竖线 + 上半圆
    "P": [
        [20.0, -10.0, 5.0],    # 底
        [20.0,  10.0, 5.0],    # 顶
        [26.0,  10.0, 5.0],    # 顶右
        [26.0,   0.0, 5.0],    # 中右
        [20.0,   0.0, 5.0]     # 中左
    ],
    # L: 竖线 + 底线
    "L": [
        [30.0, 10.0, 5.0],
        [30.0,-10.0, 5.0],
        [36.0,-10.0, 5.0]
    ]
}

# 可选：每个字母的采样点数（直的少一些，弯的多一些）
letter_num_points = {"V": 200, "I": 100, "P": 300, "L": 150}

# 生成每个字母的平滑轨迹
letters_trajectories = {
    letter: generate_smooth_3d_trajectory(wps, num_points=letter_num_points.get(letter, 200))
    for letter, wps in letters_waypoints.items()
}

# --- 绘图 ---
fig = go.Figure()

# 为每个字母分别添加轨迹线与原始航点（互不相连）
for i, (letter, wps) in enumerate(letters_waypoints.items()):
    traj = letters_trajectories[letter]
    x_s, y_s, z_s = traj[:,0], traj[:,1], traj[:,2]
    way_x, way_y, way_z = zip(*wps)

    # 平滑曲线
    fig.add_trace(go.Scatter3d(
        x=x_s, y=y_s, z=z_s,
        mode='lines',
        name=f'{letter} Trajectory',
        line=dict(width=5)  # 颜色交给 Plotly 自动分配，避免手动指定
    ))

    # 航点
    fig.add_trace(go.Scatter3d(
        x=way_x, y=way_y, z=way_z,
        mode='markers+text',
        name=f'{letter} Waypoints',
        marker=dict(size=6),
        text=[f'{letter}{j}' for j in range(len(wps))],
        textposition="top center",
        showlegend=False
    ))

# 轴与布局
fig.update_layout(
    title='VIPL 非连续 3D 参考轨迹',
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='data'  # 保持各轴比例一致，字母不变形
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=True
)

fig.show()
