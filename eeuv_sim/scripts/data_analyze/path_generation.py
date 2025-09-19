import numpy as np
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go

def generate_smooth_3d_trajectory(waypoints, num_points=200):
    """
    使用三次样条在给定3D waypoint上生成平滑轨迹
    """
    waypoints = np.array(waypoints)
    t = np.linspace(0, 1, len(waypoints))  # 参数化每个航点

    # 分别为 x(t), y(t), z(t) 拟合三次样条
    cs_x = CubicSpline(t, waypoints[:, 0])
    cs_y = CubicSpline(t, waypoints[:, 1])
    cs_z = CubicSpline(t, waypoints[:, 2])

    t_smooth = np.linspace(0, 1, num_points)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)
    z_smooth = cs_z(t_smooth)

    trajectory = np.vstack((x_smooth, y_smooth, z_smooth)).T
    return trajectory

# 示例 waypoints
# waypoints = [
#     [0, 0, 0],
#     [50, 15, 5],
#     [80, 5, -5],
#     [120, 10, 0],
#     [150, 0, 0]
# ]

# waypoints = [
#     [5,   0,  -10],    # 起点靠近x最小边界
#     [12,  10, -20],    # 向右上拐弯并下降
#     [20, -10, -5],     # 向左下折返并上升
#     [28,   5, -18],    # 向右上方再次转折下降
#     [35,   0,  -8]     # 终点靠近x最大边界
# ]

# waypoints = [
#     [5,   0,  10],    # 起点靠近x最小边界
#     [12,  -10, 20],    # 向右上拐弯并下降
#     [20, 10, 5],     # 向左下折返并上升
#     [28,   -5, 18],    # 向右上方再次转折下降
#     [35,   0,  8]     # 终点靠近x最大边界
# ]
#
# waypoints = [
#     [34.5,  0.0, 5.0],     # 0°
#     [30.25, 10.25, 5.0],   # 45°
#     [20.0,  14.5, 5.0],    # 90°
#     [9.75,  10.25, 5.0],   # 135°
#     [5.5,   0.0, 5.0],     # 180°
#     [9.75, -10.25, 5.0],   # 225°
#     [20.0, -14.5, 5.0],    # 270°
#     [30.25, -10.25, 5.0]   # 315°
# ]

# waypoints = [
#     [2,  -3,  -5],
#     [6,  5, -10],
#     [10, -5, -2],
#     [14,   2, -9],
#     [18,   0,  -4]
# ]

waypoints = [
    [1,  -3,  -2],
    [3,  2, -5],
    [5, -2, -1],
    [7,   1, -4],
    [9,   -2,  -2]
]

letters_waypoints = {
    "V": [
        [6.0,  10.0, 5.0],
        [8.0,   0.0, 5.0],
        [10.0, 10.0, 5.0]
    ],
    "I": [
        [13.0, 10.0, 5.0],
        [13.0, -10.0, 5.0]
    ],
    "P": [
        [16.0, -10.0, 5.0],
        [16.0,  10.0, 5.0],
        [22.0,  10.0, 5.0],
        [22.0,   0.0, 5.0],
        [16.0,   0.0, 5.0]
    ],
    "L": [
        [25.0, 10.0, 5.0],
        [25.0,-10.0, 5.0],
        [31.0,-10.0, 5.0]
    ]
}

trajectory = generate_smooth_3d_trajectory(waypoints)

# 分离轨迹点
x_smooth, y_smooth, z_smooth = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
way_x, way_y, way_z = zip(*waypoints)

# Plotly 交互式绘图
fig = go.Figure()

# 添加平滑轨迹线
fig.add_trace(go.Scatter3d(
    x=x_smooth, y=y_smooth, z=z_smooth,
    mode='lines',
    name='Smooth Trajectory',
    line=dict(color='blue', width=4)
))

# 添加原始 waypoints 点
fig.add_trace(go.Scatter3d(
    x=way_x, y=way_y, z=way_z,
    mode='markers+text',
    name='Waypoints',
    marker=dict(size=6, color='red'),
    text=[f'P{i}' for i in range(len(waypoints))],
    textposition="top center"
))

# 设置布局
fig.update_layout(
    title='3D Smooth Trajectory (Interactive)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    margin=dict(l=0, r=0, b=0, t=30),
    showlegend=True
)

fig.show()
