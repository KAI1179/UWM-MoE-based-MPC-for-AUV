import numpy as np
import matplotlib.pyplot as plt
# from MPC import MPCController
# from MPC_1 import MPCController
# from MPC_2 import MPCController
# from MPC_3 import MPCController
# from MPC_4 import MPCController
# from MPC_yuanquan import MPCController
# from MPC_LOS import MPCController
# from MPC_doMPC import MPCController
from MPC_5 import MPCController
# from MPC_5_1 import MPCController
# from MPC_6 import MPCController
# from MPC_ECOS import MPCController
from scipy.interpolate import CubicSpline
# from ..data_collector import thruster_wrench_exchange
import rclpy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 构造路径：5个关键点 + 三次样条插值
# waypoints = [
#     [5.0,   0.0,  10.0],    # 起点靠近x最小边界
#     [12.0,  -10.0, 20.0],    # 向右上拐弯并下降
#     [20.0, 10.0, 5.0],     # 向左下折返并上升
#     [28.0,   -5.0, 18.0],    # 向右上方再次转折下降
#     [35.0,   0.0, 8.0]     # 终点靠近x最大边界
# ]

waypoints = [
    [5.0,   0.0,  -10.0],    # 起点靠近x最小边界
    [12.0,  10.0, -20.0],    # 向右上拐弯并下降
    [20.0, -10.0, -5.0],     # 向左下折返并上升
    [28.0,   5.0, -18.0],    # 向右上方再次转折下降
    [35.0,   0.0, -8.0]     # 终点靠近x最大边界
]

# waypoints = [
#     [2.0,  -3.0,  -5.0],
#     [6.0,  5.0, -10.0],
#     [10.0, -5.0, -2.0],
#     [14.0,   2.0, -9.0],
#     [18.0,   0.0,  -4.0]
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

def main(args=None):
    rclpy.init(args=args)
    node = MPCController(waypoints, dt=0.1, horizon=10, max_steps=15000)
    # node = MPCController(waypoints, dt=0.1, horizon=10, max_steps=2000)
    # node = MPCController(waypoints, dt=0.1, horizon=10, max_steps=2000)
    node.reset_ROV()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=0.1)
    finally:
        node.destroy_node()  # 保存数据
        rclpy.shutdown()

    # rclpy.spin(node)
    # node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()
