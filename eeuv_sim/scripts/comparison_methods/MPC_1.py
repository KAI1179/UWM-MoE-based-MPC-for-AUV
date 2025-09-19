#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import h5py
import numpy as np
import cvxpy as cp
from scipy.interpolate import CubicSpline

import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float32MultiArray, Bool, Header
from eeuv_sim.srv import ResetToPose


# =========================
#   Dynamics (3-DOF trans)
# =========================
class AUV6DOFDynamics:
    """
    这里只使用平移3自由度作为MPC模型：
        p_{k+1} = p_k + dt * v_k
        v_{k+1} = v_k + dt * (1/m) * (u_k - D * v_k - g)
    其中 u=[Fx,Fy,Fz]，g 为净浮重（默认中性浮力则为0）。
    """
    def __init__(self):
        self.mass = 11.5
        # 仅取平移线性阻尼（正定）
        self.D_lin = np.diag([4.03, 6.22, 5.18])

        # 浮重项（默认中性浮力，g_vec=0）
        self.g = 9.81
        self.W = self.mass * self.g
        self.B = self.W  # 中性浮力
        self.g_vec = np.array([0.0, 0.0, -(self.W - self.B)])  # 若中性浮力则为0

    def step(self, p, v, u, dt):
        """离散一步，用于仿真内预测：返回 (p_next, v_next)"""
        vdot = (u - self.D_lin @ v - self.g_vec) / self.mass
        p_next = p + dt * v
        v_next = v + dt * vdot
        return p_next, v_next


# =========================
#        MPC Node
# =========================
class MPCController(Node):
    def __init__(self, waypoints, dt=0.1, horizon=10, max_steps=1500):
        super().__init__('rov_mpc_controller')

        # --- ROS2 I/O ---
        self.publisher = self.create_publisher(WrenchStamped, '/ucat/force_thrust', 2)
        self.reset_pub = self.create_publisher(Bool, '/ucat/reset', 1)
        self.state_sub = self.create_subscription(EntityState, '/ucat/state', self.state_callback, 1)
        self.reset_client = self.create_client(ResetToPose, '/reset_to_pose')
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('reset_to_pose service not available!')

        # --- Params ---
        self.dt = dt
        self.N = horizon
        self.max_steps = max_steps
        self.waypoints = waypoints
        self.trajectory = self.generate_smooth_3d_trajectory(waypoints)
        self.goal = self.trajectory[-1]

        # 代价权重
        self.Qp = np.eye(3) * 5.0    # 位置
        self.Qv = np.eye(3) * 0.5    # 速度
        self.Rf = np.eye(3) * 0.05   # 力
        self.u_max = 20.0            # 力约束 [-20, 20] N

        # 动力学
        self.dyn = AUV6DOFDynamics()

        # 状态缓存
        self.state = None
        self.position = np.zeros(3)
        self.velocity = np.zeros(6)  # 来自话题：前三个是线速度
        self.orientation = [1.0, 0.0, 0.0, 0.0]
        self.initialized = False

        # 轨迹进度指针（单向推进）
        self.ref_idx = 0

        # 日志
        log_path = os.path.join(os.getcwd(),
            '/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log.h5')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.h5file = h5py.File(log_path, 'w')
        self.h5_open = True
        self.data = {k: [] for k in
                     ["time", "position", "orientation", "linear_velocity",
                      "angular_velocity", "thrusts", "wrench"]}

        # 时序
        self.step_count = 0
        self.done = False
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info('MPC controller node initialized.')

    # ----------- Callbacks & Helpers ------------
    def state_callback(self, msg: EntityState):
        self.state = msg
        pos = msg.pose.position
        twist = msg.twist
        ori = msg.pose.orientation

        self.position = np.array([pos.x, pos.y, pos.z])
        self.velocity = np.array([
            twist.linear.x, twist.linear.y, twist.linear.z,
            twist.angular.x, twist.angular.y, twist.angular.z
        ])
        self.orientation = [ori.w, ori.x, ori.y, ori.z]
        self.initialized = True

    def generate_smooth_3d_trajectory(self, waypoints, num_points=500):
        waypoints = np.array(waypoints)
        t = np.linspace(0, 1, len(waypoints))
        cs_x = CubicSpline(t, waypoints[:, 0])
        cs_y = CubicSpline(t, waypoints[:, 1])
        cs_z = CubicSpline(t, waypoints[:, 2])
        t_s = np.linspace(0, 1, num_points)
        traj = np.vstack((cs_x(t_s), cs_y(t_s), cs_z(t_s))).T
        return traj

    def solve_mpc(self, p_now, v_now, ref_traj):
        """
        X = [p(3); v(3)] ∈ R^6, U = u(3)
        线性二次规划（OSQP）
        """
        X = cp.Variable((6, self.N + 1))
        U = cp.Variable((3, self.N))

        constraints = [X[:, 0] == cp.hstack([p_now, v_now])]
        cost = 0

        for k in range(self.N):
            p_k = X[0:3, k]
            v_k = X[3:6, k]
            u_k = U[:, k]

            # 代价
            pos_err = p_k - ref_traj[k]
            cost += cp.quad_form(pos_err, self.Qp)
            cost += cp.quad_form(v_k, self.Qv)
            cost += cp.quad_form(u_k, self.Rf)

            # 离散化动力学（欧拉）
            vdot_k = (u_k - self.dyn.D_lin @ v_k - self.dyn.g_vec) / self.dyn.mass
            p_next = p_k + self.dt * v_k
            v_next = v_k + self.dt * vdot_k
            constraints += [
                X[0:3, k + 1] == p_next,
                X[3:6, k + 1] == v_next
            ]

            # 输入约束
            constraints += [cp.abs(u_k) <= self.u_max]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self.get_logger().warn(f'MPC optimization failed: {prob.status}')
            return np.zeros(3)
        return U[:, 0].value

    def pub_force(self, u3):
        """将3维力发到 /ucat/force_thrust，转矩为0"""
        wrench = WrenchStamped()
        wrench.header = Header()
        wrench.header.stamp = self.get_clock().now().to_msg()
        wrench.wrench.force.x = float(u3[0])
        wrench.wrench.force.y = -float(u3[1])
        wrench.wrench.force.z = -float(u3[2])
        wrench.wrench.torque.x = 0.0
        wrench.wrench.torque.y = 0.0
        wrench.wrench.torque.z = 0.0
        self.publisher.publish(wrench)

    def control_loop(self):
        if not self.initialized or self.done:
            return

        # 终止条件
        if np.linalg.norm(self.position - self.goal) < 1.0:
            self.get_logger().info('Reached goal. Exiting...')
            self.done = True
            self.destroy_node()
            return
        if self.step_count > self.max_steps:
            self.get_logger().warn('Exceeded max steps. Exiting...')
            self.done = True
            self.destroy_node()
            return
        self.step_count += 1

        # 参考窗口（单向推进，避免跳窗）
        i_near = int(np.argmin(np.linalg.norm(self.trajectory - self.position, axis=1)))
        self.ref_idx = max(self.ref_idx, i_near)
        ref_traj = self.trajectory[self.ref_idx:self.ref_idx + self.N]
        if len(ref_traj) < self.N:
            ref_traj = np.pad(ref_traj, ((0, self.N - len(ref_traj)), (0, 0)), mode='edge')

        # 求解MPC并发布
        p_now = self.position.copy()
        v_now = self.velocity[:3].copy()
        u = self.solve_mpc(p_now, v_now, ref_traj)
        self.pub_force(u)

        # 记录
        lin = self.velocity[:3]
        ang = self.velocity[3:]
        self.data["time"].append(float(self.step_count) * self.dt)
        self.data["position"].append(self.position.tolist())
        self.data["orientation"].append(self.orientation)
        self.data["linear_velocity"].append(lin.tolist())
        self.data["angular_velocity"].append(ang.tolist())
        self.data["thrusts"].append(np.r_[u, 0.0, 0.0, 0.0].tolist())
        self.data["wrench"].append(np.r_[u, 0.0, 0.0, 0.0].tolist())

    # ----------- Reset & Shutdown ------------
    def reset_ROV(self):
        """将ROV重置到第一个航点（不再对 y/z 取反；保持坐标系一致）"""
        req = ResetToPose.Request()
        x, y, z = self.waypoints[0]
        req.x = float(x)