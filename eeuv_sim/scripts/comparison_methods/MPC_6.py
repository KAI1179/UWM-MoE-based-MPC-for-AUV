import numpy as np
import cvxpy as cp
import rclpy
import time, math
from rclpy.node import Node
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float32MultiArray, Bool, Header
from eeuv_sim.srv import ResetToPose
# from ..data_collector.thruster_wrench_exchange import ThrusterWrenchCalculator
from scipy.interpolate import CubicSpline

import h5py
from ament_index_python.packages import get_package_share_directory
import os

# ==== do-mpc / casadi ====
import do_mpc
from casadi import SX, mtimes, vertcat, horzcat, dot, sqrt, sumsqr

class AUV6DOFDynamics:
    def __init__(self):
        self.mass = 11.5
        # 平移线性阻尼（正定）
        self.D_lin = np.diag([4.03, 6.22, 5.18])
        # 角向转动惯量（简化为对角）
        self.I = np.diag([0.16, 0.16, 0.16])
        # 角速度线性阻尼（从原 D_quad 角向项抽一个近似线性阻尼）
        self.D_ang = np.diag([1.55, 1.55, 1.55])

        # 浮重：默认中性浮力
        self.g = 9.81
        self.W = self.mass * self.g
        self.B = self.W

        # world z 向上为正，净浮重为 0
        self.g_vec_world = np.array([0.0, 0.0, 0.0])

    # 机体参数（便于 do-mpc 中使用）
    def inv_inertia(self):
        return np.linalg.inv(self.I)


class MPCController(Node):
    def __init__(self, waypoints=[0.0, 0.0], dt=0.1, horizon=10, max_steps=1500):
        super().__init__('rov_data_collector')
        self.publisher = self.create_publisher(Float32MultiArray, '/ucat/thruster_cmd', 1)
        self.reset_pub = self.create_publisher(Bool, '/ucat/reset', 1)
        self.state_sub = self.create_subscription(EntityState, '/ucat/state', self.state_callback, 1)
        self.state = None
        self.start_time = None
        self.waypoints = waypoints
        # self.thru_to_wrench = ThrusterWrenchCalculator(
        #     '/home/xukai/ros2_ws/src/eeuv_sim/data/dynamics/BlueDynamics.yaml')
        self.reset_client = self.create_client(ResetToPose, '/reset_to_pose')
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('set_entity_state service not available!')

        self.publisher = self.create_publisher(WrenchStamped, '/ucat/force_thrust', 2)
        self.trajectory = self.generate_smooth_3d_trajectory(waypoints)
        # self.trajectory = self.generate_smooth_3d_trajectory([
        #     [5.0, 0.0, -10.0],
        #     [12.0, 10.0, -20.0],
        #     [20.0, -10.0, -5.0],
        #     [28.0, 5.0, -18.0],
        #     [35.0, 0.0, -8.0]
        # ])
        self.initialized = False

        self.h5_open = True

        self.timer_period = dt
        self.timer = self.create_timer(self.timer_period, self.control_loop)
        self.get_logger().info('MPC controller node initialized.')

        self.dt = dt
        self.N = horizon  # prediction horizon


        # self.Q = np.eye(3) * 1.0  # position error
        # self.R = np.eye(6) * 0.1  # control effort (Fx, Fy, Fz, Mx, My, Mz)
        # self.orientation = [1.0, 0.0, 0.0, 0.0]

        # 代价权重
        self.Qp = np.eye(3) * 1.0  # 位置误差
        self.Qq = 20.0  # 姿态误差权重（标量，对 1-(q⋅q_ref)^2）
        self.Ru = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 控制代价（力/力矩）
        self.orientation = [1.0, 0.0, 0.0, 0.0]  # wxyz

        self.dyn = AUV6DOFDynamics()

        # log_path = os.path.join(os.getcwd(), '/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log.h5')
        log_path = os.path.join(os.getcwd(), '/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_6.h5')
        self.h5file = h5py.File(log_path, 'w')
        self.data = {
            "time": [],
            "position": [],
            "orientation": [],
            "linear_velocity": [],
            "angular_velocity": [],
            "thrusts": [],
            "wrench": []
        }

        self.goal = self.trajectory[-1]
        self.step_count = 0
        self.max_steps = max_steps
        self.done = False

        # ==== do-mpc 控制器构建 ====
        self._tvp_buffer = None  # 供 tvp_fun 使用（窗口参考轨迹）
        self.dmpc = self._build_dmpc(self.dt, self.N)



    def state_callback(self, msg):
        self.state = msg
        # self.last_state_time = time.time()

        pos = msg.pose.position
        twist = msg.twist
        ori = msg.pose.orientation

        self.position = np.array([pos.x, pos.y, pos.z])
        self.velocity = np.array([
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z
        ])
        self.orientation = [ori.w, ori.x, ori.y, ori.z]
        self.initialized = True

    def pub_thrust_force(self, tau):

        wrench = WrenchStamped()
        wrench.header = Header()
        wrench.header.stamp = self.get_clock().now().to_msg()
        wrench.wrench.force.x = float(tau[0])
        wrench.wrench.force.y = float(tau[1])
        wrench.wrench.force.z = float(tau[2])
        wrench.wrench.torque.x = float(tau[3])
        wrench.wrench.torque.y = float(tau[4])
        wrench.wrench.torque.z = float(tau[5])

        self.publisher.publish(wrench)
        # self.get_logger().info_once('Publishing MPC control commands.')

    def pub_thrust_force_world(self, Fw, Tw):
        wrench = WrenchStamped()
        wrench.header = Header()
        wrench.header.stamp = self.get_clock().now().to_msg()
        wrench.wrench.force.x = float(Fw[0])
        wrench.wrench.force.y = float(Fw[1])
        wrench.wrench.force.z = float(Fw[2])
        wrench.wrench.torque.x = float(Tw[0])
        wrench.wrench.torque.y = float(Tw[1])
        wrench.wrench.torque.z = float(Tw[2])
        self.publisher.publish(wrench)

    def _rotmat_from_quat_wxyz(self, w, x, y, z):
        # body->world 旋转矩阵（右手系）
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ], dtype=float)

    def quat_to_euler_wxyz(self, w, x, y, z):
        # 右手系，RPY顺序为 roll(x)-pitch(y)-yaw(z)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return roll, pitch, yaw

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """
        Function to convert Euler angles to a rotation matrix.

        Parameters: roll (float)
        roll (float): Roll angle in degrees
        pitch (float): pitch angle in degrees
        yaw (float): Yaw angle in degrees

        Returns: rotation_matrix: rotation matrix
        rotation_matrix: rotation matrix (3x3 numpy array).
        """
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ])

        Ry = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])

        Rz = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1]
        ])

        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        return rotation_matrix

    # ====== 主控制循环（基本不变，改用 6DoF 解与发布） ======
    def control_loop(self):
        if not self.initialized or self.done:
            return

        self.step_count += 1

        print(self.step_count)

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

        i_curr = np.argmin(np.linalg.norm(self.trajectory - self.position, axis=1))
        ref_traj = self.trajectory[i_curr: i_curr + self.N]
        if len(ref_traj) < self.N:
            ref_traj = np.pad(ref_traj, ((0, self.N - len(ref_traj)), (0, 0)), mode='edge')

        # 姿态参考：默认恒等四元数（水平朝向），如需沿轨迹定向可在此生成 q_ref 序列
        ref_q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (self.N, 1))

        # 6DoF MPC
        tau = self.solve_mpc(self.position, self.velocity, self.orientation, ref_traj, ref_q)  # (6,)

        # 机体 -> 世界旋转并发布
        Fb = np.array(tau[:3], dtype=float)
        Tb = np.array(tau[3:], dtype=float)
        w, x, y, z = self.orientation
        R = self._rotmat_from_quat_wxyz(w, x, y, z)  # body->world
        Fw = R @ Fb
        Tw = R @ Tb
        self.pub_thrust_force_world(Fw, Tw)

        # 日志
        lin = self.velocity[:3];
        ang = self.velocity[3:]
        self.data["time"].append(0.0)
        self.data["position"].append(self.position.tolist())
        self.data["orientation"].append(self.orientation)
        self.data["linear_velocity"].append(lin.tolist())
        self.data["angular_velocity"].append(ang.tolist())
        self.data["thrusts"].append(tau.tolist())
        self.data["wrench"].append(np.concatenate([Fw, Tw]).tolist())

    # ====== do-mpc：构建 6DoF 模型与控制器 ======
    def _build_dmpc(self, dt, N):
        model = do_mpc.model.Model('discrete')

        # 状态
        p = model.set_variable(var_type='_x', var_name='p', shape=(3, 1))  # 位置（world）
        q = model.set_variable(var_type='_x', var_name='q', shape=(4, 1))  # 四元数 wxyz
        v = model.set_variable(var_type='_x', var_name='v', shape=(3, 1))  # 线速度（body）
        w_ = model.set_variable(var_type='_x', var_name='w', shape=(3, 1))  # 角速度（body）

        # 控制
        u = model.set_variable(var_type='_u', var_name='u', shape=(6, 1))  # [F_b(3); M_b(3)]

        # tvp 参考
        p_ref = model.set_variable(var_type='_tvp', var_name='p_ref', shape=(3, 1))
        q_ref = model.set_variable(var_type='_tvp', var_name='q_ref', shape=(4, 1))

        # 常量
        m = self.dyn.mass
        D_lin = SX(self.dyn.D_lin)
        D_ang = SX(self.dyn.D_ang)
        I_inv = SX(self.dyn.inv_inertia())

        # ===== 旋转矩阵 R(q): body->world（wxyz）
        qw, qx, qy, qz = q[0, 0], q[1, 0], q[2, 0], q[3, 0]
        R = SX(3, 3)
        R[0, 0] = 1 - 2 * (qy * qy + qz * qz);
        R[0, 1] = 2 * (qx * qy - qw * qz);
        R[0, 2] = 2 * (qx * qz + qw * qy)
        R[1, 0] = 2 * (qx * qy + qw * qz);
        R[1, 1] = 1 - 2 * (qx * qx + qz * qz);
        R[1, 2] = 2 * (qy * qz - qw * qx)
        R[2, 0] = 2 * (qx * qz - qw * qy);
        R[2, 1] = 2 * (qy * qz + qw * qx);
        R[2, 2] = 1 - 2 * (qx * qx + qy * qy)

        # ===== 四元数微分：q_dot = 0.5 * Ω(w) * q
        # wx, wy, wz = w_[0, 0], w_[1, 0], w_[2, 0]
        # Omega = SX([
        #     [0, -wx, -wy, -wz],
        #     [wx, 0, wz, -wy],
        #     [wy, -wz, 0, wx],
        #     [wz, wy, -wx, 0]
        # ])
        # q_dot = 0.5 * mtimes(Omega, q)  # 4x1

        wx, wy, wz = w_[0, 0], w_[1, 0], w_[2, 0]

        row1 = horzcat(SX(0), -wx, -wy, -wz)
        row2 = horzcat(wx, SX(0), wz, -wy)
        row3 = horzcat(wy, -wz, SX(0), wx)
        row4 = horzcat(wz, wy, -wx, SX(0))
        Omega = vertcat(row1, row2, row3, row4)

        q_dot = 0.5 * mtimes(Omega, q)

        # ===== 动力学离散化（欧拉前向 + 四元数归一化）
        u_f = u[0:3];
        u_m = u[3:6]  # 3x1, 3x1

        p_next = p + dt * mtimes(R, v)  # world
        v_dot = (u_f - mtimes(D_lin, v)) / m  # body
        v_next = v + dt * v_dot

        w_dot = mtimes(I_inv, (u_m - mtimes(D_ang, w_)))  # body
        w_next = w_ + dt * w_dot

        q_plus = q + dt * q_dot
        q_norm = sqrt(sumsqr(q_plus) + 1e-12)  # 防止除 0
        q_next = q_plus / q_norm

        model.set_rhs('p', p_next)
        model.set_rhs('q', q_next)
        model.set_rhs('v', v_next)
        model.set_rhs('w', w_next)

        model.setup()

        # ===== MPC 控制器 =====
        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(n_horizon=N, t_step=dt, store_full_solution=False)

        # 目标函数
        Qp = SX(self.Qp)
        Ru = SX(self.Ru)

        e_p = p - p_ref  # 3x1
        # 四元数误差：1 - (q^T q_ref)^2（双覆盖不变）
        e_q_scalar = 1.0 - (dot(q, q_ref)) ** 2

        lterm = mtimes([e_p.T, Qp, e_p]) + self.Qq * e_q_scalar + mtimes([u.T, Ru, u])
        mterm = 5.0 * mtimes([e_p.T, Qp, e_p]) + 5.0 * self.Qq * e_q_scalar

        mpc.set_objective(mterm=mterm, lterm=lterm)

        # 约束
        u_lower = -50.0 * np.ones((6, 1))
        u_upper = 50.0 * np.ones((6, 1))
        mpc.bounds['lower', '_u', 'u'] = u_lower
        mpc.bounds['upper', '_u', 'u'] = u_upper

        v_max = 3.0
        mpc.bounds['lower', '_x', 'v'] = - (v_max / np.sqrt(3.0)) * np.ones((3, 1))
        mpc.bounds['upper', '_x', 'v'] = (v_max / np.sqrt(3.0)) * np.ones((3, 1))

        w_max = 2.0  # rad/s
        mpc.bounds['lower', '_x', 'w'] = - w_max * np.ones((3, 1))
        mpc.bounds['upper', '_x', 'w'] = w_max * np.ones((3, 1))

        # tvp
        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            if self._tvp_buffer is None:
                for k in range(N + 1):
                    tvp_template['_tvp', k, 'p_ref'] = np.zeros((3, 1))
                    tvp_template['_tvp', k, 'q_ref'] = np.array([[1.0], [0.0], [0.0], [0.0]])
                return tvp_template

            ref_p, ref_q = self._tvp_buffer  # (N,3), (N,4)
            for k in range(N + 1):
                ip = min(k, ref_p.shape[0] - 1)
                iq = min(k, ref_q.shape[0] - 1)
                tvp_template['_tvp', k, 'p_ref'] = ref_p[ip].reshape(3, 1)
                tvp_template['_tvp', k, 'q_ref'] = ref_q[iq].reshape(4, 1)
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()

        self._dmpc_model = model
        return mpc

    # ====== 6DoF MPC 求解 ======
    def solve_mpc(self, position, velocity, quat_wxyz, ref_traj_p, ref_traj_q):
        # 缓存参考窗口（供 tvp_fun）
        if ref_traj_p.shape[0] < self.N:
            padp = np.repeat(ref_traj_p[-1][None, :], self.N - ref_traj_p.shape[0], axis=0)
            ref_p_use = np.vstack([ref_traj_p, padp])
        else:
            ref_p_use = ref_traj_p[:self.N]
        if ref_traj_q.shape[0] < self.N:
            padq = np.repeat(ref_traj_q[-1][None, :], self.N - ref_traj_q.shape[0], axis=0)
            ref_q_use = np.vstack([ref_traj_q, padq])
        else:
            ref_q_use = ref_traj_q[:self.N]
        self._tvp_buffer = (ref_p_use, ref_q_use)

        # 当前状态
        p0 = position.reshape(3, 1)
        v0 = velocity[:3].reshape(3, 1)
        w0 = velocity[3:].reshape(3, 1)
        q0 = np.array(quat_wxyz, dtype=float).reshape(4, 1)

        x0 = np.vstack([p0, q0, v0, w0])  # 13x1

        self.dmpc.x0 = x0
        self.dmpc.set_initial_guess()

        try:
            u0 = self.dmpc.make_step(x0)  # 6x1
            u0 = np.array(u0).reshape(-1)
        except Exception as e:
            self.get_logger().warn(f'do-mpc optimization failed: {e}')
            return np.zeros(6)

        return u0

    def reset_ROV(self):
        """
        Resets the environment: clears sim state, sets random ROV position.
        """
        req = ResetToPose.Request()

        x, y, z = self.waypoints[0]

        req.x = x
        req.y = -y
        req.z = -z
        req.roll = 0.0
        req.pitch = 0.0
        req.yaw = 0.0

        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(f'Reset ROV to: x={x:.2f}, y={y:.2f}, z={z:.2f}')

        self.state = None  # 清空旧状态
        timeout = time.time() + 3.0  # 最多等待3秒
        while self.state is None and time.time() < timeout:
            # rclpy.spin_once(self, timeout_sec=0.1)
            rclpy.spin_once(self)
        if self.state is None:
            self.get_logger().warn("No updated state received after reset!")

        return True

    def generate_smooth_3d_trajectory(self, waypoints, num_points=500):
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

    def destroy_node(self):

        if not getattr(self, "h5_open", False):
            super().destroy_node()
            return

        for key, value in self.data.items():
            self.h5file.create_dataset(key, data=np.array(value))

        self.h5file.flush()
        self.h5file.close()
        self.h5_open = False
        super().destroy_node()
