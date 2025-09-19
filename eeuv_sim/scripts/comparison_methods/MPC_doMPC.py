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
from casadi import SX, mtimes


class AUV6DOFDynamics:
    def __init__(self):
        self.mass = 11.5
        # 只用到平移阻尼；注意改成“正定”阵（线性阻尼系数）
        self.D_lin = np.diag([4.03, 6.22, 5.18])
        # 角向参数保留但本版本MPC不使用
        self.I = np.diag([0.16, 0.16, 0.16])
        self.Ma = -np.diag([5.5, 12.7, 14.57, 0.12, 0.12, 0.12])
        self.D_quad = -np.diag([18.18, 21.66, 36.99, 1.55, 1.55, 1.55])

        # 浮重（默认中性浮力 -> 0）
        self.g = 9.81
        self.W = self.mass * self.g
        self.B = self.W
        self.COG = np.array([0.0, 0.0, 0.0])
        self.COB = np.array([0.0, 0.0, 0.005])

        # 仅平移的净浮重项（世界系z轴向上为正时，此处为0）
        self.g_vec = np.array([0.0, 0.0, -(self.W - self.B)])

    # 下列函数保留以兼容旧接口，但MPC中不再使用它们的非线性形式
    def M_total(self):
        M_rb = np.block([
            [self.mass * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), self.I]
        ])
        return M_rb + self.Ma

    def D_total(self, v):
        return self.D_lin @ v + cp.multiply(self.D_quad @ cp.abs(v), v)

    def g_eta(self):
        fg = np.array([0, 0, -(self.W - self.B), self.COB[1] * self.B, -self.COB[0] * self.B, 0])
        return fg

    # ===== 新：仅作占位，MPC里不再调用此“速度态”更新 =====
    def dynamics(self, v, tau):
        # 保留旧接口，但不在MPC中使用该形式
        A = np.eye(6)
        B = np.eye(6) / self.mass
        return A @ v + B @ tau


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
        self.Q = np.eye(3) * 1.0  # position error
        self.R = np.eye(6) * 0.1  # control effort (Fx, Fy, Fz, Mx, My, Mz)
        self.orientation = [1.0, 0.0, 0.0, 0.0]
        self.dyn = AUV6DOFDynamics()

        # log_path = os.path.join(os.getcwd(), '/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log.h5')
        log_path = os.path.join(os.getcwd(), '/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_doMPC.h5')
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

    def control_loop(self):
        if not self.initialized:
            return
        if self.done:
            return

        self.step_count += 1

        # Exit if reached goal
        if np.linalg.norm(self.position - self.goal) < 1.0:
            self.get_logger().info('Reached goal. Exiting...')
            self.done = True
            self.destroy_node()
            # rclpy.shutdown()
            return

        # Exit if exceeded max steps
        if self.step_count > self.max_steps:
            self.get_logger().warn('Exceeded max steps. Exiting...')
            self.done = True
            self.destroy_node()
            # rclpy.shutdown()
            return

        i_curr = np.argmin(np.linalg.norm(self.trajectory - self.position, axis=1))


        ref_traj = self.trajectory[i_curr: i_curr + self.N]
        if len(ref_traj) < self.N:
            ref_traj = np.pad(ref_traj, ((0, self.N - len(ref_traj)), (0, 0)), mode='edge')

        # 执行 MPC 求解（do-mpc）
        tau = self.solve_mpc(self.position, self.velocity, ref_traj)

        # ===== 机体系 -> 世界系（仅旋转，无平移项）并发布到 /ucat/force_thrust =====
        Fb = np.array(tau[:3], dtype=float)  # [Fx_b, Fy_b, Fz_b]
        Tb = np.array(tau[3:], dtype=float)  # [Mx_b, My_b, Mz_b]

        # 由当前姿态获取 R
        w, x, y, z = self.orientation  # 你在 state_callback 里存的是 [w,x,y,z]
        roll, pitch, yaw = self.quat_to_euler_wxyz(w, x, y, z)
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)  # body -> world

        # 纯旋转变换
        Fw = R @ Fb
        Tw = R @ Tb

        # 发布（世界系；力矩按“关于 CoM/机体原点”处理，不添加 p×F）
        wrench = WrenchStamped()
        wrench.header = Header()
        wrench.header.stamp = self.get_clock().now().to_msg()
        # wrench.header.frame_id = 'world'
        wrench.wrench.force.x = float(Fw[0])
        wrench.wrench.force.y = -float(Fw[1])
        wrench.wrench.force.z = -float(Fw[2])
        wrench.wrench.torque.x = float(Tw[0])
        wrench.wrench.torque.y = -float(Tw[1])
        wrench.wrench.torque.z = -float(Tw[2])
        self.publisher.publish(wrench)


        # # 发布控制力（保持你的符号取反逻辑不变）
        # wrench = WrenchStamped()
        # wrench.header = Header()
        # wrench.header.stamp = self.get_clock().now().to_msg()
        # wrench.wrench.force.x = float(tau[0])
        # wrench.wrench.force.y = -float(tau[1])
        # wrench.wrench.force.z = -float(tau[2])
        # wrench.wrench.torque.x = float(tau[3])
        # wrench.wrench.torque.y = -float(tau[4])
        # wrench.wrench.torque.z = -float(tau[5])

        # self.publisher.publish(wrench)
        # self.get_logger().info_once('Publishing MPC control commands.')

        # Log data
        lin = self.velocity[:3]
        ang = self.velocity[3:]
        self.data["time"].append(0.0)
        self.data["position"].append(self.position.tolist())
        self.data["orientation"].append(self.orientation)
        self.data["linear_velocity"].append(lin.tolist())
        self.data["angular_velocity"].append(ang.tolist())
        self.data["thrusts"].append(tau.tolist())
        self.data["wrench"].append(tau.tolist())

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

    # ==== do-mpc 模型与控制器 ====
    def _build_dmpc(self, dt, N):
        """
        使用 do-mpc 构建离散模型与 MPC 控制器。
        """
        # ---- 1) 模型 ----
        model = do_mpc.model.Model('discrete')

        # 状态
        p = model.set_variable(var_type='_x', var_name='p', shape=(3, 1))
        v = model.set_variable(var_type='_x', var_name='v', shape=(3, 1))

        # 控制输入 u = [Fx,Fy,Fz,Mx,My,Mz]
        u = model.set_variable(var_type='_u', var_name='u', shape=(6, 1))

        # 时变参考（tvp）：位置参考 p_ref
        p_ref = model.set_variable(var_type='_tvp', var_name='p_ref', shape=(3, 1))

        # 常量与矩阵
        m = self.dyn.mass
        D_lin = SX(self.dyn.D_lin)  # 3x3
        g_vec = SX(self.dyn.g_vec.reshape(3, 1))  # 3x1

        u_f = u[0:3]  # 3x1

        # 离散动力学
        p_next = p + dt * v
        v_dot = (u_f - mtimes(D_lin, v) - g_vec) / m
        v_next = v + dt * v_dot

        model.set_rhs('p', p_next)
        model.set_rhs('v', v_next)

        model.setup()

        # ---- 2) MPC 控制器 ----
        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=N,
            t_step=dt,
            store_full_solution=False
        )

        # 目标函数
        Q = SX(self.Q)
        R = SX(self.R)
        e = p - p_ref
        lterm = mtimes([e.T, Q, e]) + mtimes([u.T, R, u])
        mterm = 5.0 * mtimes([e.T, Q, e])

        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u=0.0)  # 不额外重复加权

        # 约束：输入限幅（与原逻辑一致）
        u_lower = -50.0 * np.ones((6, 1))
        u_upper =  50.0 * np.ones((6, 1))
        # 力矩强制为 0（保持6维接口不变）
        u_lower[3:] = 0.0
        u_upper[3:] = 0.0
        mpc.bounds['lower', '_u', 'u'] = u_lower
        mpc.bounds['upper', '_u', 'u'] = u_upper

        # 约束：逐轴速度上限（QP 友好）
        v_max = 3.0
        v_bound = (v_max / np.sqrt(3.0)) * np.ones((3, 1))
        mpc.bounds['lower', '_x', 'v'] = -v_bound
        mpc.bounds['upper', '_x', 'v'] =  v_bound

        # tvp（时变参考）模板与回调
        tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            # 使用当前缓存的窗口参考 self._tvp_buffer: shape (N, 3)
            # do-mpc 要求填充 0..N 的 tvp
            if self._tvp_buffer is None:
                for k in range(N + 1):
                    tvp_template['_tvp', k, 'p_ref'] = np.zeros((3, 1))
                return tvp_template

            ref = self._tvp_buffer  # (N,3)
            for k in range(N + 1):
                idx = min(k, ref.shape[0] - 1)
                tvp_template['_tvp', k, 'p_ref'] = ref[idx].reshape(3, 1)
            return tvp_template

        mpc.set_tvp_fun(tvp_fun)

        mpc.setup()

        # 保存对象以备需要
        self._dmpc_model = model
        return mpc

    # ==== 替换：solve_mpc 使用 do-mpc ====
    def solve_mpc(self, position, velocity, ref_traj):
        """
        使用 do-mpc 计算一步控制量。
        输入:
            position: 当前 p (3,)
            velocity: 当前 [v(3); ω(3)]，此处只用前 3 个线速度
            ref_traj: 形状 (N, 3)，未来 N 步的参考位置
        输出:
            tau: (6,) 力/矩向量（后三维被约束为 0）
        """
        # 1) 把当前窗口参考轨迹塞给 tvp 回调
        if ref_traj.shape[0] < self.N:
            pad = np.repeat(ref_traj[-1][None, :], self.N - ref_traj.shape[0], axis=0)
            ref_use = np.concatenate([ref_traj, pad], axis=0)
        else:
            ref_use = ref_traj[:self.N]
        self._tvp_buffer = ref_use  # (N,3)

        # 2) 当前状态
        p0 = position.reshape(3, 1)
        v0 = velocity[:3].reshape(3, 1)
        x0 = np.vstack([p0, v0])  # 6x1

        # 3) 初始化/滚动
        self.dmpc.x0 = x0
        self.dmpc.set_initial_guess()

        # 4) 求解一步
        try:
            u0 = self.dmpc.make_step(x0)  # 形状 (6,1)
            u0 = np.array(u0).reshape(-1)
        except Exception as e:
            self.get_logger().warn(f'do-mpc optimization failed: {e}')
            return np.zeros(6)

        return u0

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
