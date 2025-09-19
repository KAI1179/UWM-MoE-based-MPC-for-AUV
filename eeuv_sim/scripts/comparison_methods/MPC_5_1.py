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

import numpy as np
import math

class AUV6DOFDynamics:
    """
    Fossen 6-DOF 模型（严格建模）
    动力学形式： M(ν) ν̇ + C(ν) ν + D(ν) ν + g(η) = τ
      - ν = [u, v, w, p, q, r]^T （体坐标）
      - η = [x, y, z, φ, θ, ψ]^T（世界坐标，RPY）
    兼容性：
      * 保留 mass, D_lin(3x3), I, Ma, D_quad（系数为正）, g_vec, 以及 M_total, D_total(v), g_eta(), dynamics()
      * 另外提供 M_RB, M_A, C_RB(ν), C_A(ν), D_matrix(ν), g_eta_full(roll, pitch, yaw)
    """

    def __init__(self):
        # ===== 基本参数（来自 YAML）=====
        self.mass = 11.5
        self.g = 9.81
        self.W = self.mass * self.g
        self.B = self.W  # 中性浮力
        self.COG = np.array([0.0, 0.0, 0.0])   # r_g
        self.COB = np.array([0.0, 0.0, 0.005]) # r_b

        # 刚体转动惯量（在 COG 处）
        self.I = np.diag([0.16, 0.16, 0.16])

        # ===== 附加质量（Fossen 记号： M_A = -diag(X_ud, Y_vd, Z_wd, K_pd, M_qd, N_rd) ）=====
        self.Ma = -np.diag([
            5.5,   # Xud
            12.7,  # Yvd
            14.57, # Zwd
            0.12,  # Kpd
            0.12,  # Mqd
            0.12   # Nrd
        ])

        # ===== 阻尼（按 Fossen： D(ν) ν；系数均为正，写在对角阵里）=====
        # 线性阻尼（取 YAML 绝对值；只用于 MPC 线性化的前三轴也单独提供）
        self._Dlin_full = np.diag([
            4.03,  # |Xu|
            6.22,  # |Yv|
            5.18,  # |Zw|
            0.07,  # |Kp|
            0.07,  # |Mq|
            0.07   # |Nr|
        ])
        # 仅平移的 3×3 线性阻尼（与你的 MPC 线性模型对齐）
        self.D_lin = np.diag([4.03, 6.22, 5.18])

        # 二次阻尼（对角：系数为正；D_quad 用作系数集合，最终 D(ν) 会做 |ν| 逐项）
        self.D_quad = np.diag([
            18.18,  # Xuu
            21.66,  # Yvv
            36.99,  # Zww
            1.55,   # Kpp
            1.55,   # Mqq
            1.55    # Nrr
        ])

        # 旧接口中用于“重/浮力净项”的简化常量（世界系 z 向上约定）
        self.g_vec = np.array([0.0, 0.0, -(self.W - self.B)])

    # ---------- 工具：反对称矩阵 ----------
    @staticmethod
    def _skew(v3):
        x, y, z = v3
        return np.array([[0, -z,  y],
                         [z,  0, -x],
                         [-y, x,  0]], dtype=float)

    # ---------- 刚体质量矩阵 M_RB ----------
    def M_RB(self):
        Srg = self._skew(self.COG)  # 这里 COG=0 => 交叉项为 0
        mI = self.mass * np.eye(3)
        upper = np.hstack((mI, -mI @ Srg))
        lower = np.hstack(( mI @ Srg, self.I))
        return np.vstack((upper, lower))

    # ---------- 附加质量矩阵 M_A ----------
    def M_A(self):
        return self.Ma  # 已按 Fossen 号设为负对角阵

    # ---------- 科氏/离心项 C_RB(ν) ----------
    def C_RB(self, nu):
        """
        nu = [u,v,w,p,q,r]
        在 r_g = 0 时：
          C_RB = [[ 0_3, -m*S(ν1)],
                  [ -m*S(ν1), -S(I*ν2) ]]
        """
        nu = np.asarray(nu).reshape(6,)
        v = nu[:3]
        w = nu[3:]
        m = self.mass
        Iv = self.I @ w
        zero3 = np.zeros((3,3))
        upper = np.hstack((zero3, -m * self._skew(v)))
        lower = np.hstack((-m * self._skew(v), -self._skew(Iv)))
        return np.vstack((upper, lower))

    # ---------- 附加质量科氏/离心项 C_A(ν) ----------
    def C_A(self, nu):
        """
        对角附加质量的常见简化（Fossen 书式）：
          C_A = [[ 0_3, -S(A11*ν1) ],
                 [ -S(A11*ν1), -S(A22*ν2) ]]
        其中 A11=diag(-Xud,-Yvd,-Zwd)，A22=diag(-Kpd,-Mqd,-Nrd)
        """
        nu = np.asarray(nu).reshape(6,)
        v = nu[:3]
        w = nu[3:]
        A11 = -np.diag(np.diag(self.Ma)[:3])  # = diag(Xud,Yvd,Zwd) 的正数
        A22 = -np.diag(np.diag(self.Ma)[3:]) # = diag(Kpd,Mqd,Nrd) 的正数
        Av = A11 @ v
        Aw = A22 @ w
        zero3 = np.zeros((3,3))
        upper = np.hstack((zero3, -self._skew(Av)))
        lower = np.hstack((-self._skew(Av), -self._skew(Aw)))
        return np.vstack((upper, lower))

    # ---------- 阻尼矩阵 D(ν)（线性+二次，对角） ----------
    def D_matrix(self, nu):
        """
        返回对角 D(ν)，使得 D(ν) ν = D_lin*ν + diag(D_quad*|ν|)*ν
        * 系数均为正；与 Fossen 方程符号一致
        """
        nu = np.asarray(nu).reshape(6,)
        Dlin = self._Dlin_full.copy()
        Dquad = np.diag(self.D_quad) * np.abs(nu)
        return Dlin + np.diag(Dquad)

    # ---------- 重/浮力项 g(η)（严格 Fossen） ----------
    def g_eta_full(self, roll, pitch, yaw):
        """
        g(η) = [ R^T * ( [0,0,(W-B)]^T )
                 r_g × (R^T*[0,0,W]^T)  +  r_b × (R^T*[0,0,-B]^T) ]
        这里采用世界 z 向上、体坐标右手系
        """
        # 旋转矩阵：body->world
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [ 0,   0, 1]])
        Ry = np.array([[ cp, 0, sp],
                       [  0, 1,  0],
                       [-sp, 0, cp]])
        Rx = np.array([[1, 0,  0],
                       [0, cr, -sr],
                       [0, sr,  cr]])
        R = Rz @ Ry @ Rx

        ez = np.array([0.0, 0.0, 1.0])
        fg_world = (self.W - self.B) * ez         # 世界系的净竖向力
        fg_body  = R.T @ fg_world                  # 转到体坐标（Fossen 用 body 表达）
        Wb =  self.W * (R.T @ ez)                  # 体系下重力方向
        Bb = -self.B * (R.T @ ez)                  # 体系下浮力方向（向上取负）
        tau_g = np.hstack((
            fg_body,
            np.cross(self.COG, Wb) + np.cross(self.COB, Bb)
        ))
        return tau_g

    # =========================
    # ===== 兼容性接口 ========
    # =========================

    def M_total(self):
        """ 返回 M_RB + M_A """
        return self.M_RB() + self.M_A()

    def D_total(self, v):
        """
        兼容旧接口：给定 6 维速度 v，返回 D(ν)ν
        * 如 v 只有 3 维（线速度），则仅返回前三轴的线性项（供老代码偶发调用）
        """
        v = np.asarray(v).reshape(-1,)
        if v.shape[0] == 3:
            return self.D_lin @ v
        Dnu = self.D_matrix(v) @ v
        return Dnu

    def g_eta(self):
        """
        兼容旧接口：返回在“中性浮力 + 小角度/或无需姿态”的简化广义力
        与旧代码一致：只保留 [0,0,-(W-B), τx, τy, 0]，且在中性浮力时为 0
        """
        return np.array([0, 0, -(self.W - self.B), self.COB[1] * self.B, -self.COB[0] * self.B, 0], dtype=float)

    def dynamics(self, v, tau):
        """
        旧占位接口（不用于 MPC）：x_{k+1} = x_k + (τ)/m
        仅为兼容保留
        """
        A = np.eye(6)
        B = np.eye(6) / self.mass
        return A @ v + B @ tau


class MPCController(Node):
    def __init__(self, waypoints=[0.0, 0.0], dt=0.1, horizon=10, max_steps=1500):
        super().__init__('rov_data_collector')
        self.publisher_cmd = self.create_publisher(Float32MultiArray, '/ucat/thruster_cmd', 1)
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
        #     [5.0, 0.0, -10.0],  # 起点靠近x最小边界
        #     [12.0, 10.0, -20.0],  # 向右上拐弯并下降
        #     [20.0, -10.0, -5.0],  # 向左下折返并上升
        #     [28.0, 5.0, -18.0],  # 向右上方再次转折下降
        #     [35.0, 0.0, -8.0]  # 终点靠近x最大边界
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

        log_path = os.path.join(os.getcwd(), '/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_5_1.h5')
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

        self.ref_index = 0  # 当前跟踪的轨迹索引
        self.search_ahead = 80  # 只在 [ref_index, ref_index+search_ahead] 内找最近点
        self.advance_thresh = 0.5  # 离当前参考点足够近就推进索引（米）

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

    def control_loop(self):
        if not self.initialized:
            return
        if self.done:
            return

        self.step_count += 1

        print(self.step_count)

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

        # i_curr = np.argmin(np.linalg.norm(self.trajectory - self.position, axis=1))
        #
        # ref_traj = self.trajectory[i_curr: i_curr + self.N]
        # if len(ref_traj) < self.N:
        #     ref_traj = np.pad(ref_traj, ((0, self.N - len(ref_traj)), (0, 0)), mode='edge')

        # 仅在前视窗口内找最近点，避免跨越整条轨迹
        i_start = self.ref_index
        i_stop = min(i_start + self.search_ahead, len(self.trajectory) - 1)
        seg = self.trajectory[i_start: i_stop + 1]
        local_i = np.argmin(np.linalg.norm(seg - self.position, axis=1))
        i_curr = i_start + int(local_i)

        # 根据接近程度推进 ref_index，保证索引单调递增
        if np.linalg.norm(self.position - self.trajectory[self.ref_index]) < self.advance_thresh:
            self.ref_index = min(self.ref_index + 1, len(self.trajectory) - 1)
        # 同时也允许“跳到”窗口内更靠前的最近点，但不后退
        self.ref_index = max(self.ref_index, i_curr)

        # 从 ref_index 向前取 N 步参考
        ref_traj = self.trajectory[self.ref_index: self.ref_index + self.N]
        if len(ref_traj) < self.N:
            ref_traj = np.pad(ref_traj, ((0, self.N - len(ref_traj)), (0, 0)), mode='edge')

        # 执行 MPC 求解
        tau = self.solve_mpc(self.position, self.velocity, ref_traj)

        # ===== 机体系 -> 世界系（仅旋转，无平移项）并发布到 /ucat/force_thrust =====
        Fb = np.array(tau[:3], dtype=float)  # [Fx_b, Fy_b, Fz_b]
        Tb = np.array(tau[3:], dtype=float)  # [Mx_b, My_b, Mz_b]

        # 由当前姿态获取 R
        w, x, y, z = self.orientation  # 你在 state_callback 里存的是 [w,x,y,z]

        R = self._rotmat_from_quat_wxyz(w, x, y, z)  # body -> world

        # roll, pitch, yaw = self.quat_to_euler_wxyz(w, x, y, z)
        # R = self.euler_to_rotation_matrix(roll, pitch, yaw)  # body -> world

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

    def generate_smooth_3d_trajectory(self, waypoints, num_points=300):
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

    def solve_mpc(self, position, velocity, ref_traj):
        """
        真 6-DOF MPC + 姿态稳态代价：
        - 位置：跟踪 ref_traj
        - 姿态：roll/pitch -> 0，yaw -> 轨迹切向（可改为保持当前/给定 yaw）
        - 角速度：惩罚 p,q,r
        其他约束/发布保持原样
        """
        dt = self.timer_period

        # === 当前姿态/旋转 ===
        wq, xq, yq, zq = self.orientation
        roll, pitch, yaw = self.quat_to_euler_wxyz(wq, xq, yq, zq)  # 弧度
        R_bw = self._rotmat_from_quat_wxyz(wq, xq, yq, zq)  # body->world
        R_wb = R_bw.T

        # === 初始状态 ===
        eta0 = np.array([position[0], position[1], position[2], roll, pitch, yaw], dtype=float)
        v_world = np.asarray(velocity[:3], dtype=float)
        w_world = np.asarray(velocity[3:], dtype=float)
        v_body = R_wb @ v_world
        w_body = R_wb @ w_world
        nu0 = np.hstack([v_body, w_body])

        # === Fossen 模型冻结线性化 ===
        M = self.dyn.M_total()
        Minv = np.linalg.inv(M)

        sphi, cphi = np.sin(roll), np.cos(roll)
        sth, cth = np.sin(pitch), np.cos(pitch)
        eps = 1e-6
        cth = np.clip(cth, eps, None)
        T_rpy = np.array([
            [1.0, sphi * np.tan(pitch), cphi * np.tan(pitch)],
            [0.0, cphi, -sphi],
            [0.0, sphi / cth, cphi / cth]
        ], dtype=float)
        J0 = np.block([
            [R_bw, np.zeros((3, 3))],
            [np.zeros((3, 3)), T_rpy]
        ])

        C0 = self.dyn.C_RB(nu0) + self.dyn.C_A(nu0)
        D0 = self.dyn.D_matrix(nu0)
        g0 = self.dyn.g_eta_full(roll, pitch, yaw)

        A_eta_nu = J0
        A_nu_nu = np.eye(6) - dt * (Minv @ (C0 + D0))
        B_nu = dt * Minv
        c_nu = -dt * (Minv @ g0)

        # === 姿态参考（弧度）===
        # yaw_ref: 轨迹切向
        diffs = np.diff(ref_traj, axis=0, append=ref_traj[-1:])
        yaw_ref_seq = np.arctan2(diffs[:self.N, 1], diffs[:self.N, 0])
        roll_ref_seq = np.zeros(self.N)
        pitch_ref_seq = np.zeros(self.N)

        # 若想“保持当前 yaw”，用下面一行替换即可：
        # yaw_ref_seq[:] = yaw

        # === 权重（可按需要微调）===
        Q_pos = self.Q  # 位置（保持与你原来一致）
        Q_att = np.diag([2.0, 2.0, 0.5])  # 姿态 roll/pitch/yaw（弧度）
        Q_omega = np.diag([0.1, 0.1, 0.1])  # 角速度 p,q,r
        R_u = self.R  # 控制代价（保持原来）

        # === 决策变量 ===
        eta = cp.Variable((6, self.N + 1))
        nu = cp.Variable((6, self.N + 1))
        u = cp.Variable((6, self.N))

        constraints = [
            eta[:, 0] == eta0,
            nu[:, 0] == nu0
        ]

        v_max = 2.0
        v_bound = v_max / np.sqrt(3.0)

        cost = 0
        for k in range(self.N):
            # 位置代价
            pos_err = eta[:3, k] - ref_traj[k]
            cost += cp.quad_form(pos_err, Q_pos)

            # 姿态代价（roll/pitch -> 0，yaw -> yaw_ref）
            att_err = cp.hstack([
                eta[3, k] - roll_ref_seq[k],
                eta[4, k] - pitch_ref_seq[k],
                eta[5, k] - yaw_ref_seq[k]
            ])
            cost += cp.quad_form(att_err, Q_att)

            # 角速度代价（抑制 p,q,r 抖动）
            cost += cp.quad_form(nu[3:6, k], Q_omega)

            # 控制代价
            cost += cp.quad_form(u[:, k], R_u)

            # 动力学约束
            constraints += [
                eta[:, k + 1] == eta[:, k] + dt * (A_eta_nu @ nu[:, k]),
                nu[:, k + 1] == A_nu_nu @ nu[:, k] + B_nu @ u[:, k] + c_nu
            ]

            # 控制与速度约束（保持原逻辑）
            constraints += [cp.abs(u[:, k]) <= 50.0]
            constraints += [cp.abs(nu[:3, k]) <= v_bound]

        # 末端位置 + 姿态终端代价（更稳）
        pos_err_T = eta[:3, self.N] - ref_traj[-1]
        cost += cp.quad_form(pos_err_T, 10.0 * Q_pos)

        att_err_T = cp.hstack([
            eta[3, self.N] - roll_ref_seq[-1],
            eta[4, self.N] - pitch_ref_seq[-1],
            eta[5, self.N] - yaw_ref_seq[-1]
        ])
        cost += cp.quad_form(att_err_T, 5.0 * Q_att)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        if prob.status != cp.OPTIMAL:
            self.get_logger().warn('MPC optimization failed.')
            return np.zeros(6)

        return u[:, 0].value

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
