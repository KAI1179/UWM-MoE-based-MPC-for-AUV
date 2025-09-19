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

        log_path = os.path.join(os.getcwd(), '/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_5.h5')
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

        i_curr = np.argmin(np.linalg.norm(self.trajectory - self.position, axis=1))

        ref_traj = self.trajectory[i_curr: i_curr + self.N]
        if len(ref_traj) < self.N:
            ref_traj = np.pad(ref_traj, ((0, self.N - len(ref_traj)), (0, 0)), mode='edge')

        # 执行 MPC 求解
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


        # # 发布控制力
        # wrench = WrenchStamped()
        # wrench.header = Header()
        # wrench.header.stamp = self.get_clock().now().to_msg()
        # wrench.wrench.force.x = float(tau[0])
        # wrench.wrench.force.y = -float(tau[1])
        # wrench.wrench.force.z = -float(tau[2])
        # wrench.wrench.torque.x = float(tau[3])
        # wrench.wrench.torque.y = -float(tau[4])
        # wrench.wrench.torque.z = -float(tau[5])

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
        速度上限版 MPC（保持原有结构/取反/日志等不变）
        - 状态 x = [p(3); v(3)]
        - 控制 u = [Fx,Fy,Fz,Mx,My,Mz]（扭矩约束为 0）
        - 动力学：p_{k+1}=p_k+dt*v_k
                  v_{k+1}=v_k+dt*(1/m)*(u_f - D*v_k - g_vec)
        - 目标：以位置误差为主 + 控制力惩罚
        - 约束：|u| ≤ 20（原有） + 新增逐轴速度上限 |v_i| ≤ v_bound
        """
        dt = self.timer_period

        # ===== 速度上限（可改）=====
        v_max = 2.0  # 希望的“最大速度” (m/s)
        v_bound = v_max / np.sqrt(3.0)  # 逐轴上限，保证 ||v||2 ≤ v_max（保守）

        # 变量
        x = cp.Variable((6, self.N + 1))  # [p; v]
        u = cp.Variable((6, self.N))  # [Fx,Fy,Fz,Mx,My,Mz]

        # 初始条件
        p0 = position
        v0 = velocity[:3]
        constraints = [x[:, 0] == cp.hstack([p0, v0])]
        cost = 0

        for k in range(self.N):
            p_k = x[:3, k]
            v_k = x[3:, k]
            u_k = u[:, k]
            u_f = u_k[:3]  # 力

            # 扭矩强制为 0（保持6维接口不变）
            constraints += [u_k[3:] == 0.0]

            # 位置误差（保持使用 self.Q）
            pos_err = p_k - ref_traj[k]
            cost += cp.quad_form(pos_err, self.Q)

            # 控制代价（保持使用 self.R）
            cost += cp.quad_form(u_k, self.R)

            # 动力学离散（线性阻尼 + 欧拉）
            vdot = (u_f - self.dyn.D_lin @ v_k - self.dyn.g_vec) / self.dyn.mass
            p_next = p_k + dt * v_k
            v_next = v_k + dt * vdot
            constraints += [
                x[:3, k + 1] == p_next,
                x[3:, k + 1] == v_next
            ]

            # 输入约束：与原逻辑一致（±20）
            # constraints += [cp.abs(u_k) <= 20.0]
            constraints += [cp.abs(u_k) <= 50.0]

            # 新增：逐轴速度上限（QP 友好）
            constraints += [cp.abs(v_k) <= v_bound]

        # 可选：末端位置代价（有助于贴合窗口尾点）
        p_T = x[:3, self.N]
        pos_err_T = p_T - ref_traj[-1]
        cost += cp.quad_form(pos_err_T, 5.0 * self.Q)

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
