import numpy as np
import cvxpy as cp
import rclpy
import time
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
        self.publisher = self.create_publisher(Float32MultiArray, '/ucat/thruster_cmd', 1)
        self.reset_pub = self.create_publisher(Bool, '/ucat/reset', 1)
        self.state_sub = self.create_subscription(EntityState, '/ucat/state', self.state_callback, 1)
        self.state = None
        self.start_time = None
        self.waypoints = waypoints

        self.reset_client = self.create_client(ResetToPose, '/reset_to_pose')
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('set_entity_state service not available!')

        # === 发布力矩的 publisher（保持原样与话题）===
        self.publisher = self.create_publisher(WrenchStamped, '/ucat/force_thrust', 2)

        self.trajectory = self.generate_smooth_3d_trajectory(waypoints)
        self.initialized = False

        self.h5_open = True
        self.timer_period = dt
        self.timer = self.create_timer(self.timer_period, self.control_loop)
        self.get_logger().info('MPC controller node initialized.')

        # === MPC 参数 ===
        self.dt = dt
        self.N = horizon  # 预测域
        self.Q = np.eye(3) * 1.0  # 位置误差权重
        self.R = np.eye(6) * 0.1  # 控制代价 (Fx,Fy,Fz,Mx,My,Mz)

        self.orientation = [1.0, 0.0, 0.0, 0.0]
        self.dyn = AUV6DOFDynamics()

        # 日志
        log_path = os.path.join(os.getcwd(),
                                '/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log_1.h5')
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

        # === 参考轨迹“单调”索引 ===
        self.i_curr = 0

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

    def state_callback(self, msg):
        self.state = msg
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
        if not self.initialized or self.done:
            return

        self.step_count += 1

        # 达到目标或超步数直接退出
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

        # ====== 参考轨迹索引：只在前向窗口内找最近点，单调前进 ======
        win = 200  # 查找窗口大小，可按路径密度调整
        i_end = min(self.i_curr + win, len(self.trajectory))
        dist = np.linalg.norm(self.trajectory[self.i_curr:i_end] - self.position, axis=1)
        self.i_curr = self.i_curr + int(np.argmin(dist))
        ref_traj = self.trajectory[self.i_curr:self.i_curr + self.N]
        if len(ref_traj) < self.N:
            ref_traj = np.pad(ref_traj, ((0, self.N - len(ref_traj)), (0, 0)), mode='edge')

        # ====== MPC 求解 ======
        tau = self.solve_mpc(self.position, self.velocity, ref_traj)

        # ====== 发布（保持你原来的取负号逻辑与话题/单位不变）======
        wrench = WrenchStamped()
        wrench.header = Header()
        wrench.header.stamp = self.get_clock().now().to_msg()
        wrench.wrench.force.x = float(tau[0])
        wrench.wrench.force.y = -float(tau[1])
        wrench.wrench.force.z = -float(tau[2])
        wrench.wrench.torque.x = float(tau[3])
        wrench.wrench.torque.y = -float(tau[4])
        wrench.wrench.torque.z = -float(tau[5])
        self.publisher.publish(wrench)

        # ====== 记录数据 ======
        lin = self.velocity[:3]
        ang = self.velocity[3:]
        self.data["time"].append(0.0)
        self.data["position"].append(self.position.tolist())
        self.data["orientation"].append(self.orientation)
        self.data["linear_velocity"].append(lin.tolist())
        self.data["angular_velocity"].append(ang.tolist())
        self.data["thrusts"].append(tau.tolist())
        self.data["wrench"].append(tau.tolist())

    def solve_mpc(self, position, velocity, ref_traj):
        """
        线性化二次阻尼的 MPC（接口保持不变）
        状态: x=[p(3); v(3)]
        控制: u=[Fx,Fy,Fz,Mx,My,Mz]（扭矩约束仍为0）
        动力学:
          p_{k+1} = p_k + dt * v_k
          v_{k+1} = v_k + dt/m * ( u_f - D_lin*v_k - D_quad_lin(v_k) - g_vec )
        其中 D_quad_lin(v) ≈ Kq(v̄) * v + c_drag(v̄)，v̄ 取当前测得速度（世界系）
        这样问题仍是 QP，可由 OSQP 稳定求解。
        """
        dt = self.timer_period
        m = self.dyn.mass

        # ===== 速度与阻尼线性化点（用当前测得的线速度）=====
        v_bar = velocity[:3]  # 世界系速度；你已确认坐标系没问题，就直接用
        # 从 self.dyn.D_quad 中取出平移轴对应的“系数绝对值”，作为 kq（正值）
        # 原始 D_quad 是负对角阵（-diag([...]))，故取反得到正系数
        kq_diag_full = -np.diag(self.dyn.D_quad)  # 6维
        kq = kq_diag_full[:3]  # 只用前三个平移轴
        # 线性化：f(v)=k|v|v
        Kq = np.diag(2.0 * kq * np.abs(v_bar))  # 对 v 的“等效线性阻尼”
        f_bar = kq * np.abs(v_bar) * v_bar  # 在 v_bar 处的二次阻尼
        c_drag = f_bar - Kq @ v_bar  # 常数项

        # ===== 变量 =====
        x = cp.Variable((6, self.N + 1))  # [p; v]
        u = cp.Variable((6, self.N))  # [Fx,Fy,Fz,Mx,My,Mz]

        # ===== 约束与代价 =====
        constraints = [x[:, 0] == cp.hstack([position, velocity[:3]])]
        cost = 0.0

        # 力/速度限制（可按需要再调）
        u_limit = 60.0  # 推力上限（放宽到更接近推进器能力）
        v_max = 2.0  # 期望最大速度
        v_bound = v_max / np.sqrt(3.0)

        for k in range(self.N):
            p_k = x[:3, k]
            v_k = x[3:, k]
            u_k = u[:, k]
            u_f = u_k[:3]  # 仅平移力

            # 位置误差代价
            pos_err = p_k - ref_traj[k]
            cost += cp.quad_form(pos_err, self.Q)

            # 控制代价
            cost += cp.quad_form(u_k, self.R)

            # 离散动力学（线性 + 线性化二次阻尼）
            vdot = (u_f - self.dyn.D_lin @ v_k - Kq @ v_k - c_drag - self.dyn.g_vec) / m
            p_next = p_k + dt * v_k
            v_next = v_k + dt * vdot

            constraints += [
                x[:3, k + 1] == p_next,
                x[3:, k + 1] == v_next
            ]

            # 输入与速度约束
            constraints += [cp.abs(u_k) <= u_limit]
            constraints += [u_k[3:] == 0.0]  # 扭矩仍置0，保持接口
            constraints += [cp.abs(v_k) <= v_bound]

        # 终端位置代价（提高贴合）
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
