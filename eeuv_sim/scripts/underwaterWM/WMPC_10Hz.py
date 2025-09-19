#!/usr/bin/env python3
# WMPC.py
# Model-Predictive Control (shooting/CEM) using the pretrained underwater world model
# for path following along a smooth 3D trajectory defined by five waypoints.
#
# Usage (ROS 2):
#   ros2 run eeuv_sim wmpc   (if you make an entry point), or:
#   python3 scripts/underwaterWM/WMPC.py
#
# Key topics/services:
#   Subscribes: /ucat/state           (gazebo_msgs/EntityState)
#   Publishes:  /ucat/force_thrust    (geometry_msgs/WrenchStamped)   ← body-frame wrench
#   Service:    /reset_to_pose        (eeuv_sim/srv/ResetToPose)      ← reset to first waypoint
#
# Notes:
# - This controller plans in thruster-force space (dim = number of thrusters, usually 8),
#   rolls out the learned world model to evaluate a cost, then publishes the equivalent wrench.
# - The thruster→wrench mapping is loaded from the same YAML as the rest of the project.
# - Coordinates: AUVMotion publishes /ucat/state with y,z flipped and quaternion(q) from (roll, -pitch, -yaw).
#   We undo that here to build the internal [p(3), q(4), v(3), w(3)] = 13D state for the world model.

import os
import time
import math
import yaml
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Bool, Header, Float32MultiArray
from eeuv_sim.srv import ResetToPose
import h5py

from ament_index_python.packages import get_package_share_directory
from scipy.interpolate import CubicSpline

# from scripts.comparison_methods.visual_path_compar import waypoints
# --- World model imports (local) ---
# Expect this file to live next to worldModel.py / utils.py etc.
from worldModel import WMConfig, ROVGRUModel
# from utils import quat_normalize_np
from utils import quat_normalize_np, Standardizer

# Thruster <-> wrench utility
from scripts.data_collector.thruster_wrench_exchange import ThrusterWrenchCalculator
# from ..data_collector.thruster_wrench_exchange import ThrusterWrenchCalculator

def quat_to_euler(q):
    """
    Convert quaternion [x,y,z,w] (ROS order) to Euler roll, pitch, yaw (radians).
    """
    x, y, z, w = q
    # ZYX convention
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

def euler_to_quat(roll, pitch, yaw):
    """
    Convert Euler roll, pitch, yaw to quaternion [w,x,y,z] (model code uses [w,x,y,z]).
    """
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)

class WMPCController(Node):
    def __init__(self,
                 waypoints,
                 wm_ckpt_path: str,
                 yaml_dynamics: str = 'BlueDynamics.yaml',
                 dt: float = 0.1,
                 horizon: int = 10,
                 max_steps: int = 2000,
                 cem_iters: int = 4,
                 cem_pop: int = 128,
                 cem_elite: int = 32,
                 action_smooth_weight: float = 1.0,
                 log_path: str = None):
        super().__init__('wmpc_controller')

        # --- ROS I/O ---
        # self.pub_wrench = self.create_publisher(WrenchStamped, '/ucat/force_thrust', 2)
        self.pub_wrench = self.create_publisher(WrenchStamped, '/ucat/force_thrust', 2)
        self.publisher_cmd = self.create_publisher(Float32MultiArray, '/ucat/thruster_cmd', 2)
        self.pub_reset  = self.create_publisher(Bool, '/ucat/reset', 1)
        self.sub_state  = self.create_subscription(EntityState, '/ucat/state', self._on_state, 2)
        self.reset_cli  = self.create_client(ResetToPose, '/reset_to_pose')

        # --- Planning params ---
        self.dt = float(dt)
        self.N  = int(horizon)
        self.max_steps = int(max_steps)

        # --- Cost weights ---
        self.Q_pos = np.diag([4.0, 4.0, 4.0])      # position error
        self.Q_vel = np.diag([0.2, 0.2, 0.2])      # linear vel
        self.Q_ang = np.diag([0.5, 0.5, 0.1])      # angular vel
        self.R     = 1e-4                           # control magnitude per thruster
        self.Rd    = 1e-3 * action_smooth_weight    # action smoothness

        # 新增：姿态稳定相关权重
        self.W_tilt = 5.0  # 惩罚“机体Z轴相对世界Z轴”的倾斜（roll/pitch）
        self.W_att_smooth = 2.0  # 惩罚相邻时刻姿态（四元数）变化

        # --- Waypoints & trajectory ---
        self.waypoints = np.asarray(waypoints, dtype=np.float32).reshape(-1, 3)
        # self.trajectory = self._build_smooth_trajectory(self.waypoints, num_points=300)
        # self.trajectory = self.generate_smooth_3d_trajectory(waypoints, num_points=300)
        self.trajectory = self.generate_smooth_3d_trajectory(self.waypoints, num_points=300)

        # --- World model ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_world_model(wm_ckpt_path).to(self.device)
        self.model.eval()

        self._inject_std_into_controller(wm_ckpt_path, std_path=None)

        # --- Thruster mapping (YAML) ---
        yaml_path = os.path.join(get_package_share_directory('eeuv_sim'),
                                 'data', 'dynamics', yaml_dynamics)
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f'Cannot find YAML dynamics: {yaml_path}')
        self.tw_calc = ThrusterWrenchCalculator(yaml_path)
        self.num_thrusters = self.tw_calc.number_of_thrusters
        self.thrust_limits = np.array(self.tw_calc.thrust_limits, dtype=np.float32)  # shape (N,2)
        self.u_low  = self.thrust_limits[:, 0]
        self.u_high = self.thrust_limits[:, 1]

        # --- Internal state ---
        self.state_msg = None           # latest EntityState
        self.step = 0
        self.done = False

        self.cem_pop = cem_pop
        self.cem_elite = cem_elite
        self.action_smooth_weight = action_smooth_weight
        self.cem_iters = cem_iters

        self.start_time = time.time()
        if log_path is None:
            base = os.path.join(os.getcwd(), 'scripts', 'underwaterWM', 'results')
            os.makedirs(base, exist_ok=True)
            log_path = os.path.join(base, 'wmpc_log.h5')
        self._log_path = log_path
        self._h5file = h5py.File(self._log_path, 'w')
        self._data = {
            "time": [],
            "position": [],
            "orientation_wxyz": [],
            "linear_velocity": [],
            "angular_velocity": [],
            "thrusters": [],
            "wrench_body": [],
            "ref_index": [],
            "ref_point": [],
        }
        self.get_logger().info(f'Logging to {self._log_path}')

        # Reset to the first waypoint
        self._reset_to_first_waypoint()

        # Initial action sequence mean/std for CEM (T,N_thrusters)
        self.act_mu  = np.zeros((self.N, self.num_thrusters), dtype=np.float32)
        self.act_std = np.tile((self.u_high - self.u_low)[None, :] * 0.25, (self.N, 1)).astype(np.float32)

        # log save path
        # log_path: str = None

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.initialized = False
        self.done = False
        self.goal = self.trajectory[-1]

        self.fast_forward = 1.0
        self.time_optimize_value = 0.0
        self.dt_optimize_gain = 0.05

        self.h5_open = True

        self.get_logger().info('WMPC ready.')

    # ---------- Standardization helpers (minimal invasive) ----------
    def _inject_std_into_controller(self, wm_ckpt_path: str, std_path: str = None):
        """
        加载训练时使用的 standardizer；优先顺序：显式 std_path → 环境变量 WM_STD → ckpt 同目录 standardizer.npz
        仅缓存用于 z-score 的均值/方差到 device，不改动你的其它成员或状态机。
        """
        import os, numpy as np, torch
        if std_path is None or not os.path.exists(std_path):
            std_path = os.environ.get('WM_STD', None)
        if std_path is None or not os.path.exists(std_path):
            std_path = os.path.join(os.path.dirname(wm_ckpt_path), 'standardizer.npz')
        if not os.path.exists(std_path):
            raise FileNotFoundError(
                f"Standardizer not found: {std_path}. 请设置 WM_STD 或把 standardizer.npz 放到 ckpt 同目录。"
            )
        self.std = Standardizer.load(std_path)
        # 缓存到 device
        self._std_x_mean = torch.as_tensor(self.std.x_mean, dtype=torch.float32, device=self.device)
        self._std_x_std  = torch.as_tensor(self.std.x_std + 1e-8, dtype=torch.float32, device=self.device)
        self._std_u_mean = torch.as_tensor(self.std.u_mean, dtype=torch.float32, device=self.device)
        self._std_u_std  = torch.as_tensor(self.std.u_std + 1e-8, dtype=torch.float32, device=self.device)
        # 仅对这些维度做 z-score：pos(0:3), lin vel(7:10), ang vel(10:13)；四元数只单位化
        self._idx_scale = torch.as_tensor(np.r_[np.arange(0,3), np.arange(7,13)], dtype=torch.long, device=self.device)
        # 便捷切片
        self._slice_pos = slice(0, 3)
        self._slice_lin = slice(7, 10)
        self._slice_ang = slice(10, 13)
        self.uses_standardizer = True
        try:
            self.get_logger().info(f"Loaded Standardizer for MPC: {std_path}")
        except Exception:
            pass

    def _std_state(self, x_phys):
        """物理态 x → 标准化域；四元数单位化，pos/vel/ang 做 z-score。"""
        import torch
        x_std = x_phys.clone()
        q = torch.nn.functional.normalize(x_std[:, 3:7], dim=1)
        x_std[:, 3:7] = q
        xs = x_std.index_select(1, self._idx_scale)
        ms = self._std_x_mean.index_select(0, self._idx_scale)
        ss = self._std_x_std.index_select(0, self._idx_scale)
        x_std.index_copy_(1, self._idx_scale, (xs - ms) / ss)
        return x_std

    def _destd_extract_phys(self, x_std):
        """从标准化域状态提取物理单位的 pos/lin vel/ang 和归一化四元数。"""
        pos = x_std[:, self._slice_pos] * self._std_x_std[self._slice_pos] + self._std_x_mean[self._slice_pos]
        vel = x_std[:, self._slice_lin] * self._std_x_std[self._slice_lin] + self._std_x_mean[self._slice_lin]
        ang = x_std[:, self._slice_ang] * self._std_x_std[self._slice_ang] + self._std_x_mean[self._slice_ang]
        quat = x_std[:, 3:7]  # 已归一化
        return pos, vel, ang, quat

    def _std_action(self, u_phys):
        """物理动作 → 标准化域（z-score）。"""
        return (u_phys - self._std_u_mean) / self._std_u_std


    # -----------------------------
    # ROS callbacks / helpers
    # -----------------------------
    def _on_state(self, msg: EntityState):
        self.state_msg = msg

        pos = msg.pose.position
        # twist = msg.twist
        # ori = msg.pose.orientation

        self.position = np.array([pos.x, pos.y, pos.z])
        # self.velocity = np.array([
        #     twist.linear.x,
        #     twist.linear.y,
        #     twist.linear.z,
        #     twist.angular.x,
        #     twist.angular.y,
        #     twist.angular.z
        # ])
        # self.orientation = [ori.w, ori.x, ori.y, ori.z]
        self.initialized = True

    def _reset_to_first_waypoint(self):
        # publish /ucat/reset (optional signal to other nodes)
        # self.pub_reset.publish(Bool(data=True))

        # call /reset_to_pose service with first waypoint (flip y,z signs like MPC_5)
        if not self.reset_cli.service_is_ready():
            self.reset_cli.wait_for_service(timeout_sec=2.0)

        req = ResetToPose.Request()
        # x, y, z = waypoints[0]
        x, y, z = self.waypoints[0]
        req.x, req.y, req.z = float(x), float(-y), float(-z)
        req.roll = 0.0; req.pitch = 0.0; req.yaw = 0.0

        future = self.reset_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        self.get_logger().info(f'Reset ROV to: x={x:.2f}, y={y:.2f}, z={z:.2f}')

    # -----------------------------
    # Trajectory (CubicSpline through 5 waypoints)
    # -----------------------------
    # def _build_smooth_trajectory(self, waypoints, num_points=300):
    #     t = np.linspace(0, 1, len(waypoints))
    #     csx = CubicSpline(t, waypoints[:, 0])
    #     csy = CubicSpline(t, waypoints[:, 1])
    #     csz = CubicSpline(t, waypoints[:, 2])
    #
    #     tt = np.linspace(0, 1, num_points)
    #     traj = np.stack([csx(tt), csy(tt), csz(tt)], axis=-1).astype(np.float32)
    #     return traj  # (num_points, 3)

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

    def _nearest_traj_index(self, p):
        d = np.linalg.norm(self.trajectory - p[None, :], axis=1)
        return int(np.argmin(d))

    # -----------------------------
    # Model utilities
    # -----------------------------
    def _load_world_model(self, ckpt_path: str) -> ROVGRUModel:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'World model checkpoint not found: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        cfg_dict = ckpt.get("cfg", {})
        cfg = WMConfig(**cfg_dict) if isinstance(cfg_dict, dict) else WMConfig()
        model = ROVGRUModel(cfg)
        model.load_state_dict(ckpt["model_state"], strict=False)
        return model

    # def _msg_to_model_state(self, msg: EntityState) -> np.ndarray:
    #     # undo sign flips for position; recover internal Euler (roll, -pitch, -yaw) -> (roll, pitch, yaw)
    #     px = msg.pose.position.x
    #     py = -msg.pose.position.y
    #     pz = -msg.pose.position.z
    #     q_ros = np.array([msg.pose.orientation.x,
    #                       msg.pose.orientation.y,
    #                       msg.pose.orientation.z,
    #                       msg.pose.orientation.w], dtype=np.float32)
    #     roll_m, pitch_m, yaw_m = quat_to_euler(q_ros)  # these correspond to (roll, -pitch, -yaw) used in AUVMotion
    #     roll_i, pitch_i, yaw_i = roll_m, -pitch_m, -yaw_m
    #     q_model = euler_to_quat(roll_i, pitch_i, yaw_i)  # [w,x,y,z]
    #     q_model = quat_normalize_np(q_model)
    #
    #     v = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=np.float32)
    #     w = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z], dtype=np.float32)
    #
    #     x = np.concatenate([np.array([px, py, pz], dtype=np.float32),
    #                         q_model.astype(np.float32),
    #                         v, w], axis=0)
    #     assert x.shape[0] == 13
    #     return x

    def _msg_to_model_state(self, msg: EntityState) -> np.ndarray:
        # undo sign flips for position; recover internal Euler (roll, -pitch, -yaw) -> (roll, pitch, yaw)
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z
        q_ros = np.array([msg.pose.orientation.x,
                          msg.pose.orientation.y,
                          msg.pose.orientation.z,
                          msg.pose.orientation.w], dtype=np.float32)
        roll_m, pitch_m, yaw_m = quat_to_euler(q_ros)  # these correspond to (roll, -pitch, -yaw) used in AUVMotion
        roll_i, pitch_i, yaw_i = roll_m, pitch_m, yaw_m
        q_model = euler_to_quat(roll_i, pitch_i, yaw_i)  # [w,x,y,z]
        q_model = quat_normalize_np(q_model)

        v = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], dtype=np.float32)
        w = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z], dtype=np.float32)

        x = np.concatenate([np.array([px, py, pz], dtype=np.float32),
                            q_model.astype(np.float32),
                            v, w], axis=0)
        assert x.shape[0] == 13
        return x

    # -----------------------------
    # Cost & rollout
    # -----------------------------
    def _stage_cost(self, x: torch.Tensor, p_ref: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x: (B,13)  [p(3), q(4), v(3), w(3)]
        p_ref: (B,3)
        u: (B,Nu)
        returns: (B,) cost
        """
        pos = x[:, 0:3]
        vel = x[:, 7:10]
        ang = x[:, 10:13]

        # position error
        e = pos - p_ref
        c_pos = torch.sum(e @ torch.as_tensor(self.Q_pos, dtype=x.dtype, device=x.device) * e, dim=1)
        c_vel = torch.sum(vel @ torch.as_tensor(self.Q_vel, dtype=x.dtype, device=x.device) * vel, dim=1)
        c_ang = torch.sum(ang @ torch.as_tensor(self.Q_ang, dtype=x.dtype, device=x.device) * ang, dim=1)
        c_u   = self.R * torch.sum(u * u, dim=1)
        return c_pos + c_vel + c_ang + c_u

    # @torch.no_grad()
    @torch.inference_mode()
    def _rollout_cost(self, x0_np: np.ndarray, U_seq_np: np.ndarray, ref_idx: int) -> float:
        """
        Evaluate cost of one candidate action sequence.
        x0_np: (13,)
        U_seq_np: (N, Nu)  thruster forces
        ref_idx: index on ref trajectory to track for first step; later steps advance along it
        """
        device = self.device
        B = 1
        T = U_seq_np.shape[0]
        # Build tensors
        x_t = torch.as_tensor(x0_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1,13)
        # Rollout using the model.compose_next and a simple Euler-like step through learned delta
        # We call model in step-by-step fashion because worldModel.rollout expects (B,T,*) sequences.
        h = None
        cost = torch.zeros((1,), dtype=torch.float32, device=device)
        last_u = None

        for t in range(T):
            u_t = torch.as_tensor(U_seq_np[t], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)  # (1,1,Nu)
            x_in = x_t.unsqueeze(1)  # (1,1,13)

            pred = self.model(x_in, u_t, h)  # dict with keys: mu (B,1,12), logvar, h
            mu = pred["mu"][:, 0]            # (1,12)
            h = pred["h"]

            # compose next
            # x_t: (1,13) ; mu encodes delta p, delta v, delta w, delta-theta
            x_t = self.model.compose_next(x_t, mu)  # (1,13)

            # Reference advances along the path
            ref_idx_t = min(ref_idx + t, self.trajectory.shape[0] - 1)
            pref = torch.as_tensor(self.trajectory[ref_idx_t], dtype=torch.float32, device=device).unsqueeze(0)  # (1,3)

            # cost
            u_flat = u_t[:, 0, :]  # (1,Nu)
            c = self._stage_cost(x_t, pref, u_flat)

            # smoothness
            if last_u is not None:
                c = c + self.Rd * torch.sum((u_flat - last_u) ** 2, dim=1)
            last_u = u_flat

            cost = cost + c

        return float(cost.item())

    # @torch.inference_mode()
    # def _rollout_cost_batch(self, x0_np: np.ndarray, U_seq_np: np.ndarray, ref_idx: int) -> np.ndarray:
    #     """
    #     批量评估多个候选序列的代价。
    #     x0_np:  (13,)                       初始状态（同原来）
    #     U_seq_np: (P, T, Nu)               P个候选，每个长度T、维度Nu的动作序列
    #     ref_idx: int                       参考轨迹起点索引（对所有候选共用）
    #     return: (P,) 的 numpy 数组，表示每个候选的总代价
    #     """
    #     device = self.device
    #     U = torch.as_tensor(U_seq_np, dtype=torch.float32, device=device)  # (P, T, Nu)
    #     P, T, Nu = U.shape
    #
    #     # 初始状态复制到 batch
    #     x_t = torch.as_tensor(x0_np, dtype=torch.float32, device=device).unsqueeze(0).expand(P, -1)  # (P, 13)
    #     h = None  # 让模型根据 batch 维自己初始化隐状态（若模型需要，也可以自己构造 zeros）
    #
    #     # 预取参考轨迹(随时间前进)，shape: (T, 3) → (P, 3) 每步广播
    #     ref_slice = self.trajectory[ref_idx:ref_idx + T]
    #     if ref_slice.shape[0] < T:
    #         # 防止越界：用最后一个点填充
    #         last = self.trajectory[-1]
    #         pad = np.repeat(last[None, :], T - ref_slice.shape[0], axis=0)
    #         ref_slice = np.concatenate([ref_slice, pad], axis=0)
    #     pref_T = torch.as_tensor(ref_slice, dtype=torch.float32, device=device)  # (T, 3)
    #
    #     cost = torch.zeros((P,), dtype=torch.float32, device=device)
    #     last_u = None
    #
    #     for t in range(T):
    #         u_t = U[:, t, :].unsqueeze(1)  # (P, 1, Nu)
    #         x_in = x_t.unsqueeze(1)  # (P, 1, 13)
    #
    #         # 世界模型一步前向（批量）
    #         pred = self.model(x_in, u_t, h)  # 期望返回 dict: {"mu": (P,1,12), "logvar":..., "h": ...}
    #         mu = pred["mu"][:, 0]  # (P, 12)
    #         h = pred["h"]
    #
    #         # 组合下一个状态（保持与你的 worldModel.compose_next 一致）
    #         x_t = self.model.compose_next(x_t, mu)  # (P, 13)
    #
    #         # 代价
    #         p_ref = pref_T[t].unsqueeze(0).expand(P, -1)  # (P, 3)
    #         u_flat = u_t[:, 0, :]  # (P, Nu)
    #         c = self._stage_cost(x_t, p_ref, u_flat)
    #
    #         # 平滑项
    #         if last_u is not None:
    #             c = c + self.Rd * torch.sum((u_flat - last_u) ** 2, dim=1)
    #         last_u = u_flat
    #
    #         cost = cost + c
    #
    #     return cost.detach().cpu().numpy()  # (P,)

    @torch.inference_mode()
    def _rollout_cost_batch(self, x0_np: np.ndarray, U_seq_np: np.ndarray, ref_idx: int) -> np.ndarray:
        """
        批量评估多个候选序列的代价（最小改动版）：
        - 模型前向与 compose_next 在“标准化域”运行
        - 代价计算在“物理单位”进行（将 pos/vel/ang 反标准化）
        - 动作平滑/能量仍用物理单位的 u（与推力限幅一致）
        """
        device = self.device

        # 动作保持物理单位，供限幅与代价使用；仅在送入模型前转标准化
        U_phys = torch.as_tensor(U_seq_np, dtype=torch.float32, device=device)  # (P, T, Nu)
        P, T, Nu = U_phys.shape

        # 初始状态：物理 → 标准化（送入模型）
        x0_phys = torch.as_tensor(x0_np, dtype=torch.float32, device=device).unsqueeze(0).expand(P, -1)
        x_t_std = self._std_state(x0_phys)
        h = None

        # 参考轨迹保持物理单位（代价使用）
        ref_slice = self.trajectory[ref_idx:ref_idx + T]
        if ref_slice.shape[0] < T:
            last = self.trajectory[-1]
            pad = np.repeat(last[None, :], T - ref_slice.shape[0], axis=0)
            ref_slice = np.concatenate([ref_slice, pad], axis=0)
        pref_T = torch.as_tensor(ref_slice, dtype=torch.float32, device=device)  # (T, 3)

        cost = torch.zeros((P,), dtype=torch.float32, device=device)
        last_u_phys = None
        # 如果你已有姿态平滑/倾斜项，可在循环里使用 q_prev；这里先保留接口
        q_prev = None

        for t in range(T):
            # 物理动作 → 标准化动作（供模型）
            u_phys_t = U_phys[:, t, :]                 # (P,Nu)
            u_std_t  = self._std_action(u_phys_t).unsqueeze(1)  # (P,1,Nu)

            # 标准化域中滚动模型一步
            x_in_std = x_t_std.unsqueeze(1)            # (P,1,13)
            pred = self.model(x_in_std, u_std_t, h)
            mu_std = pred["mu"][:, 0]
            h = pred["h"]
            x_t_std = self.model.compose_next(x_t_std, mu_std)

            # 反标准化得到物理 pos/vel/ang，用于代价
            pos_phys, vel_phys, ang_phys, quat_t = self._destd_extract_phys(x_t_std)
            p_ref = pref_T[t].unsqueeze(0).expand(P, -1)  # 物理单位参考点

            # 复用原 _stage_cost：构造 x_for_cost，仅替换用到的切片
            x_for_cost = x_t_std.clone()
            x_for_cost[:, 0:3]   = pos_phys
            x_for_cost[:, 7:10]  = vel_phys
            x_for_cost[:, 10:13] = ang_phys

            c = self._stage_cost(x_for_cost, p_ref, u_phys_t)

            # 动作平滑（维持你原逻辑；仍用物理单位）
            if last_u_phys is not None:
                c = c + self.Rd * torch.sum((u_phys_t - last_u_phys) ** 2, dim=1)
            last_u_phys = u_phys_t

            # （可选）若你已有姿态倾斜/姿态平滑项，可在此叠加：
            # Rzz = 1 - 2*(quat_t[:,1]**2 + quat_t[:,2]**2)
            # c = c + self.W_tilt * (1.0 - Rzz)**2
            # if q_prev is not None:
            #     dot_q = torch.abs(torch.sum(quat_t * q_prev, dim=1))
            #     c = c + self.W_att_smooth * (1.0 - dot_q)**2
            # q_prev = quat_t

            cost = cost + c

        # （可选）若你已有终端位置代价，这里用反标的终端 pos_phys 叠加：
        # p_ref_T = pref_T[-1].unsqueeze(0).expand(P, -1)
        # pos_T_phys, _, _, _ = self._destd_extract_phys(x_t_std)
        # e_T = pos_T_phys - p_ref_T
        # Qp_term = torch.as_tensor(self.Q_pos * self.Q_pos_term_scale, dtype=torch.float32, device=device)
        # cost = cost + torch.sum(e_T @ Qp_term * e_T,_*


    # -----------------------------
    # CEM planner
    # -----------------------------
    def _plan(self, x0: np.ndarray) -> np.ndarray:
        Nu = self.num_thrusters
        T  = self.N

        mu  = self.act_mu.copy()    # (T,Nu)
        std = self.act_std.copy()   # (T,Nu)

        ref_start = self._nearest_traj_index(x0[0:3])

        for it in range(self.cem_iters):  # fixed small number of iterations for stability
            # sample
            samples = np.random.randn(self.N, self.num_thrusters, self.cem_pop).transpose(2,0,1)  # (P,T,Nu)
            U = mu[None, :, :] + std[None, :, :] * samples                                  # (P,T,Nu)
            # clip to limits
            U = np.clip(U, self.u_low[None, None, :], self.u_high[None, None, :])



            # tmp_now = time.time()
            # evaluate
            # costs = np.empty((self.cem_pop,), dtype=np.float32)
            # for i in range(self.cem_pop):
            #     costs[i] = self._rollout_cost(x0, U[i], ref_start)

            costs = self._rollout_cost_batch(x0, U, ref_start)

            # cost_time = time.time() - tmp_now
            # print(cost_time, "\n")

            elite_idx = np.argsort(costs)[:self.cem_elite]
            elites = U[elite_idx]  # (E,T,Nu)

            # update
            mu  = 0.8 * mu  + 0.2 * elites.mean(axis=0)
            std = 0.8 * std + 0.2 * elites.std(axis=0)

        # save distribution for next time (warm start)
        self.act_mu, self.act_std = mu, np.maximum(std, 1e-3)

        # return mu[0]  # first action of planned sequence (Nu,)

        return mu[0], ref_start

    # -----------------------------
    # Main step
    # -----------------------------
    def step_once(self):
        if self.state_msg is None:
            return

        # build model state x
        x = self._msg_to_model_state(self.state_msg)

        # plan in thruster space
        # u_thr = self._plan(x)  # (Nu,)
        u_thr, ref_idx = self._plan(x)
        u_thr = np.clip(u_thr, self.u_low, self.u_high)


        msg = Float32MultiArray(data=u_thr.tolist())
        self.publisher_cmd.publish(msg)


        # # convert to body-frame wrench via YAML config
        wrench_body = self.tw_calc.compute_wrench(u_thr)  # (6,) [Fx,Fy,Fz,Tx,Ty,Tz]

        # # publish WrenchStamped (body frame)
        # msg = WrenchStamped()
        # msg.header = Header()
        # msg.header.stamp = self.get_clock().now().to_msg()
        # msg.wrench.force.x  = float(wrench_body[0])
        # msg.wrench.force.y  = float(wrench_body[1])
        # msg.wrench.force.z  = float(wrench_body[2])
        # msg.wrench.torque.x = float(wrench_body[3])
        # msg.wrench.torque.y = float(wrench_body[4])
        # msg.wrench.torque.z = float(wrench_body[5])
        # self.pub_wrench.publish(msg)

        now = time.time() - self.start_time

        print(self.step, "\n")
        print(u_thr, "\n")
        print(now, "\n")

        pos = self.state_msg.pose.position
        ori = self.state_msg.pose.orientation
        tw = self.state_msg.twist
        lin = [tw.linear.x, tw.linear.y, tw.linear.z]
        ang = [tw.angular.x, tw.angular.y, tw.angular.z]

        self._data["time"].append(now)
        self._data["position"].append([pos.x, pos.y, pos.z])  # 直接使用话题中的实际坐标
        self._data["orientation_wxyz"].append([ori.w, ori.x, ori.y, ori.z])  # 与 MPC_5.py 保持的 wxyz 顺序
        self._data["linear_velocity"].append(lin)
        self._data["angular_velocity"].append(ang)
        self._data["thrusters"].append(u_thr.tolist())
        self._data["wrench_body"].append(wrench_body.tolist())
        self._data["ref_index"].append(int(ref_idx))
        self._data["ref_point"].append(self.trajectory[min(ref_idx, self.trajectory.shape[0] - 1)].tolist())

        self.step += 1
        if self.step >= self.max_steps:
            self.done = True

    def wait_time_optimizer(self, start_time, end_time):
        # 计算时间误差
        dt_error = (self.dt / self.fast_forward) - (end_time - start_time)

        # 如果时间误差小于 0.1，更新 time_optimize_value
        if dt_error < 0.1:
            self.time_optimize_value += self.dt_optimize_gain * dt_error
            self.time_optimize_value = np.clip(self.time_optimize_value, -0.1, 0.1)  # 限制时间优化值的更新范围

    def control_loop(self):
        if not self.initialized:
            return
        if self.done:
            return

        self.step += 1

        start_time = time.perf_counter()

        print(self.step)

        # Exit if reached goal
        if np.linalg.norm(self.position - self.goal) < 0.5:
            self.get_logger().info('Reached goal. Exiting...')
            self.done = True
            self.destroy_node()
            # rclpy.shutdown()
            return

        # Exit if exceeded max steps
        if self.step > self.max_steps:
            self.get_logger().warn('Exceeded max steps. Exiting...')
            self.done = True
            self.destroy_node()
            # rclpy.shutdown()
            return
        x = self._msg_to_model_state(self.state_msg)

        # plan in thruster space
        # u_thr = self._plan(x)  # (Nu,)
        u_thr, ref_idx = self._plan(x)
        u_thr = np.clip(u_thr, self.u_low, self.u_high)

        msg = Float32MultiArray(data=u_thr.tolist())
        self.publisher_cmd.publish(msg)

        # # convert to body-frame wrench via YAML config
        wrench_body = self.tw_calc.compute_wrench(u_thr)  # (6,) [Fx,Fy,Fz,Tx,Ty,Tz]

        # # publish WrenchStamped (body frame)
        # msg = WrenchStamped()
        # msg.header = Header()
        # msg.header.stamp = self.get_clock().now().to_msg()
        # msg.wrench.force.x  = float(wrench_body[0])
        # msg.wrench.force.y  = float(wrench_body[1])
        # msg.wrench.force.z  = float(wrench_body[2])
        # msg.wrench.torque.x = float(wrench_body[3])
        # msg.wrench.torque.y = float(wrench_body[4])
        # msg.wrench.torque.z = float(wrench_body[5])
        # self.pub_wrench.publish(msg)

        now = time.time() - self.start_time

        # print(self.step, "\n")
        # print(u_thr, "\n")
        print(now, "\n")

        pos = self.state_msg.pose.position
        ori = self.state_msg.pose.orientation
        tw = self.state_msg.twist
        lin = [tw.linear.x, tw.linear.y, tw.linear.z]
        ang = [tw.angular.x, tw.angular.y, tw.angular.z]

        self._data["time"].append(now)
        self._data["position"].append([pos.x, pos.y, pos.z])  # 直接使用话题中的实际坐标
        self._data["orientation_wxyz"].append([ori.w, ori.x, ori.y, ori.z])  # 与 MPC_5.py 保持的 wxyz 顺序
        self._data["linear_velocity"].append(lin)
        self._data["angular_velocity"].append(ang)
        self._data["thrusters"].append(u_thr.tolist())
        self._data["wrench_body"].append(wrench_body.tolist())
        self._data["ref_index"].append(int(ref_idx))
        self._data["ref_point"].append(self.trajectory[min(ref_idx, self.trajectory.shape[0] - 1)].tolist())

        self.initialized = False
        self.step += 1

        # 获取结束时间并优化等待时间
        end_time = time.perf_counter()
        self.wait_time_optimizer(start_time, end_time)


    # def destroy_node(self):
    #     try:
    #         for k, v in self._data.items():
    #             self._h5file.create_dataset(k, data=np.array(v))
    #         self._h5file.flush()
    #         self._h5file.close()
    #     except Exception as e:
    #         try:
    #             self.get_logger().error(f'Failed to write HDF5: {e}')
    #         except Exception:
    #             pass
    #     super().destroy_node()

    def destroy_node(self):

        if not getattr(self, "h5_open", False):
            super().destroy_node()
            return

        for key, value in self._data.items():
            self._h5file.create_dataset(key, data=np.array(value))

        self._h5file.flush()
        self._h5file.close()
        self.h5_open = False
        super().destroy_node()

def main():
    rclpy.init()

    # ---- Define five 3D waypoints (match MPC_5 style: x, y, z in meters) ----
    # You can modify these points or pass via parameters if you integrate as a ROS node.
    # waypoints = [
    #     [2.0, -3.0, -5.0],
    #     [6.0, 5.0, -10.0],
    #     [10.0, -5.0, -2.0],
    #     [14.0, 2.0, -9.0],
    #     [18.0, 0.0, -4.0]
    # ]

    waypoints = [
        [5.0,   0.0,  -10.0],    # 起点靠近x最小边界
        [12.0,  10.0, -20.0],    # 向右上拐弯并下降
        [20.0, -10.0, -5.0],     # 向左下折返并上升
        [28.0,   5.0, -18.0],    # 向右上方再次转折下降
        [35.0,   0.0, -8.0]     # 终点靠近x最大边界
    ]

    # ---- Paths: pretrained world model checkpoint & dynamics YAML ----
    # Put your actual checkpoint path here (produced by underwaterWM/run.py).
    # The checkpoint must include the model_state and cfg in a torch.save dict.
    wm_ckpt_path = os.environ.get('WM_CKPT', './checkpoints/20250814_2031/model_epoch200.pt')
    yaml_dynamics = os.environ.get('YAML_DYN', '/home/xukai/ros2_ws/src/eeuv_sim/data/dynamics/BlueDynamics.yaml')

    node = WMPCController(
        waypoints=waypoints,
        wm_ckpt_path=wm_ckpt_path,
        yaml_dynamics=yaml_dynamics,
        dt=0.1,
        horizon=30,
        max_steps=2000,
        cem_iters=2,
        cem_pop=1024,
        cem_elite=64,
        log_path='./logs/wmpc_0815_1332.h5',
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=0.05)
            # node.step_once()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()