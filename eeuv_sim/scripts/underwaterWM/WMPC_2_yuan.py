
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMPC_1.py
基于“预训练世界模型（underwaterWM/worldModel.py）”的 MPC（CEM/随机优化）路径跟随控制器。
要点：
- 直接向 /ucat/thruster_cmd 发布 8 维推力（Float32MultiArray），不发布六维力/力矩。
- 参考路径：采用 5 个 waypoint 的三次样条光滑轨迹（与 scripts/comparison_methods/MPC_5.py 一致思路）。
- 严格加载 checkpoint 与 standardizer（与 underwaterWM/test_single.py 一致流程）。
- 话题：订阅 /ucat/state (gazebo_msgs/EntityState)，服务 /reset_to_pose (eeuv_sim/ResetToPose)。

使用方法（示例）
---------------
终端：
  ros2 run <your_pkg> WMPC_1.py --ros-args \
      -p ckpt_path:=/path/to/ckpt/model_epoch150.pt \
      -p std_path:=/path/to/ckpt/standardizer.npz \
      -p dt:=0.1 -p horizon:=15

或直接执行：
  python3 WMPC_1.py --ckpt /path/to/ckpt.pt --std /path/to/standardizer.npz

注意：本文件默认假设位于 `scripts/` 目录下；会自动将 `scripts/underwaterWM/` 加入 sys.path。
"""
import os
import sys
import time
import math
import argparse
from typing import List, Tuple

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from gazebo_msgs.msg import EntityState
from std_msgs.msg import Float32MultiArray, Bool
from eeuv_sim.srv import ResetToPose
from scipy.interpolate import CubicSpline
import h5py


# # ---------- 将 underwaterWM 加入路径（假定本文件位于 scripts/ 目录下） ----------
# _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# _UNDERWATERWM_DIR = os.path.join(_THIS_DIR, "underwaterWM")
# if _UNDERWATERWM_DIR not in sys.path:
#     sys.path.insert(0, _UNDERWATERWM_DIR)

# 导入世界模型 & 标准化工具
from worldModel import WMConfig, ROVGRUModel, rollout  # type: ignore
from utils import Standardizer, quat_normalize_np  # type: ignore


def invert_position_standardization(std: Standardizer, p_std: np.ndarray) -> np.ndarray:
    """把标准化空间的 position (...,3) 还原到物理单位"""
    p_mean = std.x_mean[0:3]
    p_stdv = std.x_std[0:3] + 1e-8
    return p_std * p_stdv + p_mean


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4,)
    n = np.linalg.norm(q) + 1e-8
    return (q / n).astype(np.float32)


class WMPCController(Node):
    """
    基于世界模型的 MPC 控制器：
    - 使用 CEM（交叉熵）在有限时域（horizon）上优化 8 维推力序列
    - 使用世界模型进行 rollout，计算与参考轨迹的偏差代价
    - 发布第一步动作到 /ucat/thruster_cmd
    """
    def __init__(
        self,
        cli_args: argparse.Namespace,
        waypoints: List[List[float]],
        dt: float = 0.1,
        horizon: int = 15,
        max_steps: int = 2000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__('wmpc_controller')

        # ---------------- ROS 话题与服务 ----------------
        # 直接发布 8 维推力（不要发布六维力与力矩）
        self.thrust_pub = self.create_publisher(Float32MultiArray, '/ucat/thruster_cmd', 1)
        # self.reset_pub = self.create_publisher(Bool, '/ucat/reset', 1)
        self.state_sub = self.create_subscription(EntityState, '/ucat/state', self.state_callback, 1)
        self.reset_client = self.create_client(ResetToPose, '/reset_to_pose')
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/reset_to_pose service not available at startup.')

        # ---------------- 参数与内部状态 ----------------
        # 运行/离线参数（优先 ROS 参数）
        self.declare_parameter('ckpt_path', '')
        self.declare_parameter('std_path', '')  ## standardizer
        self.declare_parameter('dt', dt)  ## 控制时长
        self.declare_parameter('horizon', horizon)  ## MPC 预测时域长度
        self.declare_parameter('max_steps', max_steps)

        # CEM 参数
        self.declare_parameter('pop_size', 512)  ## 128  每轮采样的轨迹规模
        self.declare_parameter('elite_frac', 0.25)  ## 选取前多少百分比作为精英（用来更新高斯分布的均值/方差）
        self.declare_parameter('n_iters', 2)  ## CEM 外层迭代次数。越大越稳，但耗时增加
        self.declare_parameter('init_std', 2.0)        # 初始采样标准差（牛顿） 决定第一轮搜索的“步幅/探索度”。

        # 代价权重
        self.declare_parameter('w_pos', 5.0)           # 位置误差  ## 5.0
        self.declare_parameter('w_u', 2)            # 控制能量  ## 1e-2
        self.declare_parameter('w_du', 2)           # 控制增量  ## 1e-2

        # 动作限制（8 维推力，单位 N；可根据你的 YAML/动力学设置调整）
        self.declare_parameter('u_min', [-20.0]*8) ## 8 维推力上下限（N）
        self.declare_parameter('u_max', [ 20.0]*8)

        # self.dt = float(self.get_parameter('dt').value)
        # self.N = int(self.get_parameter('horizon').value)
        # self.max_steps = int(self.get_parameter('max_steps').value)

        self.dt = cli_args.dt
        self.N = cli_args.horizon
        self.max_steps = cli_args.max_steps

        self.pop_size = int(self.get_parameter('pop_size').value)
        self.elite_frac = float(self.get_parameter('elite_frac').value)
        self.n_iters = int(self.get_parameter('n_iters').value)
        self.init_std = float(self.get_parameter('init_std').value)
        self.w_pos = float(self.get_parameter('w_pos').value)
        self.w_u = float(self.get_parameter('w_u').value)
        self.w_du = float(self.get_parameter('w_du').value)
        self.u_min = np.array(self.get_parameter('u_min').value, dtype=np.float32).reshape(1,1,-1)
        self.u_max = np.array(self.get_parameter('u_max').value, dtype=np.float32).reshape(1,1,-1)

        # 参考轨迹
        self.waypoints = waypoints
        self.trajectory = self.generate_smooth_3d_trajectory(self.waypoints, num_points=300)
        self.ref_index = 0                 # 跟踪进度指针
        self.search_ahead = 40             # 在 [ref_index, ref_index+search_ahead] 窗口内找最近点  ## 80
        self.advance_thresh = 0.5          # 若足够靠近当前参考点则推进索引（米）

        # 世界模型/标准化器
        # ckpt_path = self.get_parameter('ckpt_path').value
        # std_path = self.get_parameter('std_path').value

        ckpt_path = cli_args.ckpt
        std_path = cli_args.std

        if (not std_path) and ckpt_path:
            std_path = os.path.join(os.path.dirname(ckpt_path), "standardizer.npz")
        self.model, self.std = self._load_world_model(ckpt_path, std_path, device=device)

        # 滚动优化 warm start
        self.prev_mean = np.zeros((self.N, self.model.cfg.u_dim), dtype=np.float32)

        # 状态
        self.state = None                  # 最新 /ucat/state
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(6, dtype=np.float64)  # [v_world(3), w_world(3)]
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz
        self.initialized = False
        self.step_count = 0
        self.done = False

        # 日志（可选）
        # results_dir = os.path.join(_THIS_DIR, "comparison_methods", "results")
        results_dir = './logs'
        os.makedirs(results_dir, exist_ok=True)
        # self.log_path = os.path.join(results_dir, "wmpc_rov_log_0818_2142.h5")
        # self.log_path = os.path.join(results_dir, "wmpc_rov_log_0818_2147.h5")
        # self.log_path = os.path.join(results_dir, "wmpc_rov_log_0818_2157.h5")
        # self.log_path = os.path.join(results_dir, "wmpc_rov_log_0819_1942.h5")
        # self.log_path = os.path.join(results_dir, "wmpc_rov_log_0819_2001.h5")
        # self.log_path = os.path.join(results_dir, "wmpc_rov_log_0819_2005.h5")
        # self.log_path = os.path.join(results_dir, "wmpc_rov_log_0819_2008.h5")
        self.log_path = os.path.join(results_dir, "wmpc_rov_log_0819_2011.h5")
        self.h5 = h5py.File(self.log_path, "w")
        self._init_h5_dsets()


        # 定时器驱动控制
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.start_time = time.perf_counter()
        self.fast_forward = 1.0
        self.time_optimize_value = 0.0
        self.dt_optimize_gain = 0.05

        self._last_pub_t = None

        self.get_logger().info("WMPC controller initialized.")

        # 初始化：把仿真实体移动到第一个 waypoint
        self.reset_ROV_to_first_waypoint()

    # ---------------- 话题回调 ----------------
    def state_callback(self, msg: EntityState):
        """订阅 /ucat/state：缓存位置、姿态、速度"""
        self.state = msg

        pos = msg.pose.position
        tw  = msg.twist
        ori = msg.pose.orientation

        self.position = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        self.velocity = np.array([
            tw.linear.x, tw.linear.y, tw.linear.z,
            tw.angular.x, tw.angular.y, tw.angular.z
        ], dtype=np.float64)

        # ROS 是 xyzw；世界模型使用 wxyz
        self.orientation = normalize_quat_wxyz([ori.w, ori.x, ori.y, ori.z])

        self.initialized = True

    # ---------------- 控制主循环 ----------------
    def control_loop(self):
        if self.done or (self.step_count >= self.max_steps):
            if not self.done:
                self.get_logger().info("Reached max steps. Stopping.")
            self.done = True
            return

        if not self.initialized or (self.state is None):
            return  # 等待初次状态

        start_time = time.time()

        # 选择参考段（窗口内最近点 + 单调推进）
        i_start = self.ref_index
        i_stop = min(i_start + self.search_ahead, len(self.trajectory) - 1)
        seg = self.trajectory[i_start: i_stop + 1]                 # [M,3]
        local_i = int(np.argmin(np.linalg.norm(seg - self.position, axis=1)))
        i_curr = i_start + local_i

        if np.linalg.norm(self.position - self.trajectory[self.ref_index]) < self.advance_thresh:
            self.ref_index = min(self.ref_index + 1, len(self.trajectory) - 1)
        self.ref_index = max(self.ref_index, i_curr)

        ref_traj = self.trajectory[self.ref_index: self.ref_index + self.N]
        if len(ref_traj) < self.N:
            ref_traj = np.pad(ref_traj, ((0, self.N - len(ref_traj)), (0, 0)), mode='edge')
        ref_traj = ref_traj.astype(np.float32)                     # (N,3)

        # 当前状态 -> 世界模型标准化空间
        x0_raw = np.concatenate([
            self.position.astype(np.float32),
            self.orientation.astype(np.float32),
            self.velocity.astype(np.float32)
        ], axis=0)  # (13,)

        x0_std = self.std.apply_x_np(x0_raw[None, :])              # (1,13)

        # CEM 求解一段动作序列 u_seq (N,8)
        u0 = self._solve_cem(x0_std, ref_traj)                     # (8,)

        # 发布 8 维推力
        self.publish_thrusters(u0)

        # 写日志
        self._log_step(u0)

        # 终点判断
        if self.ref_index >= (len(self.trajectory) - 2) and np.linalg.norm(self.position - self.trajectory[-1]) < 0.5:
            self.get_logger().info("Goal reached.")
            self.done = True

        self.step_count += 1

        # 获取结束时间并优化等待时间
        # end_time = time.time()
        # self.wait_time_optimizer(start_time, end_time)
        # # end_time_1 = time.time()
        #
        # # print(self.step_count, end_time_1 - start_time, "\n")
        #
        # try:
        #     time.sleep((self.dt / self.fast_forward / 2) + self.time_optimize_value)
        # except Exception as e:
        #     self.get_logger().error(f"Invalid sleep time: {e}")
        #     time.sleep(self.dt / self.fast_forward)  # 在出现异常时使用默认时间

    def wait_time_optimizer(self, start_time, end_time):
        # 计算时间误差
        dt_error = (self.dt / self.fast_forward) - (end_time - start_time)

        # 如果时间误差小于 0.1，更新 time_optimize_value
        if dt_error < 0.1:
            self.time_optimize_value += self.dt_optimize_gain * dt_error
            self.time_optimize_value = np.clip(self.time_optimize_value, -0.1, 0.1)  # 限制时间优化值的更新范围

    # ---------------- 发布动作 ----------------
    # def publish_thrusters(self, thrusts: np.ndarray):
    #     thrusts = thrusts.astype(np.float32).reshape(-1,)
    #     msg = Float32MultiArray()
    #     msg.data = thrusts.tolist()
    #     self.thrust_pub.publish(msg)

    def publish_thrusters(self, thrusts: np.ndarray):
        thrusts = thrusts.astype(np.float32).reshape(-1, )
        msg = Float32MultiArray()
        msg.data = thrusts.tolist()

        # --- 新增：计算真实发布间隔/频率/抖动 ---
        now = time.perf_counter()
        if self._last_pub_t is not None:
            period = now - self._last_pub_t
            hz = (1.0 / period) if period > 0 else float('inf')
            jitter = period - self.dt
            self.get_logger().info(
                f"pub period: {period * 1000:.2f} ms (target {self.dt * 1000:.1f} ms), "
                f"hz≈{hz:.2f}, jitter {jitter * 1000:+.2f} ms"
            )
        self._last_pub_t = now
        # --------------------------------------

        self.thrust_pub.publish(msg)

    # ---------------- CEM 优化 ----------------
    def _solve_cem(self, x0_std: np.ndarray, ref_traj: np.ndarray) -> np.ndarray:
        """
        x0_std: (1,13)
        ref_traj: (N,3) 物理单位
        return: (u_dim,) 第一步动作
        """
        N = self.N
        u_dim = self.model.cfg.u_dim

        # 初始化均值/方差（沿用上次的 warm start）
        mean = self.prev_mean.copy()            # (N,u_dim)
        std  = np.ones_like(mean) * self.init_std

        # 预备 torch 张量
        device = self.model.cfg.device
        x0_t = torch.from_numpy(x0_std).to(device=device, dtype=torch.float32)  # (1,13)
        ref_t = torch.from_numpy(ref_traj).to(device=device, dtype=torch.float32)  # (N,3)

        pop_size = self.pop_size
        n_elite = max(1, int(self.elite_frac * pop_size))

        best_seq = None
        best_cost = float('inf')

        for _ in range(self.n_iters):
            # 采样动作序列（高斯 + 裁剪）: (pop,N,u_dim)
            samples = np.random.randn(pop_size, N, u_dim).astype(np.float32) * std + mean
            samples = np.clip(samples, self.u_min, self.u_max)

            # -> 标准化控制输入（供世界模型）
            #   Standardizer.apply_u_np 支持任意形状；我们批量处理
            u_std = (samples - self.std.u_mean.reshape(1,1,-1)) / (self.std.u_std.reshape(1,1,-1) + 1e-8)  # (pop,N,u_dim)

            # torch rollout（批处理）
            with torch.no_grad():
                u_t = torch.from_numpy(u_std).to(device=device, dtype=torch.float32)    # (B,N,u_dim)
                # x_hat: (B,N+1,13)
                x_hat = rollout(self.model, x0_t.repeat(pop_size, 1), u_t)
                p_pred_std = x_hat[:, 1:, 0:3]  # (B,N,3) 只取每步后的位姿位置
                # 反标准化位置（向量化）
                p_pred = p_pred_std * torch.from_numpy(self.std.x_std[0:3]).to(device) + \
                         torch.from_numpy(self.std.x_mean[0:3]).to(device)               # (B,N,3)

                # 成本：位置误差 + 控制能量 + 控制增量
                pos_err = p_pred - ref_t.unsqueeze(0)            # (B,N,3)
                pos_cost = (pos_err ** 2).sum(dim=-1).sum(dim=-1)  # (B,)

                u_energy = (u_t ** 2).sum(dim=-1).sum(dim=-1)      # (B,)

                du = u_t[:, 1:, :] - u_t[:, :-1, :]
                du_cost = (du ** 2).sum(dim=-1).sum(dim=-1)

                total_cost = self.w_pos * pos_cost + self.w_u * u_energy + self.w_du * du_cost

                # 选精英
                costs_np = total_cost.detach().cpu().numpy()       # (B,)
                elite_idx = np.argsort(costs_np)[:n_elite]
                elites = samples[elite_idx]                        # (n_elite,N,u_dim)

                # 记录当前最佳
                if float(costs_np[elite_idx[0]]) < best_cost:
                    best_cost = float(costs_np[elite_idx[0]])
                    best_seq = samples[elite_idx[0]]

            # 更新分布
            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-6
            # 可选：加入温和的均值平滑/阻尼
            mean = 0.7 * mean + 0.3 * self.prev_mean

        if best_seq is None:
            # 极端情况下，退化为零控制
            best_seq = np.zeros((N, u_dim), dtype=np.float32)

        # 第一步动作
        u0 = best_seq[0].copy()

        # 滚动 warm start：把均值向前平移一格
        self.prev_mean[:-1] = best_seq[1:]
        self.prev_mean[-1] = 0.0

        # 裁剪输出
        u0 = np.clip(u0, self.u_min.reshape(-1,), self.u_max.reshape(-1,))
        return u0

    # ---------------- 轨迹生成 ----------------
    @staticmethod
    def generate_smooth_3d_trajectory(waypoints: List[List[float]], num_points: int = 300) -> np.ndarray:
        """
        使用三次样条在给定 3D waypoints 上生成平滑轨迹
        return: (num_points, 3)
        """
        waypoints = np.array(waypoints, dtype=np.float32)
        t = np.linspace(0.0, 1.0, len(waypoints))
        cs_x = CubicSpline(t, waypoints[:, 0])
        cs_y = CubicSpline(t, waypoints[:, 1])
        cs_z = CubicSpline(t, waypoints[:, 2])

        t_s = np.linspace(0.0, 1.0, num_points)
        x_s = cs_x(t_s)
        y_s = cs_y(t_s)
        z_s = cs_z(t_s)
        return np.vstack([x_s, y_s, z_s]).T.astype(np.float32)

    # ---------------- 重置仿真到第一个航点 ----------------
    def reset_ROV_to_first_waypoint(self):
        if len(self.waypoints) == 0:
            return
        x, y, z = self.waypoints[0]
        req = ResetToPose.Request()
        req.x = float(x)
        req.y = float(-y)
        req.z = float(-z)
        req.roll = 0.0
        req.pitch = 0.0
        req.yaw = 0.0
        try:
            fut = self.reset_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
            self.get_logger().info(f"Reset to first waypoint: ({x:.2f},{y:.2f},{z:.2f})")
        except Exception as e:
            self.get_logger().warn(f"reset_to_pose call failed: {e}")

        # 兼容旧节点：可选广播 /ucat/reset
        # try:
        #     msg = Bool(); msg.data = True
        #     self.reset_pub.publish(msg)
        # except Exception:
        #     pass

    # ---------------- HDF5 日志 ----------------
    def _init_h5_dsets(self):
        self.h5.create_dataset("time", (0,), maxshape=(None,), dtype=np.float32)
        self.h5.create_dataset("position", (0,3), maxshape=(None,3), dtype=np.float32)
        self.h5.create_dataset("orientation", (0,4), maxshape=(None,4), dtype=np.float32)
        self.h5.create_dataset("linear_velocity", (0,3), maxshape=(None,3), dtype=np.float32)
        self.h5.create_dataset("angular_velocity", (0,3), maxshape=(None,3), dtype=np.float32)
        self.h5.create_dataset("thrusts", (0,8), maxshape=(None,8), dtype=np.float32)

    def _append_row(self, name: str, row: np.ndarray):
        ds = self.h5[name]
        n = ds.shape[0]
        ds.resize((n+1,) + ds.shape[1:])
        ds[n] = row

    def _log_step(self, u: np.ndarray):
        t = self.step_count * self.dt
        lv = self.velocity[:3].astype(np.float32)
        av = self.velocity[3:].astype(np.float32)
        self._append_row("time", np.array([t], dtype=np.float32))
        self._append_row("position", self.position.astype(np.float32))
        self._append_row("orientation", self.orientation.astype(np.float32))
        self._append_row("linear_velocity", lv)
        self._append_row("angular_velocity", av)
        self._append_row("thrusts", u.astype(np.float32))
        self.h5.flush()

    # ---------------- 世界模型加载 ----------------
    def _load_world_model(self, ckpt_path: str, std_path: str, device: str):
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError("必须提供有效的 ckpt_path（.pt）用于加载世界模型。")
        if not std_path or not os.path.isfile(std_path):
            raise FileNotFoundError("必须提供有效的 std_path（standardizer.npz）。")

        ckpt = torch.load(ckpt_path, map_location='cpu')
        cfg_dict = ckpt.get("cfg", None)
        if cfg_dict is None:
            raise RuntimeError("checkpoint 中缺少 cfg 字段")
        cfg = WMConfig(**cfg_dict)
        cfg.device = device

        std = Standardizer.load(std_path)

        model = ROVGRUModel(cfg).to(cfg.device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        self.get_logger().info(f"Loaded world model from: {ckpt_path}")
        self.get_logger().info(f"Loaded standardizer from: {std_path}")
        return model, std

    # def destroy_node(self):
    #
    #     if not getattr(self, "h5_open", False):
    #         super().destroy_node()
    #         return
    #
    #     for key, value in self._data.items():
    #         self._h5file.create_dataset(key, data=np.array(value))
    #
    #     self._h5file.flush()
    #     self._h5file.close()
    #     self.h5_open = False
    #     super().destroy_node()

# ---------------- main ----------------
def default_waypoints() -> List[List[float]]:
    # 参考 MPC_5.py 中注释示例的 5 个 waypoint（单位：米）
    return [
        [5.0,   0.0, -10.0],
        [12.0, 10.0, -20.0],
        [20.0,-10.0,  -5.0],
        [28.0,  5.0, -18.0],
        [35.0,  0.0,  -8.0],
    ]
#     return [
#     [34.5,  -0.0, -15.0],     # 0°
#     [30.25, -10.25, -15.0],   # 45°
#     [20.0,  -14.5, -15.0],    # 90°
#     [9.75,  -10.25, -15.0],   # 135°
#     [5.5,   0.0, -15.0],     # 180°
#     [9.75, 10.25, -15.0],   # 225°
#     [20.0, 14.5, -15.0],    # 270°
#     [30.25, 10.25, -15.0]   # 315°
#   ]



def parse_cli_args():
    parser = argparse.ArgumentParser(description="WM-based MPC path following controller")
    # parser.add_argument("--ckpt", type=str, default='./checkpoints/20250814_2031/model_epoch200.pt', help="世界模型 checkpoint 路径（.pt）")
    parser.add_argument("--ckpt", type=str, default='./checkpoints/20250819_1934/model_epoch200.pt', help="世界模型 checkpoint 路径（.pt）")
    # parser.add_argument("--std", type=str, default='./checkpoints/20250814_2031/standardizer.npz', help="standardizer.npz 路径（默认与 ckpt 同目录）")
    parser.add_argument("--std", type=str, default='./checkpoints/20250819_1934/standardizer.npz', help="standardizer.npz 路径（默认与 ckpt 同目录）")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=2000)
    return parser.parse_args()


def main():
    # 允许 CLI 覆盖（如直接 python3 WMPC_1.py 运行）
    cli = parse_cli_args()

    # cli.ckpt ="./checkpoints/20250814_2031/model_epoch200.pt"

    rclpy.init()

    node = WMPCController(
        cli_args=cli,
        waypoints=default_waypoints(),
        dt=cli.dt,
        horizon=cli.horizon,
        max_steps=cli.max_steps
    )

    # # 若通过 CLI 提供路径，写回 ROS 参数（优先以 CLI 为准）
    # if cli.ckpt:
    #     node.set_parameters([rclpy.parameter.Parameter('ckpt_path', rclpy.Parameter.Type.STRING, cli.ckpt)])
    # if cli.std:
    #     node.set_parameters([rclpy.parameter.Parameter('std_path', rclpy.Parameter.Type.STRING, cli.std)])

    # 简单执行器循环
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok() and not node.done:
            # now = time.time()
            executor.spin_once(timeout_sec=0.1)
            # tmp = time.time() - now
            # print(node.step_count, tmp, '\n')
    finally:
        try:
            node.h5.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()