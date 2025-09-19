#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMPC_MoE.py
基于“MoE 世界模型（worldmodel_MoE_3reg_1.py）”的 MPC（CEM/随机优化）路径跟随控制器。

与 WMPC_2.py 的差异（保持“最小改动”原则）：
- 将 worldModel 与 utils.Standardizer 替换为 MoE 版本：
  * 模型：WMConfigMoE3, MoEWorldModel3, rollout （来自 worldmodel_MoE_3reg_1.py）
  * 标准化：NPZStandardizer（本文件内定义，兼容 run_MoE_3reg_1.py 保存的 standardizer.npz）
- 其余话题名、参考轨迹生成、CEM 实现、日志保存格式与位置保持不变。

使用方法（示例）
---------------
终端：
  ros2 run <your_pkg> WMPC_MoE.py --ros-args \
      -p ckpt_path:=/path/to/checkpoints_moe3_aligned/moe3_epoch200.pt \
      -p std_path:=/path/to/checkpoints_moe3_aligned/standardizer.npz \
      -p dt:=0.1 -p horizon:=15

或直接执行：
  python3 WMPC_MoE.py --ckpt ./checkpoints_moe3_aligned/moe3_epoch200.pt \
                      --std  ./checkpoints_moe3_aligned/standardizer.npz \
                      --dt 0.1 --horizon 15
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

# ------- 导入 MoE 世界模型 -------
from worldmodel_MoE_3reg_1 import WMConfigMoE3, MoEWorldModel3, rollout  # type: ignore
from utils import quat_normalize_np  # type: ignore


# ---------------- MoE Standardizer（与 MoE 训练/测试脚本对齐） ----------------
class NPZStandardizer:
    """
    - 与 run_MoE_3reg_1.py 保存的 standardizer.npz 对齐；
    - 同时兼容键名：x_mean/x_std 或 mean/std；
    - apply_x_np：保持四元数不缩放，仅标准化 [pos(0:3), v(7:10), w(10:13)]。
    """
    def __init__(self, x_mean: np.ndarray, x_std: np.ndarray,
                 u_mean: np.ndarray, u_std: np.ndarray):
        self.x_mean = x_mean.astype(np.float32)
        self.x_std  = (x_std + 1e-8).astype(np.float32)
        self.u_mean = u_mean.astype(np.float32)
        self.u_std  = (u_std + 1e-8).astype(np.float32)

    @staticmethod
    def load(path: str) -> "NPZStandardizer":
        z = np.load(path)
        x_mean = z['x_mean'] if 'x_mean' in z else z['mean']
        x_std  = z['x_std']  if 'x_std'  in z else z['std']
        u_mean = z['u_mean']
        u_std  = z['u_std']
        return NPZStandardizer(x_mean, x_std, u_mean, u_std)

    # 与原 WMPC_2.py 所用 Standardizer.apply_x_np 接口保持一致
    def apply_x_np(self, x: np.ndarray) -> np.ndarray:
        """
        x: (...,13)
        - 先单位化四元数（wxyz 位于 [3:7]）
        - 仅对 [0:3] 与 [7:13] 做标准化
        """
        x = np.asarray(x, dtype=np.float32).copy()
        # 规范化四元数
        q = x[..., 3:7]
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
        x[..., 3:7] = q
        # 标准化位置与速度
        idx = np.r_[np.arange(0, 3), np.arange(7, 13)]
        x[..., idx] = (x[..., idx] - self.x_mean[idx]) / (self.x_std[idx] + 1e-8)
        return x.astype(np.float32)


def invert_position_standardization(std: NPZStandardizer, p_std: np.ndarray) -> np.ndarray:
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
    基于世界模型的 MPC 控制器（MoE 版本）：
    - 使用 CEM 在有限时域（horizon）上优化 8 维推力序列
    - 使用 MoE 世界模型进行 rollout，计算与参考轨迹的偏差代价
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

        # ---------------- ROS 话题与服务（保持不变） ----------------
        self.thrust_pub = self.create_publisher(Float32MultiArray, '/ucat/thruster_cmd', 1)
        self.state_sub = self.create_subscription(EntityState, '/ucat/state', self.state_callback, 1)
        self.reset_client = self.create_client(ResetToPose, '/reset_to_pose')
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('/reset_to_pose service not available at startup.')

        # ---------------- 参数与内部状态（保持不变） ----------------
        self.declare_parameter('ckpt_path', '')
        self.declare_parameter('std_path', '')  ## standardizer
        self.declare_parameter('dt', dt)
        self.declare_parameter('horizon', horizon)
        self.declare_parameter('max_steps', max_steps)

        # CEM 参数
        self.declare_parameter('pop_size', 1024)  ## 1024 128
        self.declare_parameter('elite_frac', 0.1)  ## 0.25
        self.declare_parameter('n_iters', 2)  ## 3
        self.declare_parameter('init_std', 3.0)  ## 2.0

        # 代价权重
        self.declare_parameter('w_pos', 50.0)  ## 10.0  30.0
        self.declare_parameter('w_u', 0.1)  ## 1e-1  0.05
        self.declare_parameter('w_du', 0.2)   ## 1e-1  0.2

        # 动作限制（8 维推力，单位 N）
        self.declare_parameter('u_min', [-20.0]*8)
        self.declare_parameter('u_max', [ 20.0]*8)

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
        self.trajectory = self.generate_smooth_3d_trajectory(self.waypoints, num_points=500)
        self.ref_index = 0
        self.search_ahead = 10  ## 40
        self.advance_thresh = 0.5  ## 0.5

        # 世界模型/标准化器（MoE）
        ckpt_path = cli_args.ckpt
        std_path = cli_args.std
        if (not std_path) and ckpt_path:
            std_path = os.path.join(os.path.dirname(ckpt_path), "standardizer.npz")
        self.model, self.std = self._load_world_model(ckpt_path, std_path, device=device)

        # 滚动优化 warm start
        self.prev_mean = np.zeros((self.N, self.model.cfg.u_dim), dtype=np.float32)

        # 状态
        self.state = None
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(6, dtype=np.float64)  # [v_world(3), w_world(3)]
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # wxyz
        self.initialized = False
        self.step_count = 0
        self.done = False

        self.last_state_time = 0.0
        self.current_state_time = 0.0

        # 日志（保持不变）
        results_dir = './logs'
        os.makedirs(results_dir, exist_ok=True)
        # self.log_path = os.path.join(results_dir, "wmpc_rov_log_0825_1950.h5")
        # self.log_path = os.path.join(results_dir, "wmpc_MoE_rov_log_0903_1130.h5")
        # self.log_path = os.path.join(results_dir, "wmpc_MoE_rov_log_0903_1140.h5")
        # self.log_path = os.path.join(results_dir, "wmpc_MoE_rov_log_0903_1150.h5")
        # self.log_path = os.path.join(results_dir, "wmpc_MoE_rov_log_0904_1320.h5")
        # self.log_path = os.path.join(results_dir, "wmpc_MoE_rov_log_0904_1330.h5")
        self.log_path = os.path.join(results_dir, "wmpc_MoE_rov_log_0904_1340.h5")
        self.h5 = h5py.File(self.log_path, "w")
        self._init_h5_dsets()

        # 定时器驱动控制
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.start_time = time.perf_counter()
        self.fast_forward = 1.0
        self.time_optimize_value = 0.0
        self.dt_optimize_gain = 0.05

        self._last_pub_t = None

        self.get_logger().info("WMPC (MoE) controller initialized.")

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

        self.current_state_time = time.time()

    # ---------------- 控制主循环 ----------------
    def control_loop(self):

        if self.done or (self.step_count >= self.max_steps):
            if not self.done:
                self.get_logger().info("Reached max steps. Stopping.")
            self.done = True
            return

        timeout = time.time() + 0.05
        while self.last_state_time == self.current_state_time and time.time() < timeout:
            self.last_state_time = self.current_state_time
            return

        self.last_state_time = self.current_state_time

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

        # 当前状态 -> 标准化空间
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

    def wait_time_optimizer(self, start_time, end_time):
        dt_error = (self.dt / self.fast_forward) - (end_time - start_time)
        if dt_error < 0.1:
            self.time_optimize_value += self.dt_optimize_gain * dt_error
            self.time_optimize_value = np.clip(self.time_optimize_value, -0.1, 0.1)

    # ---------------- 发布动作（保持不变） ----------------
    def publish_thrusters(self, thrusts: np.ndarray):
        thrusts = thrusts.astype(np.float32).reshape(-1, )
        msg = Float32MultiArray()
        msg.data = thrusts.tolist()

        # 监测发布频率/抖动
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

        self.thrust_pub.publish(msg)

    # ---------------- CEM 优化（保持不变，模型/标准化改为 MoE） ----------------
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
            # 温和的均值平滑
            mean = 0.7 * mean + 0.3 * self.prev_mean

        if best_seq is None:
            best_seq = np.zeros((N, u_dim), dtype=np.float32)

        # 第一步动作
        u0 = best_seq[0].copy()

        # 滚动 warm start：把均值向前平移一格
        self.prev_mean[:-1] = best_seq[1:]
        self.prev_mean[-1] = 0.0

        # 裁剪输出
        u0 = np.clip(u0, self.u_min.reshape(-1,), self.u_max.reshape(-1,))
        return u0

    # ---------------- 轨迹生成（保持不变） ----------------
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

    # ---------------- 重置仿真到第一个航点（保持不变） ----------------
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

    # ---------------- HDF5 日志（保持不变） ----------------
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

    # ---------------- 世界模型加载（替换为 MoE） ----------------
    def _load_world_model(self, ckpt_path: str, std_path: str, device: str):
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError("必须提供有效的 ckpt_path（.pt）用于加载 MoE 世界模型。")
        if not std_path or not os.path.isfile(std_path):
            raise FileNotFoundError("必须提供有效的 std_path（standardizer.npz）。")

        ckpt = torch.load(ckpt_path, map_location='cpu')
        cfg_dict = ckpt.get("cfg", None)
        if cfg_dict is None:
            raise RuntimeError("checkpoint 中缺少 cfg 字段")
        cfg = WMConfigMoE3(**cfg_dict)
        cfg.device = device

        std = NPZStandardizer.load(std_path)

        model = MoEWorldModel3(cfg).to(cfg.device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        self.get_logger().info(f"Loaded MoE world model from: {ckpt_path}")
        self.get_logger().info(f"Loaded standardizer from: {std_path}")
        return model, std


# ---------------- main（保持不变，仅默认路径改为 MoE 目录） ----------------
def default_waypoints() -> List[List[float]]:
    return [
        [5.0,   0.0, -10.0],
        [12.0, 10.0, -20.0],
        [20.0,-10.0,  -5.0],
        [28.0,  5.0, -18.0],
        [35.0,  0.0,  -8.0],
    ]


def parse_cli_args():
    parser = argparse.ArgumentParser(description="WM(MoE)-based MPC path following controller")
    # parser.add_argument("--ckpt", type=str, default='./checkpoints_moe3_aligned/moe3_epoch200.pt', help="MoE 世界模型 checkpoint 路径（.pt）")
    parser.add_argument("--ckpt", type=str, default='./checkpoints_moe3_18/moe3_epoch200.pt', help="MoE 世界模型 checkpoint 路径（.pt）")
    # parser.add_argument("--std", type=str, default='./checkpoints_moe3_aligned/standardizer.npz', help="standardizer.npz 路径（默认与 ckpt 同目录）")
    parser.add_argument("--std", type=str, default='./checkpoints_moe3_18/standardizer.npz', help="standardizer.npz 路径（默认与 ckpt 同目录）")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=10)  ## 10-20
    parser.add_argument("--max_steps", type=int, default=2000)
    return parser.parse_args()


def main():
    cli = parse_cli_args()

    rclpy.init()

    node = WMPCController(
        cli_args=cli,
        waypoints=default_waypoints(),
        dt=cli.dt,
        horizon=cli.horizon,
        max_steps=cli.max_steps
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        while rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=0.11)
    finally:
        try:
            node.h5.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
