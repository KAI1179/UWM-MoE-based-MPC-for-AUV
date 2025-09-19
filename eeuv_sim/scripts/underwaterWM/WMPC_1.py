#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WMPC.py
- 基于 underwaterWM/worldModel.py 的 GRU 世界模型做 MPC（CEM）
- 状态输入话题：/ucat/state （gazebo_msgs/EntityState）——与 MPC_5.py 对齐
- 控制周期：10 Hz（dt=0.1）——与 MPC_5.py 对齐
- 控制输出：/ucat/thruster_cmd （std_msgs/Float32MultiArray，8 维推力）
- 严格按 run.py/test_single.py 方式加载 checkpoint（ckpt['model_state'], ckpt['cfg']）
  和 standardizer（同目录 standardizer.npz，或 --standardizer 指定）
"""

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import EntityState  # 与 MPC_5.py 一致

from scipy.interpolate import CubicSpline
from eeuv_sim.srv import ResetToPose

# ---- 引入世界模型与工具（与仓库保持一致）----
# 期望文件结构：scripts/underwaterWM/{worldModel.py, utils.py}
# 运行时请确保 PYTHONPATH 已包含 scripts/ 或把 WMPC.py 放到同级
from worldModel import WMConfig, ROVGRUModel
from worldModel import rollout as wm_rollout  # 备用
from utils import Standardizer, quat_normalize_np

# waypoints = [
    #     [2.0, -3.0, -5.0],
    #     [6.0, 5.0, -10.0],
    #     [10.0, -5.0, -2.0],
    #     [14.0, 2.0, -9.0],
    #     [18.0, 0.0, -4.0]
    # ]

waypoints = [
    [5.0, 0.0, -10.0],  # 起点靠近x最小边界
    [12.0, 10.0, -20.0],  # 向右上拐弯并下降
    [20.0, -10.0, -5.0],  # 向左下折返并上升
    [28.0, 5.0, -18.0],  # 向右上方再次转折下降
    [35.0, 0.0, -8.0]  # 终点靠近x最大边界
]

# =========================
# 小工具
# =========================

def qnormalize(q: np.ndarray) -> np.ndarray:
    return quat_normalize_np(q)

def quat_to_yaw(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    s = 2.0*(w*z + x*y)
    c = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(s, c)

def yaw_err(q_wxyz: np.ndarray, yaw_ref: float) -> float:
    yaw = quat_to_yaw(q_wxyz)
    e = (yaw_ref - yaw + math.pi) % (2*math.pi) - math.pi
    return e

def saturate(u: np.ndarray, umin: np.ndarray, umax: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(u, umin), umax)

# =========================
# Standardizer 适配
# =========================

class StdAdapter:
    """
    使用 underwaterWM.utils.Standardizer（standardizer.npz）
    约定：apply_x_np 对 x=[p,q,v,w] 中的 p,v,w 做 z-score，q 仅单位化
    """
    def __init__(self, std: Optional[Standardizer]):
        self.std = std

    def x(self, x13: np.ndarray) -> np.ndarray:
        if self.std is None:
            x = x13.copy()
            x[3:7] = qnormalize(x[3:7])
            return x
        return self.std.apply_x_np(x13)

    def u(self, u: np.ndarray) -> np.ndarray:
        if self.std is None:
            return u
        return self.std.apply_u_np(u)

# =========================
# 世界模型包装（按 worldModel.py 的签名）
# =========================

class WM:
    def __init__(self, ckpt_path: str, device: str, std_path: Optional[str] = None):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # run.py 的保存格式：{"model_state", "cfg", ...}
        if "cfg" not in ckpt or "model_state" not in ckpt:
            raise RuntimeError("checkpoint 缺少 'cfg' 或 'model_state' 字段，请使用 run.py 训练生成的 ckpt")

        cfg = WMConfig(**ckpt["cfg"])
        cfg.device = device
        self.model = ROVGRUModel(cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

        # standardizer：优先 --standardizer，否则取 ckpt 同目录 standardizer.npz
        if std_path is None or std_path == "":
            guess = os.path.join(os.path.dirname(ckpt_path), "standardizer.npz")
            std_path = guess if os.path.exists(guess) else None

        self.std = StdAdapter(Standardizer.load(std_path) if std_path is not None else None)
        self.u_dim = cfg.u_dim
        self.x_dim = cfg.x_dim
        self.h = None  # GRU 隐状态（保持跨周期）

    @torch.no_grad()
    def reset_hidden(self):
        self.h = None

    @torch.no_grad()
    def step(self, x_np: np.ndarray, u_np: np.ndarray) -> np.ndarray:
        """
        单步：numpy -> torch -> model -> torch -> numpy（遵循 worldModel.forward 返回 dict）
        """
        x_std = self.std.x(x_np).astype(np.float32)
        u_std = self.std.u(u_np).astype(np.float32)

        x_t = torch.from_numpy(x_std).to(self.device).view(1, 1, -1)   # (B=1,T=1,13)
        u_t = torch.from_numpy(u_std).to(self.device).view(1, 1, -1)   # (B=1,T=1,u_dim)
        pred = self.model(x_t, u_t, self.h)                            # dict: mu, logvar, h
        mu_t = pred["mu"][:, 0]                                        # (1,12)
        self.h = pred["h"]                                             # 更新隐状态
        x_next = self.model.compose_next(x_t[:, 0, :], mu_t)           # (1,13)
        return x_next[0].detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def rollout_np(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """开环多步滚动，返回 (T+1,13)"""
        saved_h = self.h
        self.reset_hidden()
        T = U.shape[0]
        X = np.zeros((T+1, self.x_dim), dtype=np.float64)
        X[0] = x0
        x = x0
        for t in range(T):
            x = self.step(x, U[t])
            X[t+1] = x
        self.h = saved_h
        return X

# =========================
# CEM-MPC
# =========================

@dataclass
class MPCWeights:
    w_pos: float = 5.0
    w_yaw: float = 1.5
    w_vel: float = 0.1
    w_omega: float = 0.05
    w_u: float = 1e-3
    w_du: float = 1e-2
    w_terminal: float = 2.0

class CEM_MPC:
    def __init__(self,
                 wm: WM,
                 horizon: int = 15,
                 umin: Optional[np.ndarray] = None,
                 umax: Optional[np.ndarray] = None,
                 popsize: int = 512,
                 elite_frac: float = 0.1,
                 iters: int = 5,
                 dt: float = 0.1,
                 weights: MPCWeights = MPCWeights()):
        self.wm = wm
        self.T = horizon
        self.u_dim = wm.u_dim
        self.umin = umin if umin is not None else -np.ones(self.u_dim)
        self.umax = umax if umax is not None else +np.ones(self.u_dim)
        self.pop = popsize
        self.elite = max(1, int(popsize * elite_frac))
        self.iters = iters
        self.dt = dt
        self.w = weights

        self.mean = np.zeros((self.T, self.u_dim))
        self.std = 0.6 * np.ones((self.T, self.u_dim))
        self.u_prev = np.zeros(self.u_dim)

    def _traj_cost(self, X: np.ndarray, U: np.ndarray, p_ref: np.ndarray, yaw_ref: float) -> float:
        pos = X[:, 0:3]
        quat = X[:, 3:7]
        vel = X[:, 7:10]
        omg = X[:, 10:13]

        ep = pos[:-1] - p_ref[None, :]
        ey = np.array([yaw_err(quat[t], yaw_ref) for t in range(self.T)])

        ep_T = pos[-1] - p_ref
        ey_T = yaw_err(quat[-1], yaw_ref)

        dU = np.diff(np.vstack([self.u_prev[None, :], U]), axis=0)

        stage = (
            self.w.w_pos * np.sum(ep**2, axis=1) +
            self.w.w_yaw * (ey**2) +
            self.w.w_vel * np.sum(vel[:-1]**2, axis=1) +
            self.w.w_omega * np.sum(omg[:-1]**2, axis=1) +
            self.w.w_u * np.sum(U**2, axis=1) +
            self.w.w_du * np.sum(dU**2, axis=1)
        )
        terminal = self.w.w_terminal * (self.w.w_pos*np.sum(ep_T**2) + self.w.w_yaw*(ey_T**2))
        return float(np.sum(stage) + terminal)

    # def plan(self, x0: np.ndarray, p_ref: np.ndarray, yaw_ref: float) -> np.ndarray:
    def plan(self, x0: np.ndarray, trajectory) -> np.ndarray:
        mean = self.mean.copy()
        std = self.std.copy()

        for _ in range(self.iters):
            U = np.random.normal(mean[None, :, :], std[None, :, :], size=(self.pop, self.T, self.u_dim))
            U = np.clip(U, self.umin, self.umax)

            J = np.zeros(self.pop)
            for i in range(self.pop):
                X = self.wm.rollout_np(x0, U[i])
                J[i] = self._traj_cost(X, U[i], p_ref, yaw_ref)

            elite_idx = np.argsort(J)[:self.elite]
            E = U[elite_idx]
            mean = E.mean(axis=0)
            std = E.std(axis=0) + 1e-5

        # warm start
        self.mean = np.vstack([mean[1:], mean[-1:]])
        self.std = np.vstack([std[1:], std[-1:]])

        u0 = 0.7*mean[0] + 0.3*self.u_prev
        u0 = saturate(u0, self.umin, self.umax)
        self.u_prev = u0.copy()
        return u0

# =========================
# ROS2 节点（/ucat/state @ 10Hz → /ucat/thruster_cmd）
# =========================

class WMPCNode(Node):
    def __init__(self, args):
        super().__init__("wmpc_node")

        # ---- 基本参数 ----
        self.ckpt = args.checkpoint
        self.std_path = args.standardizer
        self.device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"

        # 10Hz 控制周期（与 MPC_5.py 一致）
        self.dt = 0.1 if args.dt is None else float(args.dt)

        # 目标
        self.target_p = np.array(args.target, dtype=np.float64)
        self.target_yaw = math.radians(args.target_yaw_deg)

        self.reset_cli = self.create_client(ResetToPose, '/reset_to_pose')

        self._reset_to_first_waypoint()

        self.trajectory = self.generate_smooth_3d_trajectory(self.waypoints, num_points=300)
        self.initialized = False
        self.done = False
        self.goal = self.trajectory[-1]

        # 限幅
        self.u_min = np.array(args.u_min, dtype=np.float64)
        self.u_max = np.array(args.u_max, dtype=np.float64)

        # ---- 世界模型 & MPC ----
        self.wm = WM(self.ckpt, device=self.device, std_path=self.std_path)
        weights = MPCWeights(
            w_pos=args.w_pos, w_yaw=args.w_yaw, w_vel=args.w_vel,
            w_omega=args.w_omega, w_u=args.w_u, w_du=args.w_du, w_terminal=args.w_terminal
        )
        self.mpc = CEM_MPC(
            wm=self.wm, horizon=args.horizon, popsize=args.popsize,
            elite_frac=args.elite_frac, iters=args.iters, dt=self.dt,
            umin=self.u_min, umax=self.u_max, weights=weights
        )

        # ---- 话题：状态订阅 /ucat/state（EntityState） & 推力发布 /ucat/thruster_cmd ----
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        self.state = None  # 13 维
        self.state_sub = self.create_subscription(EntityState, "/ucat/state", self._cb_state, qos)
        self.pub = self.create_publisher(Float32MultiArray, "/ucat/thruster_cmd", qos)

        # 定时器：10Hz
        self.timer = self.create_timer(self.dt, self._control_step)
        self.get_logger().info(f"[WMPC] Running at {1.0/self.dt:.1f} Hz, state topic=/ucat/state, cmd topic=/ucat/thruster_cmd")

    def _nearest_traj_index(self, p):
        d = np.linalg.norm(self.trajectory - p[None, :], axis=1)
        return int(np.argmin(d))

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

    # ---- 状态回调：EntityState -> x(13) ----
    def _cb_state(self, msg: EntityState):
        p = msg.pose.position
        q = msg.pose.orientation  # geometry_msgs/Quaternion (x,y,z,w)
        v = msg.twist.linear
        w = msg.twist.angular

        x = np.zeros(13, dtype=np.float64)
        x[0:3] = [p.x, p.y, p.z]
        x[3:7] = qnormalize(np.array([q.w, q.x, q.y, q.z], dtype=np.float64))  # [w,x,y,z]
        x[7:10] = [v.x, v.y, v.z]
        x[10:13] = [w.x, w.y, w.z]
        self.state = x

    # ---- 控制主循环：CEM-MPC -> /ucat/thruster_cmd ----
    def _control_step(self):
        if self.state is None:
            return
        try:



            u = self.mpc.plan(self.state.copy(), self.target_p, self.target_yaw)



        except Exception as e:
            self.get_logger().error(f"[WMPC] MPC plan error: {e}")
            u = np.zeros(8, dtype=np.float64)

        msg = Float32MultiArray()
        msg.data = [float(v) for v in u.tolist()]  # 8 维推力
        self.pub.publish(msg)

        # 简单日志
        ep = np.linalg.norm(self.state[0:3] - self.target_p)
        ey = abs(yaw_err(self.state[3:7], self.target_yaw))
        self.get_logger().info(f"[WMPC] |pos|err={ep:.3f}  |yaw|err={math.degrees(ey):.2f}deg  u0={u[0]:.2f}")

# =========================
# CLI
# =========================

def build_argparser():
    p = argparse.ArgumentParser(description="WorldModel-based MPC for ROV (10Hz).")
    p.add_argument("--checkpoint", type=str, required=True, help="world model ckpt (from run.py, contains 'model_state' & 'cfg')")
    p.add_argument("--standardizer", type=str, default="", help="path to standardizer.npz (default: same dir as ckpt)")
    p.add_argument("--cuda", action="store_true", help="use CUDA if available")

    # 10Hz 默认 dt=0.1（与 MPC_5.py 保持一致）
    p.add_argument("--dt", type=float, default=None, help="control period in seconds (default 0.1)")

    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--popsize", type=int, default=512)
    p.add_argument("--elite_frac", type=float, default=0.1)
    p.add_argument("--iters", type=int, default=4)

    p.add_argument("--u-min", dest="u_min", type=float, nargs=8, default=[-1.0]*8, help="min thrust per channel (8)")
    p.add_argument("--u-max", dest="u_max", type=float, nargs=8, default=[+1.0]*8, help="max thrust per channel (8)")

    p.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, -1.0], help="target position [x y z]")
    p.add_argument("--target-yaw-deg", dest="target_yaw_deg", type=float, default=0.0, help="target yaw in degrees")

    # 代价权重（必要时可调）
    p.add_argument("--w_pos", type=float, default=5.0)
    p.add_argument("--w_yaw", type=float, default=1.5)
    p.add_argument("--w_vel", type=float, default=0.1)
    p.add_argument("--w_omega", type=float, default=0.05)
    p.add_argument("--w_u", type=float, default=1e-3)
    p.add_argument("--w_du", type=float, default=1e-2)
    p.add_argument("--w_terminal", type=float, default=2.0)
    p.add_argument("--waypoints", type=float, default=2.0)

    return p

def main():
    args = build_argparser().parse_args()

    rclpy.init()
    node = WMPCNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
