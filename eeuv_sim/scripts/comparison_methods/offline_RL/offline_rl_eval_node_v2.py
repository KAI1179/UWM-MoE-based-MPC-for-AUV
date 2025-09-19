#!/usr/bin/env python3
# offline_rl_eval_node_v2.py
# 上线：与“未来点重标注（H 步 / D 米 lookahead）”训练严格对齐
import os
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import EntityState
from eeuv_sim.srv import ResetToPose

from offline_rl_utils import (
    Standardizer,
    build_smooth_traj_from_waypoints,  # 构建平滑外部任务轨迹
)

# ---------------------
# Model
# ---------------------
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.max_action = float(max_action)
    def forward(self, x):
        return self.max_action * torch.tanh(self.net(x))

# ---------------------
# Helpers
# ---------------------
def fuse_state_core(pos, ori_wxyz, v_lin, w_ang) -> np.ndarray:
    return np.asarray([
        pos[0], pos[1], pos[2],
        ori_wxyz[0], ori_wxyz[1], ori_wxyz[2], ori_wxyz[3],
        v_lin[0], v_lin[1], v_lin[2],
        w_ang[0], w_ang[1], w_ang[2],
    ], dtype=np.float32)

def unit(vec, eps=1e-6):
    n = float(np.linalg.norm(vec))
    if n < eps:
        return np.zeros_like(vec, dtype=np.float32), 0.0
    return (vec / n).astype(np.float32), n

def cumulative_arclength(traj_xyz: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(traj_xyz, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d).astype(np.float32)], axis=0)
    return s

def nearest_index(traj_xyz: np.ndarray, p: np.ndarray, k_window: int = 50, last_idx: int = None):
    if last_idx is None:
        d2 = np.sum((traj_xyz - p[None, :])**2, axis=1)
        return int(np.argmin(d2))
    i0 = max(0, last_idx - k_window)
    i1 = min(len(traj_xyz), last_idx + k_window + 1)
    seg = traj_xyz[i0:i1]
    d2 = np.sum((seg - p[None, :])**2, axis=1)
    return int(i0 + np.argmin(d2))

def local_goal_from_traj(traj_xyz: np.ndarray,
                         s_cum: np.ndarray,
                         p: np.ndarray,
                         last_idx: int,
                         lookahead_steps: int,
                         lookahead_dist: float = 0.0):
    """
    返回：g_pos, t_hat, idx_near, idx_goal
    - 先找轨迹最近点 idx_near（局部窗口，防止回退）
    - 若 lookahead_dist>0，用弧长表在 s[idx_near]+dist 处取 idx_goal
      否则使用 idx_near + lookahead_steps
    """
    M = traj_xyz.shape[0]
    idx0 = nearest_index(traj_xyz, p, k_window=50, last_idx=last_idx)

    if lookahead_dist is not None and lookahead_dist > 0.0:
        target_s = s_cum[idx0] + float(lookahead_dist)
        idxg = int(np.searchsorted(s_cum, target_s, side='left'))
        idxg = min(max(idxg, idx0 + 1), M - 1)
    else:
        idxg = min(idx0 + int(lookahead_steps), M - 1)

    g_pos = traj_xyz[idxg]
    e = (g_pos - p).astype(np.float32)
    t_hat, _ = unit(e)
    return g_pos.astype(np.float32), t_hat, idx0, idxg

class H5Logger:
    def __init__(self, path: str, waypoints: np.ndarray, traj: np.ndarray, hz: float, lookahead_steps: int, lookahead_dist: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = h5py.File(path, "w")
        g_run = self.f.create_group("run")
        g_meta = self.f.create_group("meta")

        g_meta.create_dataset("waypoints", data=waypoints.astype(np.float32))
        g_meta.create_dataset("traj", data=traj.astype(np.float32))
        g_meta.attrs["hz"] = float(hz)
        g_meta.attrs["lookahead_steps"] = int(lookahead_steps)
        g_meta.attrs["lookahead_dist"] = float(lookahead_dist)
        g_meta.attrs["start_time_epoch"] = float(time.time())

        self.d_time = g_run.create_dataset("time", shape=(0,), maxshape=(None,), dtype=np.float64, chunks=True)
        self.d_pos  = g_run.create_dataset("position", shape=(0,3), maxshape=(None,3), dtype=np.float32, chunks=True)
        self.d_err  = g_run.create_dataset("error", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True)
        self.n = 0
    def append(self, t_sec: float, pos_xyz: np.ndarray, err: float):
        i = self.n
        self.d_time.resize((i+1,))
        self.d_pos.resize((i+1, 3))
        self.d_err.resize((i+1,))
        self.d_time[i] = float(t_sec)
        self.d_pos[i, :] = pos_xyz.astype(np.float32)
        self.d_err[i] = float(err)
        self.n += 1
    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass

# ---------------------
# Node
# ---------------------
class OfflinePolicyNodeV2(Node):
    def __init__(self,
                 ckpt_path: str,
                 stdz_path: str,
                 waypoints: np.ndarray,
                 hz: float = 10.0,
                 max_steps: int = 3000,
                 lookahead_steps: int = 12,
                 lookahead_dist: float = 0.0,     # 若>0，优先生效（米）
                 progress_guard: bool = True,
                 progress_dot_min: float = 0.05,  # v·t_hat 下限
                 adapt_lookahead_min: int = 4,
                 action_smooth_tau: float = 0.2,  # s，指数平滑时间常数
                 action_rate_limit: float = 5.0,  # 每步最大动作变化（牛）
                 cmd_topic: str = "/ucat/thruster_cmd",
                 state_topic: str = "/ucat/state",
                 logdir: str = "./eval_logs"):
        super().__init__("offline_rl_eval_node_v2")

        # I/O
        self.pub_cmd = self.create_publisher(Float32MultiArray, cmd_topic, 2)
        self.sub_state = self.create_subscription(EntityState, state_topic, self._on_state, 2)
        self.reset_cli = self.create_client(ResetToPose, "/reset_to_pose")

        # 策略
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt["cfg"]
        self.max_action = float(cfg["max_action"])
        self.obs_dim = int(cfg.get("obs_dim", 19))
        self.action_dim = int(cfg["action_dim"])
        self.actor = Actor(self.obs_dim, self.action_dim, self.max_action)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor.to(self.device)

        # 标准化器（与训练一致）
        self.stdz = Standardizer.load(stdz_path)

        # 任务轨迹与弧长表
        self.waypoints = waypoints.reshape(-1, 3)
        self.traj = build_smooth_traj_from_waypoints(self.waypoints.tolist(), num_points=300)
        self.s_cum = cumulative_arclength(self.traj)

        # 控制参数
        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self.max_steps = int(max_steps)
        self.lookahead_steps = int(lookahead_steps)
        self.lookahead_dist = float(lookahead_dist)
        self.progress_guard = bool(progress_guard)
        self.progress_dot_min = float(progress_dot_min)
        self.adapt_lookahead_min = int(adapt_lookahead_min)

        # 平滑/限速
        self.action_prev = np.zeros(self.action_dim, dtype=np.float32)
        self.alpha_smooth = np.clip(self.dt / max(self.dt + action_smooth_tau, 1e-6), 0.0, 1.0)
        self.action_rate_limit = float(action_rate_limit)

        # 状态
        self.state = None
        self.done = False
        self.step_count = 0
        self.idx_last_near = None
        self.idx_last_goal = None

        # 复位
        self._reset_to_first_wp(self.waypoints[0])

        # 日志
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(logdir, exist_ok=True)
        self.h5_path = os.path.join(logdir, f"rov_eval_v2_{ts}.h5")
        self.logger = H5Logger(self.h5_path, self.waypoints, self.traj, hz=self.hz,
                               lookahead_steps=self.lookahead_steps, lookahead_dist=self.lookahead_dist)
        self.t0 = time.time()

        # 定时器驱动
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(f"Offline RL eval node v2 ready. Logging to: {self.h5_path}")

    def _reset_to_first_wp(self, first_wp):
        if not self.reset_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("reset service not available; skip reset.")
            return
        req = ResetToPose.Request()
        x, y, z = first_wp
        req.x, req.y, req.z = float(x), float(-y), float(-z)  # 与现有仿真坐标系保持一致
        req.roll = 0.0; req.pitch = 0.0; req.yaw = 0.0
        future = self.reset_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

    def _on_state(self, msg: EntityState):
        self.state = msg

    @torch.no_grad()
    def control_loop(self):
        if self.state is None or self.done:
            return
        if self.step_count >= self.max_steps:
            self.get_logger().info("Reached max steps. Stopping.")
            self.done = True
            return

        # 状态读取
        pos = np.array([self.state.pose.position.x,
                        self.state.pose.position.y,
                        self.state.pose.position.z], dtype=np.float32)
        ori = np.array([self.state.pose.orientation.w,
                        self.state.pose.orientation.x,
                        self.state.pose.orientation.y,
                        self.state.pose.orientation.z], dtype=np.float32)
        vlin = np.array([self.state.twist.linear.x,
                         self.state.twist.linear.y,
                         self.state.twist.linear.z], dtype=np.float32)
        wang = np.array([self.state.twist.angular.x,
                         self.state.twist.angular.y,
                         self.state.twist.angular.z], dtype=np.float32)

        # 局部未来目标（H 步 / D 米）
        g_pos, t_hat, idx0, idxg = local_goal_from_traj(
            self.traj, self.s_cum, pos, self.idx_last_near,
            lookahead_steps=self.lookahead_steps,
            lookahead_dist=self.lookahead_dist
        )
        self.idx_last_near = idx0
        self.idx_last_goal = idxg
        e_goal = (g_pos - pos).astype(np.float32)

        # 前向进展守护：若 v·t_hat 太小，尝试缩短 lookahead
        if self.progress_guard:
            dot = float(np.dot(vlin, t_hat))
            if dot < self.progress_dot_min and self.lookahead_steps > self.adapt_lookahead_min:
                _steps = max(self.adapt_lookahead_min, self.lookahead_steps // 2)
                g2, t2, _, _ = local_goal_from_traj(
                    self.traj, self.s_cum, pos, self.idx_last_near,
                    lookahead_steps=_steps, lookahead_dist=0.0 if self.lookahead_dist > 0 else 0.0
                )
                if np.dot(vlin, t2) > dot:
                    e_goal = (g2 - pos).astype(np.float32)
                    t_hat = t2

        # 观测拼接 + 标准化（pos/v/w/e_goal/t_hat 标准化，四元数原样）
        x = np.concatenate([fuse_state_core(pos, ori, vlin, wang), e_goal, t_hat], axis=0).astype(np.float32)
        x_n = x.copy()
        idx_std = np.r_[np.arange(0,3), np.arange(7,13), np.arange(13,19)]
        x_n[idx_std] = (x_n[idx_std] - self.stdz.x_mean[idx_std]) / (self.stdz.x_std[idx_std] + 1e-8)

        # 策略推理
        xt = torch.as_tensor(x_n, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(xt).cpu().numpy()[0]
        a = np.clip(a, -self.max_action, self.max_action)

        # 动作平滑 + 速率限制
        a_s = self.alpha_smooth * a + (1.0 - self.alpha_smooth) * self.action_prev
        da = np.clip(a_s - self.action_prev, -self.action_rate_limit, self.action_rate_limit)
        a_out = np.clip(self.action_prev + da, -self.max_action, self.max_action)

        # 发布
        self.pub_cmd.publish(Float32MultiArray(data=a_out.astype(np.float32).tolist()))
        self.action_prev = a_out

        # 记录
        e_norm = float(np.linalg.norm(e_goal))
        t_sec = time.time() - self.t0
        self.logger.append(t_sec, pos, e_norm)

        # 终点判断
        if np.linalg.norm(pos - self.traj[-1]) < 0.5:
            self.get_logger().info("Goal reached.")
            self.done = True

        self.step_count += 1

    def destroy_node(self):
        try:
            self.logger.close()
        except Exception:
            pass
        super().destroy_node()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--stdz", type=str, required=True)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=2000)

    # —— 与训练重标注一致的目标参数 ——
    parser.add_argument("--lookahead_steps", type=int, default=10, help="未来 H 步（若未设 lookahead_dist）")
    parser.add_argument("--lookahead_dist", type=float, default=0.0, help="未来 D 米（>0 优先生效）")

    # 守护/平滑
    parser.add_argument("--no_progress_guard", action="store_true")
    parser.add_argument("--progress_dot_min", type=float, default=0.05)
    parser.add_argument("--adapt_lookahead_min", type=int, default=4)
    parser.add_argument("--action_smooth_tau", type=float, default=0.2)
    parser.add_argument("--action_rate_limit", type=float, default=5.0)

    parser.add_argument("--cmd_topic", type=str, default="/ucat/thruster_cmd")
    parser.add_argument("--state_topic", type=str, default="/ucat/state")
    parser.add_argument("--logdir", type=str, default="./eval_logs")

    # 任务航点（外部目标）
    parser.add_argument("--waypoints", type=float, nargs="+",
                        default=[
                            5,   0, -10,
                           12,  10, -20,
                           20, -10,  -5,
                           28,   5, -18,
                           35,   0,  -8
                        ])
    args = parser.parse_args()

    rclpy.init()
    node = OfflinePolicyNodeV2(
        ckpt_path=args.ckpt,
        stdz_path=args.stdz,
        waypoints=np.asarray(args.waypoints, dtype=np.float32),
        hz=args.hz,
        max_steps=args.max_steps,
        lookahead_steps=args.lookahead_steps,
        lookahead_dist=args.lookahead_dist,
        progress_guard=(not args.no_progress_guard),
        progress_dot_min=args.progress_dot_min,
        adapt_lookahead_min=args.adapt_lookahead_min,
        action_smooth_tau=args.action_smooth_tau,
        action_rate_limit=args.action_rate_limit,
        cmd_topic=args.cmd_topic,
        state_topic=args.state_topic,
        logdir=args.logdir,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.done:
            executor.spin_once(timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
