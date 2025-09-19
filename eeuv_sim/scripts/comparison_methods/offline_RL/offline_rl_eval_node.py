#!/usr/bin/env python3
# offline_rl_td3bc_policy_node.py
# 上线：以“指定的默认 waypoints”生成目标轨迹；实时计算 e_goal 与 t_hat，输入策略；输出 8 维推力到 /ucat/thruster_cmd
import os
import time
import h5py
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32MultiArray, Float32
from gazebo_msgs.msg import EntityState
from eeuv_sim.srv import ResetToPose

from offline_rl_utils import (
    Standardizer,
    build_smooth_traj_from_waypoints,
    goal_features
)

import torch.nn as nn

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

def fuse_state_core(pos, ori_wxyz, v_lin, w_ang) -> np.ndarray:
    return np.asarray([
        pos[0], pos[1], pos[2],
        ori_wxyz[0], ori_wxyz[1], ori_wxyz[2], ori_wxyz[3],
        v_lin[0], v_lin[1], v_lin[2],
        w_ang[0], w_ang[1], w_ang[2],
    ], dtype=np.float32)

class H5Logger:
    """
    HDF5 记录器（参考 WMPC_2.py 风格）：
      /run/time      (T,)   相对起始时间（秒）
      /run/position  (T,3)  位置
      /run/error     (T,)   到参考点误差范数
      /meta/waypoints (N,3)
      /meta/traj      (M,3)
      attrs: hz, lookahead, start_time_epoch
    """
    def __init__(self, path: str, waypoints: np.ndarray, traj: np.ndarray, hz: float, lookahead: int):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = h5py.File(path, "w")
        g_run = self.f.create_group("run")
        g_meta = self.f.create_group("meta")

        # meta
        g_meta.create_dataset("waypoints", data=waypoints.astype(np.float32))
        g_meta.create_dataset("traj", data=traj.astype(np.float32))
        g_meta.attrs["hz"] = float(hz)
        g_meta.attrs["lookahead"] = int(lookahead)
        g_meta.attrs["start_time_epoch"] = float(time.time())

        max_t = None
        self.d_time = g_run.create_dataset("time", shape=(0,), maxshape=(max_t,), dtype=np.float64, chunks=True)
        self.d_pos  = g_run.create_dataset("position", shape=(0,3), maxshape=(max_t,3), dtype=np.float32, chunks=True)
        self.d_err  = g_run.create_dataset("error", shape=(0,), maxshape=(max_t,), dtype=np.float32, chunks=True)
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

class OfflinePolicyNode(Node):
    def __init__(self,
                 ckpt_path: str,
                 stdz_path: str,
                 waypoints: np.ndarray,
                 hz: float = 10.0,
                 max_steps: int = 3000,
                 lookahead: int = 5,
                 logdir: str = "./eval_logs"):
        super().__init__("offline_rl_policy_node")

        # I/O
        self.pub_cmd = self.create_publisher(Float32MultiArray, "/ucat/thruster_cmd", 2)
        # self.pub_err = self.create_publisher(Float32, "/ucat/track_error", 10)
        self.sub_state = self.create_subscription(EntityState, "/ucat/state", self._on_state, 2)
        self.reset_cli = self.create_client(ResetToPose, "/reset_to_pose")

        # 策略
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt["cfg"]
        self.max_action = float(cfg["max_action"])
        self.obs_dim = int(cfg.get("obs_dim", 19))
        self.actor = Actor(self.obs_dim, int(cfg["action_dim"]), self.max_action)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.actor.to(self.device)

        # 标准化器
        self.stdz = Standardizer.load(stdz_path)

        # 目标轨迹（测试/部署时使用的默认 waypoints）
        self.waypoints = waypoints.reshape(-1,3)
        self.traj = build_smooth_traj_from_waypoints(self.waypoints.tolist(), num_points=300)
        self.lookahead = int(lookahead)

        # 控制时序
        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self.max_steps = int(max_steps)
        self.state = None

        # 复位到首个 waypoint（安全等待服务）
        self._reset_to_first_wp(self.waypoints[0])

        ## end flag
        self.done = False


        self.last_state_time = 0.0
        self.current_state_time = 0.0
        self.step_count = 0

        # 定时器驱动控制
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.start_time = time.perf_counter()
        self.fast_forward = 1.0
        self.time_optimize_value = 0.0
        self.dt_optimize_gain = 0.05

        # 日志（HDF5）
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(logdir, exist_ok=True)
        self.h5_path = os.path.join(logdir, f"rov_eval_{ts}.h5")
        self.logger = H5Logger(self.h5_path, self.waypoints, self.traj, hz=self.hz, lookahead=self.lookahead)
        self.t0 = time.time()

        self.get_logger().info(f"Offline RL policy node ready. Logging to: {self.h5_path}")

    def _reset_to_first_wp(self, first_wp):
        if not self.reset_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("reset service not available; skip reset.")
            return
        req = ResetToPose.Request()
        x, y, z = first_wp
        req.x, req.y, req.z = float(x), float(-y), float(-z)
        req.roll = 0.0; req.pitch = 0.0; req.yaw = 0.0
        future = self.reset_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

    def _on_state(self, msg: EntityState):
        self.state = msg
        self.current_state_time = time.time()

    def wait_time_optimizer(self, start_time, end_time):
        # 计算时间误差
        dt_error = (self.dt / self.fast_forward) - (end_time - start_time)

        # 如果时间误差小于 0.1，更新 time_optimize_value
        if dt_error < 0.1:
            self.time_optimize_value += self.dt_optimize_gain * dt_error
            self.time_optimize_value = np.clip(self.time_optimize_value, -0.1, 0.1)  # 限制时间优化值的更新范围

    @torch.no_grad()
    def control_loop(self):
    # def step_once(self):
    #     print('111', time.time())

        timeout = time.time() + 0.05
        while self.last_state_time == self.current_state_time and time.time() < timeout:
            # print('111+1', time.time())
            # print('current time:', time.time(), 'last_seen:', last_seen)
            self.last_state_time = self.current_state_time
            # rclpy.spin_once(self)
            return

        if self.state is None:
            return

        if self.done or (self.step_count >= self.max_steps):
            if not self.done:
                self.get_logger().info("Reached max steps. Stopping.")
            self.done = True
            return

        self.last_state_time = self.current_state_time

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

        e_goal, t_hat, _ = goal_features(self.traj, pos, lookahead=self.lookahead)
        x = np.concatenate([fuse_state_core(pos, ori, vlin, wang), e_goal, t_hat], axis=0)

        # 标准化：p,v,w,以及 e_goal/t_hat 做标准化；四元数原样
        x_n = x.copy()
        idx = np.r_[np.arange(0,3), np.arange(7,13), np.arange(13,19)]
        x_n[idx] = (x_n[idx] - self.stdz.x_mean[idx]) / (self.stdz.x_std[idx] + 1e-8)

        # print('222', time.time())
        xt = torch.as_tensor(x_n, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(xt).cpu().numpy()[0]
        a = np.clip(a, -self.max_action, self.max_action)

        # print('333', time.time())
        # 推力发布
        self.pub_cmd.publish(Float32MultiArray(data=a.astype(np.float32).tolist()))

        # 误差发布 + HDF5 记录（10Hz）
        e_norm = float(np.linalg.norm(e_goal))
        # self.pub_err.publish(Float32(data=e_norm))
        t_sec = time.time() - self.t0
        self.logger.append(t_sec, pos, e_norm)

        # 终点判断
        # if self.ref_index >= (len(self.trajectory) - 2) and np.linalg.norm(self.position - self.trajectory[-1]) < 0.5:
        if np.linalg.norm(pos - self.traj[-1]) < 0.5:
            self.get_logger().info("Goal reached.")
            self.done = True
        print(self.step_count)

        # print('333', time.time())
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
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--logdir", type=str, default="./eval_logs")
    parser.add_argument("--waypoints", type=float, nargs="+",
                        default=[
                            [5,   0,  -10],
                            [12,  10, -20],
                            [20, -10, -5],
                            [28,   5, -18],
                            [35,   0,  -8]
                        ])
    args = parser.parse_args()

    rclpy.init()
    node = OfflinePolicyNode(
        ckpt_path=args.ckpt,
        stdz_path=args.stdz,
        waypoints=np.asarray(args.waypoints, dtype=np.float32),
        hz=args.hz,
        max_steps=args.max_steps,
        lookahead=args.lookahead,
        logdir=args.logdir
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        # steps = 0
        # 与 hz=10 对齐的循环（0.1s）
        # period = 1.0 / float(args.hz)
        while rclpy.ok() and not node.done:
            start = time.time()
            executor.spin_once(timeout_sec=0.1)
            # node.step_once()
            # steps += 1
            # 睡眠到下一个周期
            # dt = time.time() - start
            # if dt < period:
            #     time.sleep(max(0.0, period - dt))
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
