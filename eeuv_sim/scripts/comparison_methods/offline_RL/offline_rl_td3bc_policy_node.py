#!/usr/bin/env python3
# offline_rl_td3bc_policy_node.py
# 上线：以“指定的默认 waypoints”生成目标轨迹；实时计算 e_goal 与 t_hat，输入策略；输出 8 维推力到 /ucat/thruster_cmd
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float32MultiArray
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

class OfflinePolicyNode(Node):
    def __init__(self,
                 ckpt_path: str,
                 stdz_path: str,
                 waypoints: np.ndarray,
                 hz: float = 10.0,
                 max_steps: int = 3000,
                 lookahead: int = 5):
        super().__init__("offline_rl_policy_node")

        # I/O
        self.pub_cmd = self.create_publisher(Float32MultiArray, "/ucat/thruster_cmd", 2)
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
        self.traj = build_smooth_traj_from_waypoints(waypoints.reshape(-1,3).tolist(), num_points=300)
        self.lookahead = int(lookahead)

        # 控制时序
        self.dt = 1.0 / float(hz)
        self.max_steps = int(max_steps)
        self.state = None

        # 复位到首个 waypoint（安全等待服务）
        self._reset_to_first_wp(waypoints.reshape(-1,3)[0])

        self.get_logger().info("Offline RL policy node ready.")

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

    @torch.no_grad()
    def step_once(self):
        if self.state is None:
            return

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

        xt = torch.as_tensor(x_n, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(xt).cpu().numpy()[0]
        a = np.clip(a, -self.max_action, self.max_action)

        self.pub_cmd.publish(Float32MultiArray(data=a.astype(np.float32).tolist()))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--stdz", type=str, required=True)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--lookahead", type=int, default=5)
    parser.add_argument("--waypoints", type=float, nargs="+",
                        default=[
                            2.0, -3.0, -5.0,
                            6.0,  5.0,-10.0,
                           10.0, -5.0, -2.0,
                           14.0,  2.0, -9.0,
                           18.0,  0.0, -4.0
                        ])
    args = parser.parse_args()

    rclpy.init()
    node = OfflinePolicyNode(
        ckpt_path=args.ckpt,
        stdz_path=args.stdz,
        waypoints=np.asarray(args.waypoints, dtype=np.float32),
        hz=args.hz,
        max_steps=args.max_steps,
        lookahead=args.lookahead
    )
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        steps = 0
        while rclpy.ok() and steps < args.max_steps:
            executor.spin_once(timeout_sec=0.1) ## 0.05
            node.step_once()
            steps += 1
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
