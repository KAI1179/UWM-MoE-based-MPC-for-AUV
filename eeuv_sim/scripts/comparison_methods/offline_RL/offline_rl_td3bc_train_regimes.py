#!/usr/bin/env python3
# offline_rl_td3bc_train_regimes.py
# 目的：在“regimes 数据集 A”上进行离线 RL（TD3+BC）训练
# 设计：最大限度复用 offline_rl_td3bc_train.py 的结构/接口/保存方式，
#       仅对数据装载做“动作字段健壮读取”，保持与 offline_rl_eval_node.py 完全兼容。

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

# 复用 utils 中的轨迹/奖励/状态构造与标准化器，保证与评测端一致
from offline_rl_utils import (
    build_smooth_traj_from_positions,
    goal_features,
    path_tracking_reward,
    fuse_state_with_goal,
    compute_standardizer,
    Standardizer,
)

# -------------------------
# 神经网络（与参考训练脚本一致）
# -------------------------
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

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, a):
        z = torch.cat([x, a], dim=-1)
        return self.q1(z), self.q2(z)

# -------------------------
# TD3+BC（与参考训练脚本一致）
# -------------------------
class TD3BC:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 device: str = "cpu",
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha_bc: float = 4,  ## 2.5
                 policy_noise: float = 0.0,
                 noise_clip: float = 0.0):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_t = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_t = Critic(state_dim, action_dim).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.gamma = gamma
        self.tau = tau
        self.alpha_bc = alpha_bc
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_action = float(max_action)

    @torch.no_grad()
    def _target_action(self, x):
        a = self.actor_t(x)
        if self.policy_noise > 0.0:
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a = (a + noise).clamp(-self.max_action, self.max_action)
        return a

    def train_step(self, batch, update_actor: bool = True):
        obs      = torch.as_tensor(batch["obs"], device=self.device, dtype=torch.float32)
        act      = torch.as_tensor(batch["act"], device=self.device, dtype=torch.float32)
        rew      = torch.as_tensor(batch["rew"], device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device, dtype=torch.float32)
        done     = torch.as_tensor(batch["done"], device=self.device, dtype=torch.float32)

        # ------- Critic -------
        with torch.no_grad():
            next_a = self._target_action(next_obs)
            q1_t, q2_t = self.critic_t(next_obs, next_a)
            q_t = torch.min(q1_t, q2_t)
            y = rew + (1.0 - done) * self.gamma * q_t

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.opt_critic.step()

        info = {"critic_loss": float(critic_loss.item())}

        # ------- Actor (TD3+BC with adaptive lambda) -------
        if update_actor:
            a_pi = self.actor(obs)
            q1_pi, q2_pi = self.critic(obs, a_pi)
            q_pi = torch.min(q1_pi, q2_pi)

            bc_loss = F.mse_loss(a_pi, act)
            lambda_coef = self.alpha_bc / (torch.abs(q_pi).mean().detach() + 1e-4)
            policy_loss = - q_pi.mean() + lambda_coef * bc_loss

            self.opt_actor.zero_grad(set_to_none=True)
            policy_loss.backward()
            self.opt_actor.step()

            # Polyak
            with torch.no_grad():
                for p, tp in zip(self.actor.parameters(), self.actor_t.parameters()):
                    tp.copy_(self.tau * p + (1 - self.tau) * tp)
                for p, tp in zip(self.critic.parameters(), self.critic_t.parameters()):
                    tp.copy_(self.tau * p + (1 - self.tau) * tp)

            info.update({
                "actor_loss": float(policy_loss.item()),
                "bc_loss": float(bc_loss.item()),
                "q_pi": float(q_pi.mean().item()),
                "lambda": float(lambda_coef.item()),
            })
        return info

# -------------------------
# 数据加载（Regimes 专用，保持输出结构一致）
# -------------------------
def load_hdf5_dataset_regimes(paths,
                              num_traj_points: int = 300,
                              lookahead: int = 5,
                              u_max_abs: float = 20.0):
    """
    与 offline_rl_utils.load_hdf5_dataset 输出完全一致：
      obs(19), act(8), rew(1), next_obs(19), done(1)
    差异点：动作字段健壮读取，优先使用 thr_applied→thr→thr_cmd，
    以适配 data_collector_regimes.py 生成的数据集 A。
    """
    Obs, Act, Rew, Next, Done = [], [], [], [], []
    episodes = 0

    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] dataset not found: {p}")
            continue
        with h5py.File(p, "r") as f:
            for gname in sorted(list(f.keys())):
                if not gname.startswith("episode_"):
                    continue
                g = f[gname]

                pos  = np.asarray(g["position"])
                ori  = np.asarray(g["orientation"])       # [w,x,y,z]
                vlin = np.asarray(g["linear_velocity"])
                wang = np.asarray(g["angular_velocity"])

                # --- 关键改动：动作字段的健壮读取 ---
                if "thrusts_applied" in g:
                    thr = np.asarray(g["thrusts_applied"])
                elif "thrusts" in g:
                    thr = np.asarray(g["thrusts"])
                elif "thrusts_cmd" in g:
                    thr = np.asarray(g["thrusts_cmd"])
                else:
                    # 缺少动作，跳过该 episode
                    continue

                T = min(pos.shape[0], thr.shape[0], ori.shape[0], vlin.shape[0], wang.shape[0])
                if T < 2:
                    continue

                # 参考轨迹：以“该 episode 的位置序列”拟合（与参考实现一致）
                # 这里用 T 点（逐点对齐），保证 goal_features 的近邻/切线方向与评测端一致
                ref_traj = build_smooth_traj_from_positions(pos[:T], num_points=T)

                for t in range(T - 1):
                    # 当前步的目标条件特征
                    e_goal, t_hat, _ = goal_features(ref_traj, pos[t], lookahead=lookahead)
                    s = fuse_state_with_goal(pos[t],  ori[t],  vlin[t],  wang[t],  e_goal, t_hat)

                    # 下一步的目标条件特征
                    sp_e, sp_t, _ = goal_features(ref_traj, pos[t+1], lookahead=lookahead)
                    sp = fuse_state_with_goal(pos[t+1], ori[t+1], vlin[t+1], wang[t+1], sp_e, sp_t)

                    # 动作与奖励
                    a = np.clip(thr[t].astype(np.float32), -u_max_abs, u_max_abs)
                    r, _ = path_tracking_reward(pos[t], vlin[t], wang[t], a, ref_traj,
                                                lookahead=lookahead, u_max=u_max_abs)

                    done_flag = float(t == T - 2)
                    Obs.append(s); Act.append(a); Rew.append([r]); Next.append(sp); Done.append([done_flag])

                episodes += 1

    data = {
        "obs": np.asarray(Obs, dtype=np.float32),
        "act": np.asarray(Act, dtype=np.float32),
        "rew": np.asarray(Rew, dtype=np.float32),
        "next_obs": np.asarray(Next, dtype=np.float32),
        "done": np.asarray(Done, dtype=np.float32),
        "obs_dim": 19,
        "goal_dim": 6,
    }
    print(f"[INFO][regimes] Loaded episodes={episodes}, transitions={data['obs'].shape[0]}, obs_dim={data['obs_dim']}")
    return data

# -------------------------
# 训练入口（参数/保存方式与参考脚本保持一致）
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True,
                        help="HDF5 路径（支持多个），例如: /path/to/rov_data_regimes_xxx.hdf5")
    parser.add_argument("--save_dir", type=str, default="./offline_rl_ckpt_18_2")
    parser.add_argument("--max_action", type=float, default=20.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=400000)  ## 200000
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traj_points", type=int, default=300)
    parser.add_argument("--lookahead", type=int, default=5)  # 5
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) 加载数据（Regimes 数据集 A）
    data = load_hdf5_dataset_regimes(
        args.data,
        num_traj_points=args.traj_points,
        lookahead=args.lookahead,
        u_max_abs=args.max_action
    )

    # 2) 标准化器（与参考脚本一致保存）
    stdz = compute_standardizer(data)
    stdz.save(os.path.join(args.save_dir, "standardizer.npz"))

    obs = data["obs"]; act = data["act"]; rew = data["rew"]; next_obs = data["next_obs"]; done = data["done"]

    # 与参考脚本一致的观测标准化方案：p/v/w/e_goal/t_hat 做 z-score，四元数不动
    def norm_x(x):
        out = x.copy()
        idx = np.r_[np.arange(0,3), np.arange(7,13), np.arange(13,19)]
        out[:, idx] = (out[:, idx] - stdz.x_mean[idx]) / (stdz.x_std[idx] + 1e-8)
        return out

    obs_n = norm_x(obs)
    next_obs_n = norm_x(next_obs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = TD3BC(state_dim=obs.shape[1], action_dim=act.shape[1], max_action=args.max_action, device=device)

    def sample_batch(B: int):
        idx = np.random.randint(0, obs.shape[0], size=B)
        return {
            "obs": obs_n[idx],
            "act": act[idx],
            "rew": rew[idx],
            "next_obs": next_obs_n[idx],
            "done": done[idx],
        }

    log_every = 1000
    for it in range(1, args.updates + 1):
        info = agent.train_step(sample_batch(args.batch_size), update_actor=True)
        if it % log_every == 0:
            print(f"[{it}/{args.updates}] critic={info['critic_loss']:.4f} "
                  f"actor={info.get('actor_loss', np.nan):.4f} "
                  f"bc={info.get('bc_loss', np.nan):.4f} "
                  f"q={info.get('q_pi', np.nan):.3f} "
                  f"lambda={info.get('lambda', np.nan):.3f}")

    ckpt = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "cfg": {
            "state_dim": obs.shape[1],
            "action_dim": act.shape[1],
            "max_action": args.max_action,
            "goal_dim": 6,
            "obs_dim": obs.shape[1],
        }
    }
    torch.save(ckpt, os.path.join(args.save_dir, "td3bc_policy.pt"))
    print(f"[DONE][regimes] Saved to: {args.save_dir}")

if __name__ == "__main__":
    main()
