#!/usr/bin/env python3
# offline_rl_td3bc_train_regimes_multiway.py
# 目标：只用数据集A（regimes）的 (s,a,s')，但用“多样化外部轨迹库”重标注 (e_goal, t_hat)
#      训练与评测目标含义一致；评测轨迹从未在训练出现，体现泛化。
import os, argparse, json, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import h5py

from offline_rl_utils import (
    build_smooth_traj_from_waypoints,  # 生成/平滑外部轨迹
    goal_features,                      # 从外部轨迹 + 当前pos 生成 (e_goal, t_hat)
    path_tracking_reward,               # 奖励（仍参照外部轨迹）
    fuse_state_with_goal,
    compute_standardizer,
    Standardizer,
)

# ------------------ 模型（与原TD3+BC保持一致） ------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.max_action = float(max_action)
    def forward(self, x): return self.max_action * torch.tanh(self.net(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
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

class TD3BC:
    def __init__(self, state_dim, action_dim, max_action, device="cpu",
                 gamma=0.99, tau=0.005, alpha_bc=2.5, policy_noise=0.0, noise_clip=0.0):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_t = Actor(state_dim, action_dim, max_action).to(device); self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_t = Critic(state_dim, action_dim).to(device); self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.gamma, self.tau = gamma, tau
        self.alpha_bc, self.policy_noise, self.noise_clip = alpha_bc, policy_noise, noise_clip
        self.max_action = float(max_action)

    @torch.no_grad()
    def _targ(self, x):
        a = self.actor_t(x)
        if self.policy_noise > 0:
            n = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a = (a + n).clamp(-self.max_action, self.max_action)
        return a

    def train_step(self, batch, update_actor=True):
        obs      = torch.as_tensor(batch["obs"], device=self.device, dtype=torch.float32)
        act      = torch.as_tensor(batch["act"], device=self.device, dtype=torch.float32)
        rew      = torch.as_tensor(batch["rew"], device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(batch["next_obs"], device=self.device, dtype=torch.float32)
        done     = torch.as_tensor(batch["done"], device=self.device, dtype=torch.float32)

        with torch.no_grad():
            next_a = self._targ(next_obs)
            q1t, q2t = self.critic_t(next_obs, next_a)
            y = rew + (1.0 - done) * self.gamma * torch.min(q1t, q2t)

        q1, q2 = self.critic(obs, act)
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_critic.zero_grad(set_to_none=True); c_loss.backward(); self.opt_critic.step()
        info = {"critic_loss": float(c_loss.item())}

        if update_actor:
            api = self.actor(obs)
            q1p, q2p = self.critic(obs, api)
            qpi = torch.min(q1p, q2p)
            bc = F.mse_loss(api, act)
            lam = self.alpha_bc / (torch.abs(qpi).mean().detach() + 1e-4)
            a_loss = - qpi.mean() + lam * bc
            self.opt_actor.zero_grad(set_to_none=True); a_loss.backward(); self.opt_actor.step()
            with torch.no_grad():
                for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                    pt.copy_(self.tau * p + (1 - self.tau) * pt)
                for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                    pt.copy_(self.tau * p + (1 - self.tau) * pt)
            info.update({"actor_loss": float(a_loss.item()), "bc_loss": float(bc.item()),
                         "q_pi": float(qpi.mean().item()), "lambda": float(lam.item())})
        return info

# ------------------ 轨迹库：加载或随机生成 ------------------
def _traj_from_waypoints_list(flat):
    arr = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
    return build_smooth_traj_from_waypoints(arr.tolist(), num_points=300)

def load_traj_bank_from_file(path):
    """
    支持两种格式：
    1) .json: [{"waypoints":[x1,y1,z1,...]},{"waypoints":[...]}]
    2) .npy:  shape=(K, M, 3) 的数组（K条轨迹，每条M点）
    """
    if path.endswith(".json"):
        js = json.load(open(path, "r"))
        bank = []
        for item in js:
            traj = _traj_from_waypoints_list(item["waypoints"])
            bank.append(traj)
        return bank
    elif path.endswith(".npy"):
        arr = np.load(path)  # (K, M, 3)
        return [arr[i] for i in range(arr.shape[0])]
    else:
        raise ValueError("Unsupported traj_bank file (use .json or .npy)")

def random_traj_bank(n_paths=32, n_ctrl=5, bounds=((0,40), (-15,15), (-22,-2)), seed=0):
    """
    用随机控制点生成若干条样条轨迹（与评测空间同量级但形状多样）
    - n_paths: 生成多少条
    - n_ctrl:  每条轨迹控制点个数（>=4）
    - bounds:  (x_range, y_range, z_range)
    """
    rng = np.random.default_rng(seed)
    bank = []
    for _ in range(n_paths):
        xs = rng.uniform(*bounds[0], size=(n_ctrl,))
        ys = rng.uniform(*bounds[1], size=(n_ctrl,))
        zs = rng.uniform(*bounds[2], size=(n_ctrl,))
        wps = np.stack([np.sort(xs), ys, zs], axis=1)  # x 单调增，避免自交
        traj = build_smooth_traj_from_waypoints(wps.tolist(), num_points=300)
        bank.append(traj)
    return bank

# ------------------ 数据加载：A集 + 轨迹库重标注 ------------------
def load_regimes_with_traj_bank(paths, traj_bank, lookahead=5, u_max_abs=20.0, pick_mode="per_episode"):
    """
    pick_mode:
      - "per_episode": 每个episode随机选一条轨迹
      - "per_step":    每个时间步随机选（更强数据增强）
    输出与原版一致：obs(19), act(8), rew(1), next_obs(19), done(1)
    """
    Obs, Act, Rew, Next, Done = [], [], [], [], []
    episodes = 0
    K = len(traj_bank)

    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] {p} not found"); continue
        with h5py.File(p, "r") as f:
            for gname in sorted([k for k in f.keys() if k.startswith("episode_")]):
                g = f[gname]
                pos  = np.asarray(g["position"])
                ori  = np.asarray(g["orientation"])
                vlin = np.asarray(g["linear_velocity"])
                wang = np.asarray(g["angular_velocity"])
                if   "thrusts_applied" in g: thr = np.asarray(g["thrusts_applied"])
                elif "thrusts"          in g: thr = np.asarray(g["thrusts"])
                elif "thrusts_cmd"      in g: thr = np.asarray(g["thrusts_cmd"])
                else: continue

                T = min(pos.shape[0], thr.shape[0], ori.shape[0], vlin.shape[0], wang.shape[0])
                if T < 2: continue

                if pick_mode == "per_episode":
                    traj = traj_bank[np.random.randint(0, K)]

                for t in range(T - 1):
                    if pick_mode == "per_step":
                        traj = traj_bank[np.random.randint(0, K)]

                    # 以“外部轨迹”生成条件目标与奖励
                    e, t_hat, _ = goal_features(traj, pos[t],   lookahead=lookahead)
                    s          = fuse_state_with_goal(pos[t],  ori[t],  vlin[t],  wang[t],  e, t_hat)
                    e2, t2, _  = goal_features(traj, pos[t+1], lookahead=lookahead)
                    sp         = fuse_state_with_goal(pos[t+1], ori[t+1], vlin[t+1], wang[t+1], e2, t2)

                    a = np.clip(thr[t].astype(np.float32), -u_max_abs, u_max_abs)
                    r, _ = path_tracking_reward(pos[t], vlin[t], wang[t], a, traj,
                                                lookahead=lookahead, u_max=u_max_abs)
                    Obs.append(s); Act.append(a); Rew.append([r]); Next.append(sp); Done.append([float(t==T-2)])

                episodes += 1

    data = {
        "obs": np.asarray(Obs, dtype=np.float32),
        "act": np.asarray(Act, dtype=np.float32),
        "rew": np.asarray(Rew, dtype=np.float32),
        "next_obs": np.asarray(Next, dtype=np.float32),
        "done": np.asarray(Done, dtype=np.float32),
        "obs_dim": 19, "goal_dim": 6,
    }
    print(f"[INFO][A|traj-bank] episodes={episodes}, transitions={data['obs'].shape[0]}")
    return data

# ------------------ 训练入口：保留原有接口/保存方式 ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="regimes 数据集A hdf5 路径（可多文件）")
    ap.add_argument("--save_dir", type=str, default="./offline_rl_ckpt_18_1")
    ap.add_argument("--max_action", type=float, default=20.0)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--updates", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lookahead", type=int, default=5)

    # 轨迹库来源（3选1，优先级从高到低）
    ap.add_argument("--traj_bank_file", type=str, default="", help=".json 或 .npy，含多条外部轨迹")
    ap.add_argument("--train_waypoints", type=float, nargs="+", default=[], help="单条外部轨迹（不推荐做泛化，但可用于对照）")
    ap.add_argument("--traj_bank_random", type=int, default=32, help="若未提供文件，则随机生成N条轨迹")
    ap.add_argument("--traj_bank_seed", type=int, default=0)

    ap.add_argument("--traj_pick_mode", type=str, choices=["per_episode","per_step"], default="per_episode",
                    help="per_episode=每个episode固定一条轨迹；per_step=每步随机一条（更强数据增强）")

    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) 构建“训练轨迹库”（与评测轨迹严格区分）
    if args.traj_bank_file:
        traj_bank = load_traj_bank_from_file(args.traj_bank_file)
    elif len(args.train_waypoints) > 0:
        # 单条（不建议做泛化）——但保留接口以方便做消融/对照
        traj_bank = [_traj_from_waypoints_list(args.train_waypoints)]
    else:
        traj_bank = random_traj_bank(n_paths=args.traj_bank_random, seed=args.traj_bank_seed)

    # 2) 加载数据（A集 + 轨迹库重标注）
    data = load_regimes_with_traj_bank(
        args.data, traj_bank=traj_bank, lookahead=args.lookahead,
        u_max_abs=args.max_action, pick_mode=args.traj_pick_mode
    )

    # 3) 标准化并保存（与原脚本一致）
    stdz = compute_standardizer(data)
    stdz.save(os.path.join(args.save_dir, "standardizer.npz"))

    obs, act, rew, next_obs, done = data["obs"], data["act"], data["rew"], data["next_obs"], data["done"]

    # 与原脚本一致：pos/v/w/e_goal/t_hat 标准化，四元数不动
    def norm_x(x):
        out = x.copy()
        idx = np.r_[np.arange(0,3), np.arange(7,13), np.arange(13,19)]
        out[:, idx] = (out[:, idx] - stdz.x_mean[idx]) / (stdz.x_std[idx] + 1e-8)
        return out
    obs_n, next_obs_n = norm_x(obs), norm_x(next_obs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent  = TD3BC(state_dim=obs.shape[1], action_dim=act.shape[1], max_action=args.max_action, device=device)

    def sample(B):
        idx = np.random.randint(0, obs.shape[0], size=B)
        return {"obs": obs_n[idx], "act": act[idx], "rew": rew[idx], "next_obs": next_obs_n[idx], "done": done[idx]}

    log_every = 1000
    for it in range(1, args.updates + 1):
        info = agent.train_step(sample(args.batch_size), update_actor=True)
        if it % log_every == 0:
            print(f"[{it}/{args.updates}] critic={info['critic_loss']:.4f} "
                  f"actor={info.get('actor_loss', np.nan):.4f} "
                  f"bc={info.get('bc_loss', np.nan):.4f} "
                  f"q={info.get('q_pi', np.nan):.3f} lambda={info.get('lambda', np.nan):.3f}")

    ckpt = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "cfg": {"state_dim": obs.shape[1], "action_dim": act.shape[1], "max_action": args.max_action,
                "goal_dim": 6, "obs_dim": obs.shape[1]}
    }
    torch.save(ckpt, os.path.join(args.save_dir, "td3bc_policy.pt"))
    print(f"[DONE] saved to: {args.save_dir}")
if __name__ == "__main__":
    main()
