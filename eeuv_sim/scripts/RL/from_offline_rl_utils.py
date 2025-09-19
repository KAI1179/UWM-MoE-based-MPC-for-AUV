#!/usr/bin/env python3
# offline_rl_utils.py
# 基于每个 episode 的演示轨迹做训练参考；上线时以给定 waypoints 生成目标轨迹。
import os
import h5py
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from scipy.interpolate import CubicSpline

# -------------------------
# 轨迹构建
# -------------------------
def build_smooth_traj_from_waypoints(waypoints: List[List[float]], num_points: int = 300) -> np.ndarray:
    wp = np.asarray(waypoints, dtype=np.float32).reshape(-1, 3)
    t = np.linspace(0.0, 1.0, len(wp))
    csx = CubicSpline(t, wp[:, 0])
    csy = CubicSpline(t, wp[:, 1])
    csz = CubicSpline(t, wp[:, 2])
    tt = np.linspace(0.0, 1.0, num_points)
    traj = np.stack([csx(tt), csy(tt), csz(tt)], axis=-1).astype(np.float32)
    return traj

def build_smooth_traj_from_positions(positions: np.ndarray, num_points: int = 300) -> np.ndarray:
    """
    使用某个 episode 的 position 序列拟合样条，得到平滑参考轨迹。
    positions: (T,3)
    """
    pos = np.asarray(positions, dtype=np.float32)
    T = pos.shape[0]
    if T < 4:
        return pos.copy()
    t = np.linspace(0.0, 1.0, T)
    csx = CubicSpline(t, pos[:, 0])
    csy = CubicSpline(t, pos[:, 1])
    csz = CubicSpline(t, pos[:, 2])
    tt = np.linspace(0.0, 1.0, num_points)
    traj = np.stack([csx(tt), csy(tt), csz(tt)], axis=-1).astype(np.float32)
    return traj

def nearest_traj_index(traj_xyz: np.ndarray, p: np.ndarray) -> int:
    d = np.linalg.norm(traj_xyz - p[None, :], axis=1)
    return int(np.argmin(d))

def traj_direction(traj_xyz: np.ndarray, idx: int) -> np.ndarray:
    i0 = max(idx - 1, 0)
    i1 = min(idx + 1, traj_xyz.shape[0] - 1)
    d = traj_xyz[i1] - traj_xyz[i0]
    n = np.linalg.norm(d) + 1e-8
    return (d / n).astype(np.float32)

def goal_features(traj_xyz: np.ndarray, p: np.ndarray, lookahead: int = 5) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    目标条件化特征：
      e = p_ref - p  (3)
      t_hat = 局部切线单位向量 (3)
    返回 (e, t_hat, ref_idx)
    """
    i = nearest_traj_index(traj_xyz, p)
    j = min(i + lookahead, traj_xyz.shape[0] - 1)
    p_ref = traj_xyz[j]
    e = (p_ref - p).astype(np.float32)
    t_hat = traj_direction(traj_xyz, j)
    return e, t_hat, j

# -------------------------
# 奖励
# -------------------------
def path_tracking_reward(p: np.ndarray,
                         v_lin: np.ndarray,
                         w_ang: np.ndarray,
                         a_thr: np.ndarray,
                         traj_xyz: np.ndarray,
                         lookahead: int = 5,
                         w_pos: float = 1.0,
                         w_v: float = 0.05,
                         w_w: float = 0.05,
                         w_u: float = 0.01,
                         u_max: float = 20.0) -> Tuple[float, int]:
    e, t_hat, idx = goal_features(traj_xyz, p, lookahead=lookahead)
    r = - w_pos * float(np.linalg.norm(e)) \
        - w_v * float(np.linalg.norm(v_lin)) \
        - w_w * float(np.linalg.norm(w_ang)) \
        - w_u * float(np.linalg.norm(a_thr) / (u_max + 1e-8))

    r *= 0.2
    return r, idx

# -------------------------
# 状态与标准化
# -------------------------
def fuse_state_core(pos, ori_wxyz, v_lin, w_ang) -> np.ndarray:
    """
    核心 13 维：p(3)+q(wxyz)(4)+v(3)+w(3)
    """
    return np.asarray([
        pos[0], pos[1], pos[2],
        ori_wxyz[0], ori_wxyz[1], ori_wxyz[2], ori_wxyz[3],
        v_lin[0], v_lin[1], v_lin[2],
        w_ang[0], w_ang[1], w_ang[2],
    ], dtype=np.float32)

def fuse_state_with_goal(pos, ori_wxyz, v_lin, w_ang, e_goal, t_hat) -> np.ndarray:
    """
    拼接目标条件化特征：e_goal(3) + t_hat(3)
    最终观测维度 = 13 + 6 = 19
    """
    core = fuse_state_core(pos, ori_wxyz, v_lin, w_ang)
    return np.concatenate([core, e_goal.astype(np.float32), t_hat.astype(np.float32)], axis=0)

@dataclass
class Standardizer:
    x_mean: np.ndarray
    x_std:  np.ndarray
    u_mean: np.ndarray
    u_std:  np.ndarray

    def apply_x(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        # 对 19 维中：p(0:3)、v(7:10)、w(10:13)、e_goal+t_hat(13:19) 做标准化；四元数(3:7)不做线性标准化
        idx = np.r_[np.arange(0,3), np.arange(7,13), np.arange(13,19)]
        out[idx] = (out[idx] - self.x_mean[idx]) / (self.x_std[idx] + 1e-8)
        return out

    def apply_u(self, u: np.ndarray) -> np.ndarray:
        return (u - self.u_mean) / (self.u_std + 1e-8)

    def invert_u(self, u_norm: np.ndarray) -> np.ndarray:
        return u_norm * (self.u_std + 1e-8) + self.u_mean

    def save(self, path: str):
        np.savez_compressed(path, x_mean=self.x_mean, x_std=self.x_std, u_mean=self.u_mean, u_std=self.u_std)

    @staticmethod
    def load(path: str) -> "Standardizer":
        z = np.load(path)
        return Standardizer(z["x_mean"], z["x_std"], z["u_mean"], z["u_std"])

def compute_standardizer(transitions: Dict[str, np.ndarray]) -> Standardizer:
    X = transitions["obs"]
    U = transitions["act"]
    x_mean = X.mean(axis=0)
    x_std  = X.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    u_mean = U.mean(axis=0)
    u_std  = U.std(axis=0)
    u_std[u_std < 1e-6] = 1.0
    return Standardizer(x_mean, x_std, u_mean, u_std)

# -------------------------
# 数据加载（按 episode 轨迹作为参考）
# -------------------------
def load_hdf5_dataset(paths: List[str],
                      num_traj_points: int = 300,
                      lookahead: int = 5,
                      u_max_abs: float = 20.0) -> Dict[str, np.ndarray]:
    """
    以每个 episode 的 position 序列拟合得到参考轨迹；对每个 step 计算
    目标特征 e_goal(3)、t_hat(3)，并以此计算奖励。
    输出：
      obs(19=13+6), act(8), rew(1), next_obs(19), done(1)
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
                ori  = np.asarray(g["orientation"])     # [w,x,y,z]
                vlin = np.asarray(g["linear_velocity"])
                wang = np.asarray(g["angular_velocity"])
                # thr  = np.asarray(g["thrusts"])  ## 不分工况的 hdf5数据集
                thr  = np.asarray(g["thrusts_cmd"])  ## 分工况的 hdf5数据集

                T = min(pos.shape[0], thr.shape[0], ori.shape[0], vlin.shape[0], wang.shape[0])
                if T < 2:
                    continue

                # 用该 episode 自身的位置序列拟合平滑参考轨迹
                # ref_traj = build_smooth_traj_from_positions(pos[:T], num_points=num_traj_points)
                ref_traj = build_smooth_traj_from_positions(pos[:T], num_points=T)

                for t in range(T-1):
                    # 构造 (s_t, a_t, r_t, s_{t+1}, done)
                    e_goal, t_hat, _ = goal_features(ref_traj, pos[t], lookahead=lookahead)
                    s  = fuse_state_with_goal(pos[t],  ori[t],  vlin[t],  wang[t],  e_goal, t_hat)

                    sp_e, sp_t, _ = goal_features(ref_traj, pos[t+1], lookahead=lookahead)
                    sp = fuse_state_with_goal(pos[t+1], ori[t+1], vlin[t+1], wang[t+1], sp_e, sp_t)

                    a  = np.clip(thr[t].astype(np.float32), -u_max_abs, u_max_abs)
                    r, _ = path_tracking_reward(pos[t], vlin[t], wang[t], a, ref_traj,
                                                lookahead=lookahead, u_max=u_max_abs)

                    done_flag = float(t == T-2)  # 只有最后一条 transition 置 1
                    Obs.append(s); Act.append(a); Rew.append([r]); Next.append(sp); Done.append([done_flag])

                episodes += 1

    data = {
        "obs": np.asarray(Obs, dtype=np.float32),
        "act": np.asarray(Act, dtype=np.float32),
        "rew": np.asarray(Rew, dtype=np.float32),
        "next_obs": np.asarray(Next, dtype=np.float32),
        "done": np.asarray(Done, dtype=np.float32),
        "obs_dim": 19,
        "goal_dim": 6
    }
    print(f"[INFO] Loaded episodes={episodes}, transitions={data['obs'].shape[0]}, obs_dim={data['obs_dim']}")
    return data
