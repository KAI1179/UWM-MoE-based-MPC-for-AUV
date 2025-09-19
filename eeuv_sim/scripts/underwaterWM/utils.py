# utils.py
# 四元数 / SO(3) 工具、标准化器、增量标签、随机种子

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch

# ---------- NumPy 端（用于数据集预处理） ----------
def quat_normalize_np(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """单位化四元数 (w,x,y,z)，q: (...,4)"""
    n = np.linalg.norm(q, axis=-1, keepdims=True) + eps
    return q / n

@dataclass
class Standardizer:
    """对 x=[p(3), q(4), v(3), w(3)] 与 u 做标准化；四元数只单位化不缩放。"""
    x_mean: np.ndarray  # (13,)
    x_std:  np.ndarray  # (13,)
    u_mean: np.ndarray  # (u_dim,)
    u_std:  np.ndarray  # (u_dim,)

    def apply_x_np(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        # 单位化四元数
        out[..., 3:7] = quat_normalize_np(out[..., 3:7])
        # 仅缩放 p(0:3), v(7:10), w(10:13)
        idx = np.r_[np.arange(0,3), np.arange(7,13)]
        out[..., idx] = (out[..., idx] - self.x_mean[idx]) / (self.x_std[idx] + 1e-8)
        return out

    def apply_u_np(self, u: np.ndarray) -> np.ndarray:
        return (u - self.u_mean) / (self.u_std + 1e-8)

    def save(self, path: str):
        np.savez_compressed(path,
                            x_mean=self.x_mean, x_std=self.x_std,
                            u_mean=self.u_mean, u_std=self.u_std)

    @staticmethod
    def load(path: str) -> "Standardizer":
        data = np.load(path)
        return Standardizer(
            x_mean=data["x_mean"], x_std=data["x_std"],
            u_mean=data["u_mean"], u_std=data["u_std"]
        )

# ---------- Torch 端（用于模型/损失） ----------
def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """四元数乘 [w,x,y,z]"""
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack((
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ), dim=-1)

def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """单位四元数的共轭即逆"""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def so3_exp(phi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SO(3) 指数映射：小角度向量 -> 四元数 [w,x,y,z]"""
    theta = torch.clamp(phi.norm(dim=-1, keepdim=True), min=eps)
    half = 0.5 * theta
    k = torch.sin(half) / theta
    w = torch.cos(half)
    v = k * phi
    return quat_normalize(torch.cat([w, v], dim=-1))

def so3_log(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SO(3) 对数映射：单位四元数 -> 小角度向量"""
    q = quat_normalize(q, eps)
    w, xyz = q[..., :1], q[..., 1:]
    w = torch.clamp(w, -1.0, 1.0)
    theta = 2.0 * torch.acos(w)
    s = torch.sqrt(torch.clamp(1 - w**2, min=0.0))
    k = torch.where(s < eps, torch.ones_like(s) * 2.0, theta / (s + eps))
    return k * xyz

def build_delta_targets_seq(x: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
    """
    (B,T,13) -> (B,T,12) 的增量标签 [Δp, Δv, Δw, Δθ]
    """
    p, q, v, w = x[..., 0:3], x[..., 3:7], x[..., 7:10], x[..., 10:13]
    pn, qn, vn, wn = x_next[..., 0:3], x_next[..., 3:7], x_next[..., 7:10], x_next[..., 10:13]
    dp = pn - p
    dv = vn - v
    dw = wn - w
    q = quat_normalize(q)
    qn = quat_normalize(qn)
    q_err = quat_mul(qn, quat_conj(q))  # qn = dq * q
    dth = so3_log(q_err)
    return torch.cat([dp, dv, dw, dth], dim=-1)

def build_delta_targets_step(x_t: torch.Tensor, x_tp1: torch.Tensor) -> torch.Tensor:
    """
    (B,13),(B,13) -> (B,12)
    """
    return build_delta_targets_seq(x_t.unsqueeze(1), x_tp1.unsqueeze(1))[:,0]

def set_seed(seed: int = 42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
