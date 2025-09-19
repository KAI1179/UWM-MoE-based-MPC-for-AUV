# worldModel.py
from dataclasses import dataclass
from typing import Dict, Optional
import math
import torch
import torch.nn as nn

from utils import (
    quat_normalize, quat_mul, so3_exp, quat_conj,
    build_delta_targets_seq, build_delta_targets_step
)

@dataclass
class WMConfig:
    x_dim: int = 13
    u_dim: int = 8
    h_dim: int = 256
    ff_hidden: int = 256
    min_logvar: float = -8.0
    max_logvar: float = 5.0
    k_consistency: int = 5
    device: str = "cpu"

class ROVGRUModel(nn.Module):
    """
    GRU state-only world model:
      inputs: (B,T,x_dim), (B,T,u_dim)
      outputs: diagonal Gaussian over 12-delta (mu, logvar)
    """
    def __init__(self, cfg: WMConfig):
        super().__init__()
        self.cfg = cfg
        self.gru_in = nn.Linear(cfg.x_dim + cfg.u_dim, cfg.h_dim)
        self.gru = nn.GRU(cfg.h_dim, cfg.h_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(cfg.h_dim, cfg.ff_hidden), nn.SiLU())
        self.mu = nn.Linear(cfg.ff_hidden, 12)
        # softplus parameterization for log-variance
        self.logvar_raw = nn.Linear(cfg.ff_hidden, 12)

    def forward(self, x: torch.Tensor, u: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        z = torch.cat([x, u], dim=-1)
        z = self.gru_in(z)
        y, h = self.gru(z, h0)
        f = self.head(y)
        mu = self.mu(f)
        # map softplus -> [min_logvar, max_logvar]
        span = self.cfg.max_logvar - self.cfg.min_logvar
        logvar = self.cfg.min_logvar + torch.nn.functional.softplus(self.logvar_raw(f)) * (span / math.log1p(math.e)) ## 仅设下限
        # logvar = self.cfg.min_logvar + torch.sigmoid(self.logvar_raw(f)) * span  ## 夹在区间
        return {"mu": mu, "logvar": logvar, "h": h}

    @staticmethod
    def split(delta: torch.Tensor):
        return delta[..., :3], delta[..., 3:6], delta[..., 6:9], delta[..., 9:12]

    def compose_next(self, x_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        p, q, v, w = x_t[..., 0:3], x_t[..., 3:7], x_t[..., 7:10], x_t[..., 10:13]
        dp, dv, dw, dth = self.split(delta)
        dq = so3_exp(dth)
        q_next = quat_mul(dq, quat_normalize(q))
        return torch.cat([p + dp, quat_normalize(q_next), v + dv, w + dw], dim=-1)

    def nll(self, pred: Dict[str, torch.Tensor], delta_gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mu, logv = pred["mu"], pred["logvar"]
        inv_var = torch.exp(-logv)
        nll = 0.5 * (((delta_gt - mu) ** 2) * inv_var + logv).sum(dim=-1)  # (B,T)
        if mask is not None:
            nll = (nll * mask).sum() / mask.sum().clamp(min=1)
        else:
            nll = nll.mean()
        return nll

def multi_step_consistency_loss(model: ROVGRUModel,
                                x_seq: torch.Tensor, u_seq: torch.Tensor,
                                k_steps: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    B, T, _ = x_seq.shape

    if mask is not None:
        # 只统计 t 与 t+1 均有效的步
        valid_pair = (mask[:, :-1] * mask[:, 1:]).sum(dim=0) > 0  # (T-1,)
        max_valid_k = int(valid_pair[:k_steps].sum().item())
        k_steps = max(1, min(k_steps, max_valid_k))
        if k_steps <= 0:
            return torch.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype)
    else:
        assert T >= k_steps + 1, "sequence too short for consistency"

    h = None
    x_t = x_seq[:, 0]
    losses = []

    for t in range(k_steps):
        pred = model(x_t.unsqueeze(1), u_seq[:, t:t+1], h)
        mu_t, logv_t, h = pred["mu"][:, 0], pred["logvar"][:, 0], pred["h"]

        delta_gt_t = build_delta_targets_step(x_t, x_seq[:, t+1])
        inv_var = torch.exp(-logv_t)
        nll_t = 0.5 * (((delta_gt_t - mu_t) ** 2) * inv_var + logv_t).sum(dim=-1)  # (B,)

        if mask is not None:
            m = (mask[:, t] * mask[:, t+1]).float()
            if m.sum() == 0:
                x_t = model.compose_next(x_t, mu_t)
                continue
            nll_t = (nll_t * m).sum() / m.sum()
        else:
            nll_t = nll_t.mean()

        losses.append(nll_t)
        x_t = model.compose_next(x_t, mu_t)

    if len(losses) == 0:
        return torch.tensor(0.0, device=x_seq.device, dtype=x_seq.dtype)
    return sum(losses) / len(losses)

def train_one_epoch(model: ROVGRUModel, optimizer, loader, cfg: WMConfig):
    model.train()
    total_nll, total_cons = 0.0, 0.0

    for batch in loader:
        x = batch["x"].to(cfg.device)         # (B,T,13)
        u = batch["u"].to(cfg.device)         # (B,T,u_dim)
        x_next = batch["x_next"].to(cfg.device)
        mask = batch["mask"].to(cfg.device) if "mask" in batch else None

        delta_gt = build_delta_targets_seq(x, x_next)

        optimizer.zero_grad()
        pred = model(x, u)
        nll = model.nll(pred, delta_gt, mask)

        # 轻微的 logvar 正则，抑制极端方差
        target_logvar = 0.0
        lv_reg = ((pred["logvar"] - target_logvar) ** 2)
        if mask is not None:
            lv_reg = (lv_reg.sum(dim=-1) * mask).sum() / mask.sum().clamp(min=1)
        else:
            lv_reg = lv_reg.mean()

        cons = multi_step_consistency_loss(model, x, u, cfg.k_consistency, mask)
        loss = nll + cons + 1e-4 * lv_reg
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_nll += float(nll.detach())
        total_cons += float(cons.detach())

    n_batches = max(1, len(loader))
    return {"nll": total_nll / n_batches, "cons": total_cons / n_batches}

@torch.no_grad()
def rollout(model, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
    """
    闭环多步预测：返回 [x0, x1, ..., xT]
    x0:   (B, 13)
    u_seq:(B, T, u_dim)
    ->    (B, T+1, 13)
    """
    assert x0.dim() == 2, f"x0 should be (B,13), got {tuple(x0.shape)}"
    assert u_seq.dim() == 3, f"u_seq should be (B,T,u_dim), got {tuple(u_seq.shape)}"

    B, T, _ = u_seq.shape
    device = x0.device
    x_hat = torch.empty(B, T + 1, model.cfg.x_dim, device=device, dtype=x0.dtype)
    x_hat[:, 0] = x0

    h = None
    x_t = x0  # (B,13)
    for t in range(T):
        x_in = x_t.unsqueeze(1)          # (B,1,13)
        u_in = u_seq[:, t:t+1, :]        # (B,1,u_dim)  ← 关键：确保三维
        pred = model(x_in, u_in, h)      # forward 期望三维输入
        mu_t, h = pred["mu"][:, 0], pred["h"]
        x_t = model.compose_next(x_t, mu_t)   # (B,13)
        x_hat[:, t+1] = x_t
    return x_hat