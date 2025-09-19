
from dataclasses import dataclass
from typing import Dict, Optional, List, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    quat_normalize, quat_mul, so3_exp,
    build_delta_targets_seq, build_delta_targets_step
)

@dataclass
class WMConfigMoE3:
    # core dims
    x_dim: int = 13
    u_dim: int = 8
    n_experts: int = 3
    # shared/experts
    h_dim: int = 256
    ff_hidden: int = 256
    # gate
    gate_h_dim: int = 256
    gumbel_tau: float = 0.6
    gate_train_noisy: bool = True
    use_hard_routing: bool = False
    # variance bounds (match worldModel.py)
    min_logvar: float = -8.0
    max_logvar: float = 5.0
    # multi-step consistency
    k_consistency: int = 15  ## 5 稳定
    # device
    device: str = "cpu"

    ##
    regime_ce_weight: float = 0.01
    lb_weight: float = 0.01
    entropy_weight: float = 0.01

class GRUExpert(nn.Module):
    """An expert predicts a diagonal Gaussian over 12-dim deltas."""
    def __init__(self, cfg: WMConfigMoE3):
        super().__init__()
        self.cfg = cfg
        self.gru_in = nn.Linear(cfg.x_dim + cfg.u_dim, cfg.h_dim)
        self.gru = nn.GRU(cfg.h_dim, cfg.h_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(cfg.h_dim, cfg.ff_hidden), nn.SiLU())
        self.mu = nn.Linear(cfg.ff_hidden, 12)
        self.logvar_raw = nn.Linear(cfg.ff_hidden, 12)

    def forward(self, x: torch.Tensor, u: torch.Tensor, h0: Optional[torch.Tensor] = None):
        z = torch.cat([x, u], dim=-1)
        z = self.gru_in(z)
        y, h = self.gru(z, h0)
        f = self.head(y)
        mu = self.mu(f)
        # map raw -> [min_logvar, +inf) using softplus scaling like worldModel.py
        span = self.cfg.max_logvar - self.cfg.min_logvar
        logvar = self.cfg.min_logvar + F.softplus(self.logvar_raw(f)) * (span / math.log1p(math.e))
        return mu, logvar, h

class GateNet(nn.Module):
    def __init__(self, cfg: WMConfigMoE3):
        super().__init__()
        self.cfg = cfg
        self.gru_in = nn.Linear(cfg.x_dim + cfg.u_dim, cfg.gate_h_dim)
        self.gru = nn.GRU(cfg.gate_h_dim, cfg.gate_h_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(cfg.gate_h_dim, cfg.gate_h_dim), nn.SiLU(), nn.Linear(cfg.gate_h_dim, cfg.n_experts))

    def forward(self, x: torch.Tensor, u: torch.Tensor, h0: Optional[torch.Tensor] = None, train: bool = True):
        z = torch.cat([x, u], dim=-1)
        z = self.gru_in(z)
        y, h = self.gru(z, h0)
        logits = self.head(y)
        if train and self.cfg.gate_train_noisy:
            w = F.gumbel_softmax(logits, tau=self.cfg.gumbel_tau, hard=False, dim=-1)
        else:
            if self.cfg.use_hard_routing:
                w = F.gumbel_softmax(logits, tau=self.cfg.gumbel_tau, hard=True, dim=-1)
            else:
                w = F.softmax(logits, dim=-1)
        return logits, w, h

class MoEWorldModel3(nn.Module):
    """Mixture-of-Experts world model that matches the loss/training style of worldModel.py.

    It outputs a *single* diagonal Gaussian by moment-matching the expert mixture:
      mu = sum_k w_k * mu_k
      var = sum_k w_k * (var_k + (mu_k - mu)^2)
    so we can reuse the exact NLL and multi-step consistency losses.
    """
    def __init__(self, cfg: WMConfigMoE3):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList([GRUExpert(cfg) for _ in range(cfg.n_experts)])
        self.gate = GateNet(cfg)

    @staticmethod
    def split(delta: torch.Tensor):
        return delta[..., :3], delta[..., 3:6], delta[..., 6:9], delta[..., 9:12]

    def compose_next(self, x_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        p, q, v, w = x_t[..., 0:3], x_t[..., 3:7], x_t[..., 7:10], x_t[..., 10:13]
        dp, dv, dw, dth = self.split(delta)
        dq = so3_exp(dth)
        q_next = quat_mul(dq, quat_normalize(q))
        return torch.cat([p + dp, quat_normalize(q_next), v + dv, w + dw], dim=-1)

    def _forward_all(self, x: torch.Tensor, u: torch.Tensor, h0: Optional[Dict[str, Any]] = None, train: bool = True):
        # Experts
        mu_list: List[torch.Tensor] = []
        lv_list: List[torch.Tensor] = []
        h_exp_in = (h0.get("exp") if (h0 is not None and isinstance(h0, dict) and "exp" in h0) else [None] * len(self.experts))
        new_h_exp: List[torch.Tensor] = []
        for i, exp in enumerate(self.experts):
            mu_k, lv_k, h_k = exp(x, u, h_exp_in[i])
            mu_list.append(mu_k); lv_list.append(lv_k); new_h_exp.append(h_k)
        M = torch.stack(mu_list, dim=-2)      # (B,T,K,12)
        LV = torch.stack(lv_list, dim=-2)     # (B,T,K,12)
        # Gate
        h_gate_in = (h0.get("gate") if (h0 is not None and isinstance(h0, dict)) else None)
        logits, w, h_gate = self.gate(x, u, h_gate_in, train=train)  # (B,T,K)
        # Moment-matched single Gaussian
        var = torch.exp(LV).clamp_min(1e-8)   # (B,T,K,12)
        mu = (M * w.unsqueeze(-1)).sum(dim=-2)  # (B,T,12)
        var_total = (var + (M - mu.unsqueeze(-2)) ** 2)  # (B,T,K,12)
        var_mm = (var_total * w.unsqueeze(-1)).sum(dim=-2)  # (B,T,12)
        logvar = var_mm.clamp_min(1e-12).log()
        h_out = {"exp": new_h_exp, "gate": h_gate}
        return {"mu": mu, "logvar": logvar, "w": w, "logits": logits, "h": h_out}

    def forward(self, x: torch.Tensor, u: torch.Tensor, h0: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        return self._forward_all(x, u, h0=h0, train=self.training)

    def nll(self, pred: Dict[str, torch.Tensor], delta_gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mu, logv = pred["mu"], pred["logvar"]
        inv_var = torch.exp(-logv)
        nll = 0.5 * (((delta_gt - mu) ** 2) * inv_var + logv).sum(dim=-1)  # (B,T)
        if mask is not None:
            nll = (nll * mask).sum() / mask.sum().clamp(min=1)
        else:
            nll = nll.mean()
        return nll

def multi_step_consistency_loss(model: MoEWorldModel3,
                                x_seq: torch.Tensor, u_seq: torch.Tensor,
                                k_steps: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    B, T, _ = x_seq.shape
    if mask is not None:
        valid_pair = (mask[:, :-1] * mask[:, 1:]).sum(dim=0) > 0
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

def train_one_epoch(model: MoEWorldModel3, optimizer, loader, cfg: WMConfigMoE3):
    model.train()
    total_nll, total_cons = 0.0, 0.0
    for batch in loader:
        x = batch["x"].to(cfg.device)
        u = batch["u"].to(cfg.device)
        x_next = batch["x_next"].to(cfg.device)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(cfg.device)

        delta_gt = build_delta_targets_seq(x, x_next)

        optimizer.zero_grad()
        pred = model(x, u)
        nll = model.nll(pred, delta_gt, mask)

        # light log-variance regularizer, same as worldModel.py
        target_logvar = 0.0
        lv_reg = ((pred["logvar"] - target_logvar) ** 2)
        if mask is not None:
            lv_reg = (lv_reg.sum(dim=-1) * mask).sum() / mask.sum().clamp(min=1)
        else:
            lv_reg = lv_reg.mean()

        cons = multi_step_consistency_loss(model, x, u, cfg.k_consistency, mask)
        loss = nll + cons + 1e-4 * lv_reg

        # if cfg.regime_ce_weight > 0 and ("regime" in batch):
        if cfg.regime_ce_weight > 0:
            reg = batch["regime_step"].to(cfg.device).long()  # (B,T)
            ce = F.cross_entropy(pred["logits"].reshape(-1, cfg.n_experts),
                                 reg.reshape(-1), reduction="mean")
            loss += cfg.regime_ce_weight * ce

        # if cfg.lb_weight > 0:
        #     p = pred["w"].mean(dim=(0, 1))  # (K,)
        #     kl = torch.sum(p * (torch.log(p + 1e-8) - math.log(1.0 / cfg.n_experts)))
        #     loss += cfg.lb_weight * kl
        #
        # if cfg.entropy_weight > 0:
        #     neg_entropy = (pred["w"] * torch.log(pred["w"] + 1e-8)).sum(dim=-1).mean()
        #     loss += cfg.entropy_weight * neg_entropy  # 这是 -H(w)；最小化等于最大化熵

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_nll += float(nll.detach())
        total_cons += float(cons.detach())

    n_batches = max(1, len(loader))
    return {"nll": total_nll / n_batches, "cons": total_cons / n_batches}

@torch.no_grad()
def rollout(model: MoEWorldModel3, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
    assert x0.dim() == 2, f"x0 should be (B,13), got {tuple(x0.shape)}"
    assert u_seq.dim() == 3, f"u_seq should be (B,T,u_dim), got {tuple(u_seq.shape)}"
    B, T, _ = u_seq.shape
    device = x0.device
    x_hat = torch.empty(B, T + 1, model.cfg.x_dim, device=device, dtype=x0.dtype)
    x_hat[:, 0] = x0
    h = None
    x_t = x0
    for t in range(T):
        pred = model(x_t.unsqueeze(1), u_seq[:, t:t+1, :], h)
        mu_t, h = pred["mu"][:, 0], pred["h"]
        x_t = model.compose_next(x_t, mu_t)
        x_hat[:, t + 1] = x_t
    return x_hat