
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

# 依赖你项目中的工具模块；保持与原训练脚本一致
from utils import quat_normalize, quat_mul, so3_exp

@dataclass
class WMConfigMoE3:
    # Core dims
    x_dim: int = 13
    u_dim: int = 8
    n_experts: int = 3
    # Expert GRU
    gru_hidden: int = 256
    gru_layers: int = 1
    mlp_hidden: int = 256
    # Gate GRU
    gate_gru_hidden: int = 256
    gate_gru_layers: int = 1
    gate_gru_dropout: float = 0.0
    # Training / loss knobs (与训练脚本字段保持一致，便于从ckpt复原)
    rollout_horizon: int = 20
    huber_delta: float = 0.1
    gate_entropy_beta: float = 1e-2
    gate_tv_gamma: float = 2e-2
    gate_lb_alpha: float = 5e-2
    use_hard_routing: bool = False
    gumbel_tau: float = 0.6
    gate_train_noisy: bool = True
    fdi_bce_alpha: float = 0.0
    fdi_tv_alpha: float = 0.0
    fdi_sparse_alpha: float = 0.0
    act_consist_alpha: float = 0.0
    sat_weight: float = 0.3
    umax: float = 20.0
    du_max: float = 0.0
    reg_ce_alpha: float = 1e-1
    device: str = "cpu"

class GRUExpert(nn.Module):
    def __init__(self, cfg: WMConfigMoE3):
        super().__init__()
        self.inp = nn.Linear(cfg.x_dim + cfg.u_dim, cfg.mlp_hidden)
        self.rnn = nn.GRU(cfg.mlp_hidden, cfg.gru_hidden, num_layers=cfg.gru_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.mlp_hidden),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden, 12)  # Δ=[dp(3), dv(3), dw(3), dθ(3)]
        )

    def forward(self, x, u, h0=None):
        z = torch.cat([x, u], dim=-1)
        y, h = self.rnn(self.inp(z), h0)
        delta = self.head(y)
        return delta, h

class GateNet(nn.Module):
    def __init__(self, cfg: WMConfigMoE3):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=cfg.x_dim + cfg.u_dim,
            hidden_size=cfg.gate_gru_hidden,
            num_layers=cfg.gate_gru_layers,
            batch_first=True,
            dropout=cfg.gate_gru_dropout if cfg.gate_gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.gate_gru_hidden, 128),
            nn.SiLU(),
            nn.Linear(128, cfg.n_experts),
        )
        self.tau = cfg.gumbel_tau
        self.train_noisy = cfg.gate_train_noisy
        self.use_hard = cfg.use_hard_routing

    def forward(self, x, u, train=True):
        z = torch.cat([x, u], dim=-1)
        y, _ = self.rnn(z)
        logits = self.head(y)
        if train:
            w = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1) if self.train_noisy else F.softmax(logits, dim=-1)
        else:
            w = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1) if self.use_hard else F.softmax(logits, dim=-1)
        return logits, w

class FDIHead(nn.Module):
    def __init__(self, cfg: WMConfigMoE3):
        super().__init__();
        self.rnn = nn.GRU(cfg.x_dim + cfg.u_dim, 64, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, cfg.u_dim))

    def forward(self, x, u):
        z = torch.cat([x, u], dim=-1)
        y, _ = self.rnn(z)
        return torch.sigmoid(self.mlp(y))

class MoEWorldModel3(nn.Module):
    def __init__(self, cfg: WMConfigMoE3):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList([GRUExpert(cfg) for _ in range(cfg.n_experts)])
        self.gate = GateNet(cfg)
        self.fdi  = FDIHead(cfg)

    @staticmethod
    def split_delta(delta):
        return delta[..., :3], delta[..., 3:6], delta[..., 6:9], delta[..., 9:12]

    def compose_next(self, x_t, delta):
        p, q, v, w = x_t[..., 0:3], x_t[..., 3:7], x_t[..., 7:10], x_t[..., 10:13]
        dp, dv, dw, dth = self.split_delta(delta)
        dq = so3_exp(dth)
        q_next = quat_mul(dq, quat_normalize(q))
        return torch.cat([p + dp, quat_normalize(q_next), v + dv, w + dw], dim=-1)

    def forward_moe(self, x, u, train=True):
        logits, w = self.gate(x, u, train=train)
        deltas, hs = [], []
        for exp in self.experts:
            d, h = exp(x, u, None)
            deltas.append(d); hs.append(h)
        D = torch.stack(deltas, dim=-2)  # (B,T,K,12)
        return {"deltas": D, "w": w, "logits": logits, "h": hs}

    def forward(self, x, u, h0=None):
        out = self.forward_moe(x, u, train=False)
        D, w = out["deltas"], out["w"]
        if not self.cfg.use_hard_routing:
            mu = (D * w.unsqueeze(-1)).sum(dim=-2)
        else:
            mu = D[
                torch.arange(D.size(0))[:, None],
                torch.arange(D.size(1))[None, :],
                w.argmax(dim=-1)
            ]
        logvar = torch.zeros_like(mu)
        return {"mu": mu, "logvar": logvar, "h": None}

@torch.no_grad()
def rollout(model: MoEWorldModel3, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
    """Autoregressive rollout using the MoE routing."""
    assert x0.dim() == 2
    B, T, _ = u_seq.shape
    x_hat = torch.empty(B, T + 1, model.cfg.x_dim, device=x0.device, dtype=x0.dtype)
    x_hat[:, 0] = x0
    x_t = x0
    for t in range(T):
        out = model.forward_moe(x_t.unsqueeze(1), u_seq[:, t:t + 1], train=False)
        D, w = out["deltas"][:, 0], out["w"][:, 0]
        if not model.cfg.use_hard_routing:
            delta = (D * w.unsqueeze(-1)).sum(dim=-2)
        else:
            delta = D[torch.arange(B, device=x0.device), w.argmax(dim=-1)]
        x_t = model.compose_next(x_t, delta)
        x_hat[:, t + 1] = x_t
    return x_hat