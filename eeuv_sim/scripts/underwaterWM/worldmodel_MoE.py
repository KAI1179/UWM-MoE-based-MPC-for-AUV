# ===== worldmodel_MoE.py (REPLACEMENT) =====
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    quat_normalize, quat_mul, so3_exp,
    build_delta_targets_seq
)

@dataclass
class WMConfigMoE:
    # 基本尺寸
    x_dim: int = 13
    u_dim: int = 8
    n_experts: int = 4
    gru_hidden: int = 128
    gru_layers: int = 2
    mlp_hidden: int = 256

    # >>> 新增：门控（工况预测）GRU参数 <<<
    gate_gru_hidden: int = 128
    gate_gru_layers: int = 1
    gate_gru_dropout: float = 0.0  # 当 layers>1 时才会生效

    # rollout & loss
    rollout_horizon: int = 20
    huber_delta: float = 0.1

    # 门控正则
    gate_entropy_beta: float = 1e-2
    gate_tv_gamma: float = 5e-2
    gate_lb_alpha: float = 5e-2

    # Gumbel-Softmax
    use_hard_routing: bool = False
    gumbel_tau: float = 0.6
    gate_train_noisy: bool = True  # 训练期使用带噪软路由

    # FDI 分支与执行器一致性
    fdi_bce_alpha: float = 1.0
    fdi_tv_alpha: float = 1e-2
    fdi_sparse_alpha: float = 1e-3
    act_consist_alpha: float = 1e-2
    sat_weight: float = 0.3
    umax: float = 20.0
    du_max: float = 0.0

    # 工况轻监督
    reg_ce_alpha: float = 2e-2

    # 设备
    device: str = "cpu"

class GRUExpert(nn.Module):
    def __init__(self, cfg: WMConfigMoE):
        super().__init__()
        self.inp = nn.Linear(cfg.x_dim + cfg.u_dim, cfg.mlp_hidden)
        self.rnn = nn.GRU(cfg.mlp_hidden, cfg.gru_hidden, num_layers=cfg.gru_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.gru_hidden, cfg.mlp_hidden),
            nn.SiLU(),
            nn.Linear(cfg.mlp_hidden, 12),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([x, u], dim=-1)
        z = self.inp(z)
        y, h = self.rnn(z, h0)
        delta = self.head(y)
        return delta, h

class GateNet(nn.Module):
    """
    序列门控网络（工况预测）：
      - 先将 (x,u) 按时间拼接得到 z∈R^{B×T×(x_dim+u_dim)}
      - 过 GRU 提取时序上下文
      - MLP 映射到每步 K 个专家的 logits
      - 训练期：带噪 Gumbel-Softmax（hard=False）
      - 推理期：softmax 或 Gumbel-Softmax(hard=True) 由 cfg.use_hard_routing 控制
    """
    def __init__(self, cfg):
        super().__init__()
        in_dim = cfg.x_dim + cfg.u_dim
        self.rnn = nn.GRU(
            input_size=in_dim,
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

    def forward(self, x: torch.Tensor, u: torch.Tensor, train: bool = True):
        # x: (B,T,13), u: (B,T,8)  ->  z: (B,T,21)
        z = torch.cat([x, u], dim=-1)
        y, _ = self.rnn(z)              # (B,T,H)
        logits = self.head(y)           # (B,T,K)

        if train:
            if self.train_noisy:
                w = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)  # 带噪软路由
            else:
                w = F.softmax(logits, dim=-1)
        else:
            if self.use_hard:
                w = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)   # top-1 硬路由
            else:
                w = F.softmax(logits, dim=-1)

        return logits, w

class FDIHead(nn.Module):
    def __init__(self, cfg: WMConfigMoE):
        super().__init__()
        self.rnn = nn.GRU(cfg.x_dim + cfg.u_dim, 64, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, cfg.u_dim)
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, u], dim=-1)
        y, _ = self.rnn(z)
        m_logits = self.head(y)
        return torch.sigmoid(m_logits)

class MoEWorldModel(nn.Module):
    def __init__(self, cfg: WMConfigMoE):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList([GRUExpert(cfg) for _ in range(cfg.n_experts)])
        self.gate = GateNet(cfg)
        self.fdi = FDIHead(cfg)

    @staticmethod
    def split(delta: torch.Tensor):
        return delta[..., :3], delta[..., 3:6], delta[..., 6:9], delta[..., 9:12]

    def compose_next(self, x_t: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        p, q, v, w = x_t[..., 0:3], x_t[..., 3:7], x_t[..., 7:10], x_t[..., 10:13]
        dp, dtheta, dv, dw = self.split(delta)
        dq = so3_exp(dtheta)
        q_next = quat_mul(dq, quat_normalize(q))
        return torch.cat([p + dp, quat_normalize(q_next), v + dv, w + dw], dim=-1)

    def forward(self, x: torch.Tensor, u: torch.Tensor, h_states: Optional[List[torch.Tensor]] = None, train: bool = True):
        logits, w = self.gate(x, u, train=train)
        deltas = []
        new_h = []
        for k, exp in enumerate(self.experts):
            h0 = None if h_states is None else h_states[k]
            d_k, h_k = exp(x, u, h0)
            deltas.append(d_k)
            new_h.append(h_k)
        D = torch.stack(deltas, dim=-2)
        return {"deltas": D, "w": w, "logits": logits, "h": new_h}

    def huber(self, x: torch.Tensor, delta: float) -> torch.Tensor:
        return F.huber_loss(x, torch.zeros_like(x), delta=delta, reduction='none')

    @staticmethod
    def tv_1d(x: torch.Tensor) -> torch.Tensor:
        if x.size(1) <= 1: return x.new_tensor(0.0)
        return (x[:, 1:] - x[:, :-1]).abs().mean()

    def single_step_loss(self, x: torch.Tensor, u: torch.Tensor, x_next: torch.Tensor,
                         mask: Optional[torch.Tensor] = None,
                         regime_idx: Optional[torch.Tensor] = None,
                         regime_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        out = self.forward(x, u, train=True)
        D, w, logits = out["deltas"], out["w"], out["logits"]

        delta_gt = build_delta_targets_seq(x, x_next)  # (B,T,12)
        res = delta_gt.unsqueeze(-2) - D               # (B,T,K,12)
        hub = self.huber(res, self.cfg.huber_delta).sum(dim=-1)  # (B,T,K)

        step_loss = (hub * w).sum(dim=-1)              # (B,T)
        if mask is not None:
            step_loss = (step_loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            step_loss = step_loss.mean()

        # 门控正则
        ent_loss = (-(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(dim=-1)).mean()
        tv_loss  = self.tv_1d(w)

        loss = step_loss + self.cfg.gate_entropy_beta * ent_loss + self.cfg.gate_tv_gamma * tv_loss

        # 负载均衡 KL
        w_mean = w.mean(dim=(0,1))
        uniform = torch.full_like(w_mean, 1.0 / w_mean.numel())
        lb_loss = F.kl_div((w_mean + 1e-8).log(), uniform, reduction="batchmean")
        loss = loss + self.cfg.gate_lb_alpha * lb_loss

        # 工况轻监督（带逐步权重）
        if (regime_idx is not None) and (self.cfg.reg_ce_alpha > 0.0):
            B, T, K = logits.shape
            ce_per = F.cross_entropy(logits.reshape(B*T, K), regime_idx.reshape(B*T), reduction="none")
            if regime_weight is not None:
                rw = regime_weight.reshape(B*T).to(ce_per.dtype)
                rw_sum = rw.sum().clamp(min=1.0)
                ce = (ce_per * rw).sum() / rw_sum
            else:
                ce = ce_per.mean()
            loss = loss + self.cfg.reg_ce_alpha * ce
        else:
            ce = torch.tensor(0.0, device=x.device)

        stats = {
            "step": step_loss.detach(),
            "ent":  ent_loss.detach(),
            "tv":   tv_loss.detach(),
            "lb":   lb_loss.detach(),
            "reg_ce": ce.detach(),
        }
        return loss, stats

    @torch.no_grad()
    def _choose_delta(self, x_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        out = self.forward(x_t.unsqueeze(1), u_t.unsqueeze(1), train=False)
        D = out["deltas"][:, 0]
        w = out["w"][:, 0]
        if self.cfg.use_hard_routing:
            idx = w.argmax(dim=-1)
            delta = D[torch.arange(D.size(0), device=D.device), idx]
        else:
            delta = (D * w.unsqueeze(-1)).sum(dim=-2)
        return delta

    def rollout_loss(self, x: torch.Tensor, u: torch.Tensor, x_next: torch.Tensor,
                     mask: Optional[torch.Tensor] = None, sched_samp_p: float = 0.0) -> Tuple[torch.Tensor, Dict]:
        B, T, _ = x.shape
        H = min(self.cfg.rollout_horizon, T - 1)

        x_hat = x[:, 0]
        total = 0.0
        n_terms = 0

        for t in range(H):
            u_t = u[:, t]
            delta_t = self._choose_delta(x_hat, u_t)
            x_hat = self.compose_next(x_hat, delta_t)

            x_gt = x[:, t + 1]
            if sched_samp_p > 0.0:
                use_gt = (torch.rand(B, device=x.device) < sched_samp_p).float().unsqueeze(-1)
                x_hat = use_gt * x_gt + (1.0 - use_gt) * x_hat

            p_err = self.huber(x_hat[:, 0:3] - x_gt[:, 0:3], self.cfg.huber_delta)
            v_err = self.huber(x_hat[:, 7:10] - x_gt[:, 7:10], self.cfg.huber_delta)
            w_err = self.huber(x_hat[:, 10:13] - x_gt[:, 10:13], self.cfg.huber_delta)
            q_hat = quat_normalize(x_hat[:, 3:7])
            q_gt  = quat_normalize(x_gt[:, 3:7])
            q_err = 1.0 - (q_hat * q_gt).sum(dim=-1).pow(2.0)
            q_err = q_err.unsqueeze(-1)

            step_err = (p_err.sum(dim=-1, keepdim=True) +
                        3.0 * q_err +
                        v_err.sum(dim=-1, keepdim=True) +
                        w_err.sum(dim=-1, keepdim=True))

            if mask is not None:
                m = mask[:, t+1].unsqueeze(-1)
                total = total + (step_err * m).sum() / m.sum().clamp(min=1)
            else:
                total = total + step_err.mean()
            n_terms += 1

        loss = total / max(1, n_terms)
        return loss, {"roll": loss.detach()}

    def fdi_forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.fdi(x, u)

    @staticmethod
    def actuator_slew_sat(u_cmd: torch.Tensor, umax: float, du_max: float) -> torch.Tensor:
        u = u_cmd.clone()
        B, T, M = u.shape
        u = u.clamp(min=-umax, max=umax)
        if du_max is None or du_max <= 0:
            return u.clamp(-umax, umax)
        out = torch.empty_like(u)
        out[:, 0] = u[:, 0].clamp(-umax, umax)
        for t in range(1, T):
            du = (u[:, t] - out[:, t-1]).clamp(min=-du_max, max=du_max)
            out[:, t] = (out[:, t-1] + du).clamp(-umax, umax)
        return out

    def fdi_actuator_losses(self, x: torch.Tensor, u_in: torch.Tensor,
                            health_gt: torch.Tensor, is_saturated: torch.Tensor,
                            u_cmd_raw: torch.Tensor, u_app_raw: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        cfg = self.cfg
        m_hat = self.fdi_forward(x, u_in)

        bce = F.binary_cross_entropy(m_hat, health_gt, reduction="mean") if cfg.fdi_bce_alpha > 0 else x.new_tensor(0.0)
        tv = self.tv_1d(m_hat) if cfg.fdi_tv_alpha > 0 else x.new_tensor(0.0)
        sparse = (1.0 - m_hat).mean() if cfg.fdi_sparse_alpha > 0 else x.new_tensor(0.0)

        if cfg.act_consist_alpha > 0:
            with torch.no_grad():
                u_cmd_sat = self.actuator_slew_sat(u_cmd_raw, cfg.umax, cfg.du_max)
            u_pred_app = u_cmd_sat * m_hat
            w = (1.0 - is_saturated) + cfg.sat_weight * is_saturated
            u_mse = ((u_pred_app - u_app_raw) ** 2 * w).mean()
        else:
            u_mse = x.new_tensor(0.0)

        loss = cfg.fdi_bce_alpha * bce + cfg.fdi_tv_alpha * tv + cfg.fdi_sparse_alpha * sparse + cfg.act_consist_alpha * u_mse
        stats = {"fdi_bce": bce.detach(), "fdi_tv": tv.detach(), "fdi_sparse": sparse.detach(), "u_cons": u_mse.detach()}
        return loss, stats

def train_one_epoch_moe(model: MoEWorldModel, optimizer, loader, cfg: WMConfigMoE,
                        sched_samp_max: float = 0.5, epoch: int = 0, max_epochs: int = 100):
    model.train()
    agg = {k: 0.0 for k in ["step", "roll", "ent", "tv", "lb", "reg_ce", "fdi_bce", "fdi_tv", "fdi_sparse", "u_cons"]}
    ss_p = min(sched_samp_max, sched_samp_max * (epoch / max(1, (max_epochs - 1)))) if sched_samp_max > 0 else 0.0

    for batch in loader:
        x      = batch["x"].to(cfg.device, dtype=torch.float32)
        u      = batch["u"].to(cfg.device, dtype=torch.float32)
        x_next = batch["x_next"].to(cfg.device, dtype=torch.float32)

        regime_idx     = batch.get("regime_idx", None)
        if regime_idx is not None: regime_idx = regime_idx.to(cfg.device)
        regime_weight  = batch.get("regime_weight", None)
        if regime_weight is not None: regime_weight = regime_weight.to(cfg.device, dtype=torch.float32)

        health_gt      = batch.get("health_gt", None)
        if health_gt is not None:  health_gt = health_gt.to(cfg.device, dtype=torch.float32)
        is_saturated   = batch.get("is_saturated", None)
        if is_saturated is not None: is_saturated = is_saturated.to(cfg.device, dtype=torch.float32)
        u_cmd_raw      = batch.get("u_cmd_raw", None)
        if u_cmd_raw is not None:    u_cmd_raw = u_cmd_raw.to(cfg.device, dtype=torch.float32)
        u_app_raw      = batch.get("u_app_raw", None)
        if u_app_raw is not None:    u_app_raw = u_app_raw.to(cfg.device, dtype=torch.float32)

        optimizer.zero_grad()

        step_loss, stats_s = model.single_step_loss(x, u, x_next, mask=None,
                                                    regime_idx=regime_idx,
                                                    regime_weight=regime_weight)

        fdi_loss, stats_fdi = model.fdi_actuator_losses(
            x, u,
            health_gt=health_gt if health_gt is not None else torch.ones_like(u),
            is_saturated=is_saturated if is_saturated is not None else torch.zeros_like(u),
            u_cmd_raw=u_cmd_raw if u_cmd_raw is not None else torch.zeros_like(u),
            u_app_raw=u_app_raw if u_app_raw is not None else torch.zeros_like(u)
        )

        roll_loss, stats_r = model.rollout_loss(x, u, x_next, mask=None, sched_samp_p=ss_p)

        loss = step_loss + roll_loss + fdi_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        agg["step"]     += stats_s["step"].item()
        agg["roll"]     += stats_r["roll"].item()
        agg["ent"]      += stats_s["ent"].item()
        agg["tv"]       += stats_s["tv"].item()
        agg["lb"]       += stats_s["lb"].item()
        agg["reg_ce"]   += stats_s["reg_ce"].item()
        agg["fdi_bce"]  += stats_fdi["fdi_bce"].item()
        agg["fdi_tv"]   += stats_fdi["fdi_tv"].item()
        agg["fdi_sparse"] += stats_fdi["fdi_sparse"].item()
        agg["u_cons"]   += stats_fdi["u_cons"].item()

    n_batches = max(1, len(loader))
    return {
        "step": agg["step"] / n_batches,
        "roll": agg["roll"] / n_batches,
        "ent":  agg["ent"]  / n_batches,
        "tv":   agg["tv"]   / n_batches,
        "lb":   agg["lb"]   / n_batches,
        "reg_ce": agg["reg_ce"] / n_batches,
        "fdi_bce": agg["fdi_bce"] / n_batches,
        "fdi_tv":  agg["fdi_tv"]  / n_batches,
        "fdi_sparse": agg["fdi_sparse"] / n_batches,
        "u_cons": agg["u_cons"] / n_batches,
        "ss_p": ss_p,
    }

@torch.no_grad()
def rollout(model: MoEWorldModel, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
    assert x0.dim() == 2
    B, T, _ = u_seq.shape
    device = x0.device
    x_hat = torch.empty(B, T + 1, model.cfg.x_dim, device=device, dtype=x0.dtype)
    x_hat[:, 0] = x0
    x_t = x0
    for t in range(T):
        u_t = u_seq[:, t]
        delta_t = model._choose_delta(x_t, u_t)
        x_t = model.compose_next(x_t, delta_t)
        x_hat[:, t+1] = x_t
    return x_hat
# ===== END =====
