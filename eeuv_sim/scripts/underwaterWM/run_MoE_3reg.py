
"""
训练脚本：保持三文件结构。
- 模型结构定义放在 worldmodel_MoE_3reg.py
- 本文件包含数据读取、标准化、损失与训练循环
"""
import argparse, os, numpy as np, torch, h5py, random
from torch.utils.data import Dataset, DataLoader, BatchSampler
from worldmodel_MoE_3reg import WMConfigMoE3, MoEWorldModel3
from utils import set_seed, quat_normalize, build_delta_targets_seq  # 项目内工具

# --------------------------- 数据集 & 标准化 ---------------------------
class SimpleStandardizer:
    def __init__(self):
        self.mean=None; self.std=None; self.u_mean=None; self.u_std=None
    def fit_states(self, X: torch.Tensor):
        mean = X.mean(0); std = X.std(0).clamp(min=1e-6)
        mean[3:7] = 0.0; std[3:7] = 1.0  # 四元数不缩放
        self.mean = mean.float(); self.std = std.float()
    def transform_states(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std
    def fit_actions(self, U: torch.Tensor):
        self.u_mean = U.mean(0).float()
        self.u_std = U.std(0).clamp(min=1e-6).float()
    def transform_actions(self, U: torch.Tensor) -> torch.Tensor:
        return (U - self.u_mean) / self.u_std
    def save(self, path: str):
        import numpy as np, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean.cpu().numpy(), std=self.std.cpu().numpy(),
                 u_mean=self.u_mean.cpu().numpy(), u_std=self.u_std.cpu().numpy())

class H5SeqDataset(Dataset):
    """读取HDF5序列，只保留 regime_id ∈ {1,2,3}。"""
    def __init__(self, file_path, seq_len=64, stride=32, use_standardizer=True):
        self.fp=file_path; self.seq_len=seq_len; self.stride=stride; self.use_std=use_standardizer
        self.episodes=[]; self.keys=[]; self.ep_meta={}
        with h5py.File(self.fp,"r") as f:
            all_eps=sorted([k for k in f.keys() if k.startswith("episode_")], key=lambda s:int(s.split("_")[1]))
            kept=[]
            for k in all_eps:
                reg=int(f[k]["regime_id"][0]) if len(f[k]["regime_id"])>0 else -1
                if reg in (1,2,3): kept.append(k)
            self.episodes=kept
            for e,k in enumerate(self.episodes):
                N=len(f[k]["time"]); reg=int(f[k]["regime_id"][0]) if len(f[k]["regime_id"])>0 else -1
                self.ep_meta[k]=reg
                if N<seq_len+1: continue
                for s in range(0,N-seq_len,stride): self.keys.append((e,s))
        self.std=None
        if self.use_std:
            xs,us=[],[]
            with h5py.File(self.fp,"r") as f:
                for k in self.episodes:
                    x_all=np.concatenate([f[k]["position"][()],f[k]["orientation"][()],f[k]["linear_velocity"][()],f[k]["angular_velocity"][()]],axis=1).astype(np.float32)
                    u_all=f[k]["thrusts_applied"][()].astype(np.float32); xs.append(x_all); us.append(u_all)
            X=torch.tensor(np.concatenate(xs,0),dtype=torch.float32); U=torch.tensor(np.concatenate(us,0),dtype=torch.float32)
            self.std=SimpleStandardizer(); self.std.fit_states(X); self.std.fit_actions(U)
    def __len__(self): return len(self.keys)
    def __getitem__(self, idx):
        ep_idx,s=self.keys[idx]
        with h5py.File(self.fp,"r") as f:
            k=self.episodes[ep_idx]
            pos=f[k]["position"][s:s+self.seq_len+1].astype(np.float32)
            ori=f[k]["orientation"][s:s+self.seq_len+1].astype(np.float32)
            lin=f[k]["linear_velocity"][s:s+self.seq_len+1].astype(np.float32)
            ang=f[k]["angular_velocity"][s:s+self.seq_len+1].astype(np.float32)
            x=np.concatenate([pos,ori,lin,ang],axis=1).astype(np.float32)
            u_app=f[k]["thrusts_applied"][s:s+self.seq_len].astype(np.float32)
            u_cmd=f[k]["thrusts_cmd"][s:s+self.seq_len].astype(np.float32)
            health=is_sat=regime=None
            if "labels" in f[k]:
                lab=f[k]["labels"]
                health=lab.get("health_mask_gt_hys",None)
                is_sat=lab.get("is_saturated",None)
                regime=lab.get("regime_step",None)
            T=self.seq_len
            health_seq=(health[s:s+T].astype(np.float32) if health is not None else np.ones((T,8),np.float32))
            is_sat_seq=(is_sat[s:s+T].astype(np.float32) if is_sat is not None else np.zeros((T,8),np.float32))
            if regime is None:
                reg_val=int(f[k]["regime_id"][0]) if len(f[k]["regime_id"])>0 else 1
                reg_seq=np.full((T,), reg_val, dtype=np.int64)
            else:
                reg_seq=regime[s:s+T].astype(np.int64)
            regime_idx=np.clip(reg_seq,1,3)-1
            x_t,x_t1=x[:-1],x[1:]
            if self.std is not None:
                x_t=torch.from_numpy(x_t); x_t1=torch.from_numpy(x_t1); u_app_t=torch.from_numpy(u_app)
                x_t=self.std.transform_states(x_t).numpy()
                x_t1=self.std.transform_states(x_t1).numpy()
                u_app_n=self.std.transform_actions(u_app_t).numpy()
            else:
                u_app_n=u_app
            return {"x":torch.from_numpy(x_t).float(),"x_next":torch.from_numpy(x_t1).float(),"u":torch.from_numpy(u_app_n).float(),
                    "health_gt":torch.from_numpy(health_seq).float(),"is_saturated":torch.from_numpy(is_sat_seq).float(),
                    "regime_idx":torch.from_numpy(regime_idx).long(),
                    "u_cmd_raw":torch.from_numpy(u_cmd).float(),"u_app_raw":torch.from_numpy(u_app).float()}

class BalancedBatchSampler(BatchSampler):
    """每个batch尽量包含reg1/2/3样本（若可）。"""
    def __init__(self, dataset:H5SeqDataset, batch_size:int, seed:int=42):
        self.dataset=dataset; self.batch_size=int(batch_size); self.rng=random.Random(seed)
        self.group_to_indices={"reg1":[], "reg2":[], "reg3":[]}
        for i in range(len(dataset)):
            ep_idx,_=dataset.keys[i]; ep_name=dataset.episodes[ep_idx]; reg=dataset.ep_meta[ep_name]
            g=f"reg{reg}" if reg in (1,2,3) else None
            if g: self.group_to_indices[g].append(i)
        self.all_indices=list(range(len(dataset))); self._shuffle_pool()
        self.num_batches=(len(self.all_indices)+self.batch_size-1)//self.batch_size
    def _shuffle_pool(self): self.rng.shuffle(self.all_indices); self.pool_ptr=0
    def __len__(self): return self.num_batches
    def __iter__(self):
        self._shuffle_pool()
        for _ in range(self.num_batches):
            batch=[]
            for g in ["reg1","reg2","reg3"]:
                if self.group_to_indices[g]: batch.append(self.rng.choice(self.group_to_indices[g]))
            batch=list(dict.fromkeys(batch))
            need=self.batch_size-len(batch)
            while need>0:
                if self.pool_ptr>=len(self.all_indices): self._shuffle_pool()
                idx=self.all_indices[self.pool_ptr]; self.pool_ptr+=1
                if idx in batch: continue
                batch.append(idx); need-=1
            yield batch

def collate_pad(batch_list):
    out={}
    for key in batch_list[0].keys():
        out[key]=torch.stack([b[key] for b in batch_list], dim=0)
    return out

# --------------------------- 训练损失/工具 ---------------------------
import torch.nn.functional as F

def huber(x, delta=0.1):
    return F.huber_loss(x, torch.zeros_like(x), delta=delta, reduction='none')

def tv_1d(x: torch.Tensor):
    if x.size(1) <= 1: return x.new_tensor(0.0)
    return (x[:, 1:] - x[:, :-1]).abs().mean()

def single_step_loss(model: MoEWorldModel3, x, u, x_next, regime_idx=None):
    out = model.forward_moe(x, u, train=True)
    D, w, logits = out["deltas"], out["w"], out["logits"]

    delta_gt = build_delta_targets_seq(x, x_next)  # 来自 utils，与训练时保持一致
    hub = huber(delta_gt.unsqueeze(-2) - D, delta=model.cfg.huber_delta).sum(dim=-1)  # (B,T,K)
    step_loss = (hub * w).sum(dim=-1).mean()

    # 门控正则（与原训练实现一致）
    ent_loss = (-(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(dim=-1)).mean()
    def _tv(x):  # allow (B,T,K)
        if x.dim() == 3: return tv_1d(x.mean(-1, keepdim=False))
        return tv_1d(x)
    tv_loss = _tv(w)
    w_mean = w.mean(dim=(0, 1))
    uniform = torch.full_like(w_mean, 1.0 / w_mean.numel())
    lb_loss = F.kl_div((w_mean + 1e-8).log(), uniform, reduction="batchmean")

    loss = step_loss + model.cfg.gate_entropy_beta * ent_loss + model.cfg.gate_tv_gamma * tv_loss + model.cfg.gate_lb_alpha * lb_loss

    if (regime_idx is not None) and (model.cfg.reg_ce_alpha > 0):
        B, T, K = logits.shape
        loss = loss + model.cfg.reg_ce_alpha * F.cross_entropy(logits.reshape(B * T, K), regime_idx.reshape(B * T), reduction="mean")

    return loss, {"step": step_loss.detach(), "ent": ent_loss.detach(), "tv": tv_loss.detach(),
                  "lb": lb_loss.detach(), "reg_ce": torch.tensor(0.0, device=x.device)}

def rollout_loss(model: MoEWorldModel3, x, u, x_next, sched_samp_p: float = 0.0):
    B, T, _ = x.shape
    H = min(model.cfg.rollout_horizon, T - 1)
    x_hat = x[:, 0]
    total = 0.0
    n = 0
    for t in range(H):
        out = model.forward_moe(x_hat.unsqueeze(1), u[:, t:t+1], train=False)
        D, w = out["deltas"][:, 0], out["w"][:, 0]
        delta_t = (D * w.unsqueeze(-1)).sum(dim=-2) if (not model.cfg.use_hard_routing) else D[torch.arange(B, device=x.device), w.argmax(dim=-1)]
        x_hat = model.compose_next(x_hat, delta_t)
        x_gt = x[:, t + 1]
        if sched_samp_p > 0.0:
            use_gt = (torch.rand(B, device=x.device) < sched_samp_p).float().unsqueeze(-1)
            x_hat = use_gt * x_gt + (1.0 - use_gt) * x_hat

        p_err = huber(x_hat[:, 0:3] - x_gt[:, 0:3], model.cfg.huber_delta)
        v_err = huber(x_hat[:, 7:10] - x_gt[:, 7:10], model.cfg.huber_delta)
        w_err = huber(x_hat[:, 10:13] - x_gt[:, 10:13], model.cfg.huber_delta)
        q_hat, q_gt = quat_normalize(x_hat[:, 3:7]), quat_normalize(x_gt[:, 3:7])
        q_err = 1.0 - (q_hat * q_gt).sum(dim=-1).pow(2.0)
        total += (p_err.sum(-1, keepdim=True) + 3.0 * q_err.unsqueeze(-1) + v_err.sum(-1, keepdim=True) + w_err.sum(-1, keepdim=True)).mean()
        n += 1
    loss = total / max(1, n)
    return loss, {"roll": loss.detach()}

def actuator_slew_sat(u_cmd, umax, du_max):
    u = u_cmd.clone().clamp(-umax, umax)
    B, T, M = u.shape
    if (du_max is None) or (du_max <= 0): return u
    out = torch.empty_like(u); out[:, 0] = u[:, 0]
    for t in range(1, T):
        du = (u[:, t] - out[:, t - 1]).clamp(-du_max, du_max)
        out[:, t] = (out[:, t - 1] + du).clamp(-umax, umax)
    return out

def fdi_actuator_losses(model: MoEWorldModel3, x, u_in, health_gt, is_saturated, u_cmd_raw, u_app_raw):
    m_hat = model.fdi(x, u_in)
    bce = F.binary_cross_entropy(m_hat, health_gt, reduction="mean") if model.cfg.fdi_bce_alpha > 0 else x.new_tensor(0.0)
    tv  = tv_1d(m_hat) if model.cfg.fdi_tv_alpha > 0 else x.new_tensor(0.0)
    sparse = (1.0 - m_hat).mean() if model.cfg.fdi_sparse_alpha > 0 else x.new_tensor(0.0)
    if model.cfg.act_consist_alpha > 0:
        with torch.no_grad():
            u_cmd_sat = actuator_slew_sat(u_cmd_raw, model.cfg.umax, model.cfg.du_max)
        u_pred_app = u_cmd_sat * m_hat
        w = (1.0 - is_saturated) + model.cfg.sat_weight * is_saturated
        u_mse = ((u_pred_app - u_app_raw) ** 2 * w).mean()
    else:
        u_mse = x.new_tensor(0.0)
    loss = (model.cfg.fdi_bce_alpha * bce +
            model.cfg.fdi_tv_alpha * tv +
            model.cfg.fdi_sparse_alpha * sparse +
            model.cfg.act_consist_alpha * u_mse)
    return loss, {"fdi_bce": bce.detach(), "fdi_tv": tv.detach(), "fdi_sparse": sparse.detach(), "u_cons": u_mse.detach()}

def train_one_epoch(model: MoEWorldModel3, optimizer, loader, cfg: WMConfigMoE3, sched_samp_max=0.8, epoch=0, max_epochs=100):
    model.train()
    agg = {k: 0.0 for k in ["step","roll","ent","tv","lb","reg_ce","fdi_bce","fdi_tv","fdi_sparse","u_cons"]}
    ss_p = min(sched_samp_max, sched_samp_max * (epoch / max(1, (max_epochs - 1)))) if sched_samp_max > 0 else 0.0

    for batch in loader:
        x = batch["x"].to(cfg.device); u = batch["u"].to(cfg.device); x_next = batch["x_next"].to(cfg.device)
        regime_idx = batch.get("regime_idx", None); regime_idx = (regime_idx.to(cfg.device) if regime_idx is not None else None)
        health_gt = (batch.get("health_gt").to(cfg.device) if batch.get("health_gt") is not None else torch.ones_like(u))
        is_saturated = (batch.get("is_saturated").to(cfg.device) if batch.get("is_saturated") is not None else torch.zeros_like(u))
        u_cmd_raw = (batch.get("u_cmd_raw").to(cfg.device) if batch.get("u_cmd_raw") is not None else torch.zeros_like(u))
        u_app_raw = (batch.get("u_app_raw").to(cfg.device) if batch.get("u_app_raw") is not None else torch.zeros_like(u))

        optimizer.zero_grad()
        s_loss, s_stats = single_step_loss(model, x, u, x_next, regime_idx=regime_idx)
        f_loss, f_stats = fdi_actuator_losses(model, x, u, health_gt, is_saturated, u_cmd_raw, u_app_raw)
        r_loss, r_stats = rollout_loss(model, x, u, x_next, sched_samp_p=ss_p)
        loss = s_loss + f_loss + r_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        agg["step"] += float(s_stats["step"]); agg["roll"] += float(r_stats["roll"]); agg["ent"] += float(s_stats["ent"])
        agg["tv"]   += float(s_stats["tv"]);   agg["lb"]   += float(s_stats["lb"]);   agg["reg_ce"] += float(s_stats["reg_ce"])
        agg["fdi_bce"] += float(f_stats["fdi_bce"]); agg["fdi_tv"] += float(f_stats["fdi_tv"]); agg["fdi_sparse"] += float(f_stats["fdi_sparse"])
        agg["u_cons"] += float(f_stats["u_cons"])

    n_batches = max(1, len(loader))
    return {k: v / n_batches for k, v in agg.items()} | {"ss_p": ss_p}

# --------------------------- 训练入口 ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file_path", type=str, default="/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_10_labeled.hdf5")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--rollout_h", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./checkpoints_moe3_3")
    p.add_argument("--std_path", type=str, default=None, help="保存标准化文件（npz）。默认 <save_dir>/std_stats.npz 与 std_state.npz")
    p.add_argument("--no_standardize", action="store_true")
    # Gate regularizers / routing
    p.add_argument("--gate_ent", type=float, default=1e-2)
    p.add_argument("--gate_tv", type=float, default=2e-2)
    p.add_argument("--gate_lb", type=float, default=5e-2)
    p.add_argument("--use_hard", action="store_true", help="推理时使用hard Gumbel路由")
    # FDI & actuator consistency
    p.add_argument("--fdi_bce", type=float, default=0.0)  ## 故障检测隔离 参数 健康标签监督 仿真器中用不到 用的话设置为1.0
    p.add_argument("--fdi_tv", type=float, default=0.0)  ## 约束健康概率随时间平滑，抑制抖动  仿真器中用不到 用的话设置为1e-2
    p.add_argument("--fdi_sparse", type=float, default=0.0)  ## 健康先验 | 稀疏惩罚 仿真器中用不到 用的话设置为1e-3
    p.add_argument("--act_consist", type=float, default=0.0)  ## 执行器一致  仿真器中用不到 用的话设置为1e-2
    p.add_argument("--sat_weight", type=float, default=0.3)
    p.add_argument("--umax", type=float, default=20.0)
    p.add_argument("--du_max", type=float, default=0.0)
    # Regime supervision
    p.add_argument("--reg_ce", type=float, default=1e-1)
    # Scheduled Sampling
    p.add_argument("--ss_max", type=float, default=0.8)
    return p.parse_args()

def main():
    args = parse_args(); set_seed(args.seed); os.makedirs(args.save_dir, exist_ok=True)

    dataset = H5SeqDataset(args.file_path, seq_len=args.seq_len, stride=args.stride, use_standardizer=(not args.no_standardize))

    # 保存标准化（两个常见命名都保存，兼容你的 ckpt 配套文件）
    std_path = args.std_path or os.path.join(args.save_dir, 'std_stats.npz')
    if getattr(dataset, 'std', None) is not None:
        try:
            dataset.std.save(std_path)
            dataset.std.save(os.path.join(args.save_dir, "std_state.npz"))
            dataset.std.save(os.path.join(args.save_dir, "standardizer_moe.npz"))
            print(f"[Info] Saved standardizer to: {std_path} and std_state.npz")
        except Exception as e:
            print(f"[Warn] Failed to save standardizer: {e}")

    loader = DataLoader(dataset, batch_sampler=BalancedBatchSampler(dataset, batch_size=args.batch_size, seed=args.seed),
                        num_workers=0, collate_fn=collate_pad)

    cfg = WMConfigMoE3(
        x_dim=13, u_dim=8, n_experts=3,
        gru_hidden=128, gru_layers=2, mlp_hidden=256,
        rollout_horizon=args.rollout_h,
        gate_entropy_beta=args.gate_ent, gate_tv_gamma=args.gate_tv, gate_lb_alpha=args.gate_lb,
        use_hard_routing=args.use_hard, gumbel_tau=0.6, gate_train_noisy=True,
        fdi_bce_alpha=args.fdi_bce, fdi_tv_alpha=args.fdi_tv, fdi_sparse_alpha=args.fdi_sparse,
        act_consist_alpha=args.act_consist, sat_weight=args.sat_weight, umax=args.umax, du_max=args.du_max,
        reg_ce_alpha=args.reg_ce, device=args.device
    )
    model = MoEWorldModel3(cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda e: min(1.0, (e+1)/5))

    best = 1e9
    for ep in range(1, args.epochs + 1):
        stats = train_one_epoch(model, optim, loader, cfg, sched_samp_max=args.ss_max, epoch=ep - 1, max_epochs=args.epochs)
        scheduler.step()
        print(f"[Epoch {ep:03d}] step:{stats['step']:.4f} roll:{stats['roll']:.4f} ent:{stats['ent']:.4f} tv:{stats['tv']:.4f} "
              f"lb:{stats['lb']:.4f} regCE:{stats['reg_ce']:.4f} FDI[bce:{stats['fdi_bce']:.4f} tv:{stats['fdi_tv']:.4f} "
              f"sp:{stats['fdi_sparse']:.4f}] u_cons:{stats['u_cons']:.4f} ss_p:{stats['ss_p']:.2f}")

        # 简单的checkpoint策略（与原脚本一致风格）
        if ep % 20 == 0:
            ckpt_path = os.path.join(args.save_dir, f"moe3_epoch{ep:03d}.pt")
            torch.save({"epoch": ep, "model_state": model.state_dict(), "optimizer_state": optim.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
            print(f"[Info] checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    main()
