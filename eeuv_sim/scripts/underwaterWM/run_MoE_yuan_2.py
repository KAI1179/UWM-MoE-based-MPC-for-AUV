# run_MoE.py
# 读取 HDF5 -> 制作序列数据集（TBPTT）-> 标准化 -> 训练 MoE 世界模型（含 FDI 与执行器一致性）
import argparse, os, h5py, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

from utils import set_seed  # 复用你项目里的随机种子工具
from worldmodel_MoE import WMConfigMoE, MoEWorldModel, train_one_epoch_moe

# ------------------------------
# 简单标准化器（位置/速度/角速/推力；四元数不标准化）
# ------------------------------
class SimpleStandardizer:
    def __init__(self):
        self.mean = None
        self.std  = None
        self.u_mean = None
        self.u_std  = None

    def fit_states(self, X: torch.Tensor):  # X: (N,13)
        mean = X.mean(0)
        std  = X.std(0).clamp(min=1e-6)
        # 四元数保持单位（不减均值不缩放）
        mean[3:7] = 0.0
        std[3:7]  = 1.0
        self.mean = mean.float()
        self.std  = std.float()

    def transform_states(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def fit_actions(self, U: torch.Tensor):  # U: (N,8) —— 用 applied 拟合统计
        self.u_mean = U.mean(0).float()
        self.u_std  = U.std(0).clamp(min=1e-6).float()

    def transform_actions(self, U: torch.Tensor) -> torch.Tensor:
        return (U - self.u_mean) / self.u_std

    def save(self, path: str):
        import numpy as np, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 mean=self.mean.detach().cpu().numpy(),
                 std=self.std.detach().cpu().numpy(),
                 u_mean=self.u_mean.detach().cpu().numpy(),
                 u_std=self.u_std.detach().cpu().numpy())
# ------------------------------

# HDF5 -> 序列数据集
# ------------------------------
class H5SeqDataset(Dataset):
    def __init__(self, file_path, seq_len=64, stride=32, use_standardizer=True):
        self.fp = file_path
        self.seq_len = seq_len
        self.stride = stride
        self.use_std = use_standardizer

        self.episodes = []
        self.keys = []  # (ep_idx, start)
        with h5py.File(self.fp, "r") as f:
            self.episodes = sorted([k for k in f.keys() if k.startswith("episode_")],
                                   key=lambda s: int(s.split("_")[1]))
            for e, k in enumerate(self.episodes):
                N = len(f[k]["time"])
                if N < seq_len + 1:
                    continue
                for s in range(0, N - seq_len, stride):
                    self.keys.append((e, s))

        # 计算标准化参数（用 u_applied）
        self.std = None
        if self.use_std:
            xs, us = [], []
            with h5py.File(self.fp, "r") as f:
                for k in self.episodes:
                    x_all = np.concatenate([
                        f[k]["position"][()],
                        f[k]["orientation"][()],
                        f[k]["linear_velocity"][()],
                        f[k]["angular_velocity"][()],
                    ], axis=1).astype(np.float32)  # (N,13)
                    u_all = f[k]["thrusts_applied"][()].astype(np.float32)
                    xs.append(x_all)
                    us.append(u_all)
            X = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32)
            U = torch.tensor(np.concatenate(us, axis=0), dtype=torch.float32)
            self.std = SimpleStandardizer()
            self.std.fit_states(X)
            self.std.fit_actions(U)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        ep_idx, s = self.keys[idx]
        with h5py.File(self.fp, "r") as f:
            k = self.episodes[ep_idx]
            # 状态 (T+1,13)
            pos  = f[k]["position"][s:s+self.seq_len+1].astype(np.float32)
            ori  = f[k]["orientation"][s:s+self.seq_len+1].astype(np.float32)
            lin  = f[k]["linear_velocity"][s:s+self.seq_len+1].astype(np.float32)
            ang  = f[k]["angular_velocity"][s:s+self.seq_len+1].astype(np.float32)
            x = np.concatenate([pos, ori, lin, ang], axis=1).astype(np.float32)

            # 动作 (标准化输入用 applied；一致性损失用 raw)
            u_app = f[k]["thrusts_applied"][s:s+self.seq_len].astype(np.float32)  # (T,8)
            u_cmd = f[k]["thrusts_cmd"][s:s+self.seq_len].astype(np.float32)      # (T,8)

            # labels/*
            if "labels" in f[k]:
                lab = f[k]["labels"]
                health = lab.get("health_mask_gt_hys", None)
                is_sat = lab.get("is_saturated", None)
                regime = lab.get("regime_step", None)
                # 可选：du_abs/kappa 如需分析可读出
                du_abs = lab.get("du_abs", None)
                kappa  = lab.get("kappa", None)
            else:
                health = is_sat = regime = du_abs = kappa = None

            # fallback
            T = self.seq_len
            if health is None:
                health_seq = np.ones((T, 8), dtype=np.float32)
            else:
                health_seq = health[s:s+T].astype(np.float32)
            if is_sat is None:
                is_sat_seq = np.zeros((T, 8), dtype=np.float32)
            else:
                is_sat_seq = is_sat[s:s+T].astype(np.float32)
            if regime is None:
                reg_val = int(f[k]["regime_id"][0]) if len(f[k]["regime_id"])>0 else 1
                reg_seq = np.full((T,), reg_val, dtype=np.int64)
            else:
                reg_seq = regime[s:s+T].astype(np.int64)

            regime_idx = np.clip(reg_seq - 1, 0, 3)   # {1..4}→{0..3}

            # 切片 x(t), x(t+1)
            x_t   = x[:-1]  # (T,13)
            x_t1  = x[1:]   # (T,13)

            # 标准化
            if self.std is not None:
                x_t  = self.std.transform_states(torch.from_numpy(x_t)).numpy()
                x_t1 = self.std.transform_states(torch.from_numpy(x_t1)).numpy()
                u_app_n = self.std.transform_actions(torch.from_numpy(u_app)).numpy()
            else:
                u_app_n = u_app

            batch = {
                "x": torch.from_numpy(x_t).float(),         # (T,13)  标准化
                "x_next": torch.from_numpy(x_t1).float(),   # (T,13)
                "u": torch.from_numpy(u_app_n).float(),     # (T,8)   标准化
                # FDI / 执行器一致性所需
                "health_gt": torch.from_numpy(health_seq).float(),      # (T,8)
                "is_saturated": torch.from_numpy(is_sat_seq).float(),   # (T,8)
                "regime_idx": torch.from_numpy(regime_idx).long(),      # (T,)
                "u_cmd_raw": torch.from_numpy(u_cmd).float(),           # (T,8) 原单位
                "u_app_raw": torch.from_numpy(u_app).float(),           # (T,8) 原单位
            }
        return batch

def collate_pad(batch_list):
    # 所有序列长度一致（seq_len），可直接堆叠
    out = {}
    for key in batch_list[0].keys():
        out[key] = torch.stack([b[key] for b in batch_list], dim=0)
    return out

# ------------------------------
# 训练脚本
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file_path", type=str,
        default="/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_10.hdf5")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--rollout_h", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./checkpoints_moe")
    p.add_argument("--std_path", type=str, default=None,
                   help="路径用于保存标准化统计(npz)。默认: <save_dir>/std_stats.npz")
    p.add_argument("--no_standardize", action="store_true")
    # 门控正则/路由
    p.add_argument("--gate_ent", type=float, default=1e-2)  ## 5e-2
    p.add_argument("--gate_tv", type=float, default=5e-2)  ## 1e-1
    p.add_argument("--gate_lb", type=float, default=5e-2)
    p.add_argument("--use_hard", action="store_true", help="推理期使用Gumbel硬路由")
    # FDI & 执行器一致性
    p.add_argument("--fdi_bce", type=float, default=1.0)
    p.add_argument("--fdi_tv", type=float, default=1e-2)
    p.add_argument("--fdi_sparse", type=float, default=1e-3)
    p.add_argument("--act_consist", type=float, default=1e-2)
    p.add_argument("--sat_weight", type=float, default=0.3)
    p.add_argument("--umax", type=float, default=20.0)
    p.add_argument("--du_max", type=float, default=0.0)  # <=0 关闭限斜率
    # 工况轻监督
    p.add_argument("--reg_ce", type=float, default=2e-2)  # 0 关闭
    # Scheduled Sampling
    p.add_argument("--ss_max", type=float, default=0.5)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    dataset = H5SeqDataset(args.file_path, seq_len=args.seq_len, stride=args.stride, use_standardizer=(not args.no_standardize))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_pad)
    # 保存 standardizer 统计
    std_path = args.std_path or os.path.join(args.save_dir, 'std_stats.npz')
    if getattr(dataset, 'std', None) is not None:
        try:
            dataset.std.save(std_path)
            print(f"[Info] Saved standardizer stats to: {std_path}")
        except Exception as e:
            print(f"[Warn] Failed to save standardizer: {e}")

    # Save standardizer
    if dataset.std is not None:
        std_path = os.path.join(args.save_dir, "standardizer_moe.npz")
        dataset.std.save(std_path)
        print(f"[Info] Standardizer saved to: {std_path}")

    # Model
    cfg = WMConfigMoE(
        x_dim=13, u_dim=8, n_experts=4,
        gru_hidden=128, gru_layers=2, mlp_hidden=256,
        rollout_horizon=args.rollout_h,
        gate_entropy_beta=args.gate_ent,
        gate_tv_gamma=args.gate_tv,
        gate_lb_alpha=args.gate_lb,
        use_hard_routing=args.use_hard,
        # FDI & 执行器一致性
        fdi_bce_alpha=args.fdi_bce,
        fdi_tv_alpha=args.fdi_tv,
        fdi_sparse_alpha=args.fdi_sparse,
        act_consist_alpha=args.act_consist,
        sat_weight=args.sat_weight,
        umax=args.umax,
        du_max=args.du_max,
        # 工况轻监督
        reg_ce_alpha=args.reg_ce,
        device=args.device
    )
    model = MoEWorldModel(cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda e: min(1.0, (e + 1) / 5))

    # Train
    best = 1e9
    for ep in range(1, args.epochs + 1):
        stats = train_one_epoch_moe(model, optim, loader, cfg, sched_samp_max=args.ss_max, epoch=ep-1, max_epochs=args.epochs)
        scheduler.step()
        print(f"[Epoch {ep:03d}] "
              f"step:{stats['step']:.4f} roll:{stats['roll']:.4f} "
              f"ent:{stats['ent']:.4f} tv:{stats['tv']:.4f} lb:{stats['lb']:.4f} regCE:{stats['reg_ce']:.4f} "
              f"FDI[bce:{stats['fdi_bce']:.4f} tv:{stats['fdi_tv']:.4f} sp:{stats['fdi_sparse']:.4f}] "
              f"u_cons:{stats['u_cons']:.4f} ss_p:{stats['ss_p']:.2f}")

        # Save ckpt（以 step+roll+FDI BCE 作为简易指标）
        val_metric = stats["roll"] + stats["step"] + 0.2 * stats["fdi_bce"]
        if val_metric < best or ep % 20 == 0:
            best = min(best, val_metric)
            ckpt_path = os.path.join(args.save_dir, f"moe_epoch{ep:03d}.pt")
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "cfg": cfg.__dict__,
            }, ckpt_path)
            print(f"[Info] checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    main()