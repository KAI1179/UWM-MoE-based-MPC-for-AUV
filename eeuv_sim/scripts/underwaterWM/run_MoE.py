# ===== run_MoE.py (REPLACEMENT) =====
import argparse, os, h5py, numpy as np, torch, random
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler

from utils import set_seed
from worldmodel_MoE import WMConfigMoE, MoEWorldModel, train_one_epoch_moe

class SimpleStandardizer:
    def __init__(self):
        self.mean = None
        self.std  = None
        self.u_mean = None
        self.u_std  = None
    def fit_states(self, X: torch.Tensor):
        mean = X.mean(0)
        std  = X.std(0).clamp(min=1e-6)
        mean[3:7] = 0.0
        std[3:7]  = 1.0
        self.mean = mean.float()
        self.std  = std.float()
    def transform_states(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std
    def fit_actions(self, U: torch.Tensor):
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

class H5SeqDataset(Dataset):
    """
    读取HDF5 -> 序列数据；提供 regime_weight（工况4仅post-fail步为1）。
    """
    def __init__(self, file_path, seq_len=64, stride=32, use_standardizer=True):
        self.fp = file_path
        self.seq_len = seq_len
        self.stride = stride
        self.use_std = use_standardizer

        self.episodes = []
        self.keys = []     # (ep_idx, start)
        self.ep_meta = {}  # ep_name -> (regime_id, fault_index)
        with h5py.File(self.fp, "r") as f:
            self.episodes = sorted([k for k in f.keys() if k.startswith("episode_")],
                                   key=lambda s: int(s.split("_")[1]))
            for e, k in enumerate(self.episodes):
                N = len(f[k]["time"])
                reg = int(f[k]["regime_id"][0]) if len(f[k]["regime_id"])>0 else -1
                fidx= int(f[k]["fault_index"][0]) if len(f[k]["fault_index"])>0 else -1
                self.ep_meta[k] = (reg, fidx)
                if N < seq_len + 1:
                    continue
                for s in range(0, N - seq_len, stride):
                    self.keys.append((e, s))

        # 标准化统计（u_applied）
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
                    ], axis=1).astype(np.float32)
                    u_all = f[k]["thrusts_applied"][()].astype(np.float32)
                    xs.append(x_all); us.append(u_all)
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

            # 动作
            u_app = f[k]["thrusts_applied"][s:s+self.seq_len].astype(np.float32)
            u_cmd = f[k]["thrusts_cmd"][s:s+self.seq_len].astype(np.float32)

            # labels/*
            health = is_sat = regime = du_abs = kappa = postfail = failgap = None
            if "labels" in f[k]:
                lab = f[k]["labels"]
                health = lab.get("health_mask_gt_hys", None)
                is_sat = lab.get("is_saturated", None)
                regime = lab.get("regime_step", None)
                du_abs = lab.get("du_abs", None)
                kappa  = lab.get("kappa", None)
                postfail = lab.get("postfail_mask", None)
                failgap  = lab.get("fail_gap_mask", None)

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

            # 生成 regime_weight：工况1/2/3全1；工况4仅post-fail为1
            reg_id = int(f[k]["regime_id"][0]) if len(f[k]["regime_id"])>0 else 1
            if reg_id == 4:
                if postfail is not None:
                    pf_full = postfail[()]  # (N,)
                    pf_seq = pf_full[s:s+T].astype(np.float32)
                else:
                    # 回退：由 health 推导（任一通道为0即认为post-fail）
                    pf_seq = (health_seq.min(axis=1) < 1.0).astype(np.float32)
                regime_weight = pf_seq
                if failgap is not None:
                    fg_full = failgap[()]
                    fg_seq = fg_full[s:s+T].astype(np.float32)
                    regime_weight = regime_weight * (1.0 - fg_seq)
            else:
                regime_weight = np.ones((T,), dtype=np.float32)

            # 切片 x(t), x(t+1)
            x_t   = x[:-1]
            x_t1  = x[1:]

            # 标准化
            if self.std is not None:
                x_t  = self.std.transform_states(torch.from_numpy(x_t)).numpy()
                x_t1 = self.std.transform_states(torch.from_numpy(x_t1)).numpy()
                u_app_n = self.std.transform_actions(torch.from_numpy(u_app)).numpy()
            else:
                u_app_n = u_app

            batch = {
                "x": torch.from_numpy(x_t).float(),         # (T,13)
                "x_next": torch.from_numpy(x_t1).float(),   # (T,13)
                "u": torch.from_numpy(u_app_n).float(),     # (T,8)
                "health_gt": torch.from_numpy(health_seq).float(),
                "is_saturated": torch.from_numpy(is_sat_seq).float(),
                "regime_idx": torch.from_numpy(regime_idx).long(),
                "regime_weight": torch.from_numpy(regime_weight).float(),
                "u_cmd_raw": torch.from_numpy(u_cmd).float(),
                "u_app_raw": torch.from_numpy(u_app).float(),
            }
        return batch

    # 供采样器使用：返回某个样本的“核心组”与“子组”标签
    def group_of_index(self, idx: int):
        ep_idx, _ = self.keys[idx]
        ep_name = self.episodes[ep_idx]
        reg, fidx = self.ep_meta[ep_name]
        core = f"reg{reg}" if reg in (1,2,3,4) else "reg?"
        sub = core if reg != 4 else f"reg4_f{max(fidx,0)}"
        return core, sub

def collate_pad(batch_list):
    out = {}
    for key in batch_list[0].keys():
        out[key] = torch.stack([b[key] for b in batch_list], dim=0)
    return out

class BalancedBatchSampler(BatchSampler):
    """
    最小覆盖批采样器：每个batch至少包含 reg1/reg2/reg3/reg4 各1条，其余随机填充。
    - 不严格改变全局分布；coverage样本允许带替换抽样，简单稳妥。
    """
    def __init__(self, dataset: H5SeqDataset, batch_size: int, seed: int = 42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rng = random.Random(seed)

        # 建立核心组到索引的映射
        self.group_to_indices = {"reg1": [], "reg2": [], "reg3": [], "reg4": []}
        for i in range(len(dataset)):
            core, _ = dataset.group_of_index(i)
            if core in self.group_to_indices:
                self.group_to_indices[core].append(i)

        self.all_indices = list(range(len(dataset)))
        self._shuffle_pool()

        # 一个epoch的批次数（与普通DataLoader等价）
        self.num_batches = (len(self.all_indices) + self.batch_size - 1) // self.batch_size

    def _shuffle_pool(self):
        self.rng.shuffle(self.all_indices)
        self.pool_ptr = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # 每个epoch重新洗牌主池
        self._shuffle_pool()
        for _ in range(self.num_batches):
            batch = []

            # 先放置最小覆盖（若某组空，则跳过）
            for g in ["reg1", "reg2", "reg3", "reg4"]:
                if self.group_to_indices[g]:
                    batch.append(self.rng.choice(self.group_to_indices[g]))

            # 去重（避免后续补齐重复）
            batch = list(dict.fromkeys(batch))

            # 补齐剩余名额，从主池无放回取样
            need = self.batch_size - len(batch)
            while need > 0:
                if self.pool_ptr >= len(self.all_indices):
                    # 主池用尽则重洗
                    self._shuffle_pool()
                idx = self.all_indices[self.pool_ptr]
                self.pool_ptr += 1
                if idx in batch:
                    continue
                batch.append(idx)
                need -= 1

            yield batch

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
    p.add_argument("--rollout_h", type=int, default=30)         # ↑ 20 -> 30
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./checkpoints_moe")
    p.add_argument("--std_path", type=str, default=None,
                   help="路径用于保存标准化统计(npz)。默认: <save_dir>/std_stats.npz")
    p.add_argument("--no_standardize", action="store_true")
    # 门控正则/路由
    p.add_argument("--gate_ent", type=float, default=1e-2)
    p.add_argument("--gate_tv", type=float, default=2e-2)       # ↓ 5e-2 -> 2e-2
    p.add_argument("--gate_lb", type=float, default=5e-2)
    p.add_argument("--use_hard", action="store_true", help="推理期使用Gumbel硬路由")
    # FDI & 执行器一致性
    p.add_argument("--fdi_bce", type=float, default=1.0)
    p.add_argument("--fdi_tv", type=float, default=1e-2)
    p.add_argument("--fdi_sparse", type=float, default=1e-3)
    p.add_argument("--act_consist", type=float, default=1e-2)
    p.add_argument("--sat_weight", type=float, default=0.3)
    p.add_argument("--umax", type=float, default=20.0)
    p.add_argument("--du_max", type=float, default=0.0)
    # 工况轻监督
    p.add_argument("--reg_ce", type=float, default=1e-1)        # ↑ 2e-2 -> 1e-1
    # Scheduled Sampling
    p.add_argument("--ss_max", type=float, default=0.8)         # ↑ 0.5 -> 0.8
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    dataset = H5SeqDataset(args.file_path, seq_len=args.seq_len, stride=args.stride, use_standardizer=(not args.no_standardize))
    # 保存 standardizer
    std_path = args.std_path or os.path.join(args.save_dir, 'std_stats.npz')
    if getattr(dataset, 'std', None) is not None:
        try:
            dataset.std.save(std_path)
            print(f"[Info] Saved standardizer stats to: {std_path}")
        except Exception as e:
            print(f"[Warn] Failed to save standardizer: {e}")
        # 兼容原命名
        std_path2 = os.path.join(args.save_dir, "standardizer_moe.npz")
        try:
            dataset.std.save(std_path2)
            print(f"[Info] Standardizer saved to: {std_path2}")
        except Exception as e:
            print(f"[Warn] Failed to save standardizer: {e}")

    # 使用“最小覆盖”批采样器
    batch_sampler = BalancedBatchSampler(dataset, batch_size=args.batch_size, seed=args.seed)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, collate_fn=collate_pad)

    # Model
    cfg = WMConfigMoE(
        x_dim=13, u_dim=8, n_experts=4,
        gru_hidden=128, gru_layers=2, mlp_hidden=256,
        rollout_horizon=args.rollout_h,
        gate_entropy_beta=args.gate_ent,
        gate_tv_gamma=args.gate_tv,
        gate_lb_alpha=args.gate_lb,
        use_hard_routing=args.use_hard,
        # 训练期门控带噪（Gumbel-Softmax）
        gumbel_tau=0.6,
        gate_train_noisy=True,
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
        # if val_metric < best or ep % 20 == 0:
        if ep % 20 == 0:
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
# ===== END =====
