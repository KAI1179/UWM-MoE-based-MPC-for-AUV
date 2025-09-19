
"""
MoE(3 regimes) training script aligned with worldModel/run:
 - Same single-step diagonal-Gaussian NLL on deltas
 - Same multi-step consistency loss
 - Same standardization (no scaling on quaternion [3:7])
 - No FDI / actuator terms
"""
import argparse, os, numpy as np, torch, h5py, random
from torch.utils.data import Dataset, DataLoader, BatchSampler
from typing import Optional
from worldmodel_MoE_3reg_1 import WMConfigMoE3, MoEWorldModel3, train_one_epoch
from utils import set_seed

# --------------------------- Standardizer ---------------------------
class SimpleStandardizer:
    def __init__(self):
        self.mean=None; self.std=None; self.u_mean=None; self.u_std=None
    def fit_states(self, X: torch.Tensor):
        mean = X.mean(0); std = X.std(0).clamp(min=1e-6)
        # strictly follow baseline: do NOT scale quaternion
        mean[3:7] = 0.0; std[3:7] = 1.0
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

# --------------------------- Dataset ---------------------------
class H5SeqDataset(Dataset):
    def __init__(self, file_path, seq_len=64, stride=32, use_standardizer=True):
        self.fp=file_path; self.seq_len=seq_len; self.stride=stride; self.use_std=use_standardizer
        self.episodes=[]; self.keys=[]
        with h5py.File(self.fp,"r") as f:
            all_eps=sorted([k for k in f.keys() if k.startswith("episode_")], key=lambda s:int(s.split("_")[1]))
            kept=[k for k in all_eps]
            self.episodes=kept
            for e,k in enumerate(self.episodes):
                N=len(f[k]["time"])
                if N<seq_len+1: continue
                for s in range(0,N-seq_len,stride): self.keys.append((e,s))
        self.std=None
        if self.use_std:
            xs,us=[],[]
            with h5py.File(self.fp,"r") as f:
                for k in self.episodes:
                    x_all=np.concatenate([f[k]["position"][()],
                                          f[k]["orientation"][()],
                                          f[k]["linear_velocity"][()],
                                          f[k]["angular_velocity"][()]],axis=1).astype(np.float32)
                    u_all=f[k]["thrusts_applied"][()].astype(np.float32)
                    xs.append(x_all); us.append(u_all)
            X=torch.tensor(np.concatenate(xs,0),dtype=torch.float32)
            U=torch.tensor(np.concatenate(us,0),dtype=torch.float32)
            self.std=SimpleStandardizer(); self.std.fit_states(X); self.std.fit_actions(U)

    def __len__(self): return len(self.keys)
    # def __getitem__(self, idx):
    #     ep_idx,s=self.keys[idx]
    #     with h5py.File(self.fp,"r") as f:
    #         k=self.episodes[ep_idx]
    #         pos=f[k]["position"][s:s+self.seq_len+1].astype(np.float32)
    #         ori=f[k]["orientation"][s:s+self.seq_len+1].astype(np.float32)
    #         lin=f[k]["linear_velocity"][s:s+self.seq_len+1].astype(np.float32)
    #         ang=f[k]["angular_velocity"][s:s+self.seq_len+1].astype(np.float32)
    #         x=np.concatenate([pos,ori,lin,ang],axis=1).astype(np.float32)  # (T+1, 13)
    #         u=f[k]["thrusts_applied"][s:s+self.seq_len].astype(np.float32) # (T, 8)
    #
    #     x_t, x_t1 = x[:-1], x[1:]
    #     if self.std is not None:
    #         x_t=torch.from_numpy(x_t); x_t1=torch.from_numpy(x_t1); u=torch.from_numpy(u)
    #         x_t=self.std.transform_states(x_t).numpy()
    #         x_t1=self.std.transform_states(x_t1).numpy()
    #         u=self.std.transform_actions(u).numpy()
    #
    #     return {
    #         "x": torch.from_numpy(x_t).float(),        # (T, 13)
    #         "x_next": torch.from_numpy(x_t1).float(),  # (T, 13)
    #         "u": torch.from_numpy(u).float(),          # (T, 8)
    #     }

    def __getitem__(self, idx):
        ep_idx, s = self.keys[idx]
        with h5py.File(self.fp, "r") as f:
            k = self.episodes[ep_idx]
            pos = f[k]["position"][s:s + self.seq_len + 1].astype(np.float32)
            ori = f[k]["orientation"][s:s + self.seq_len + 1].astype(np.float32)
            lin = f[k]["linear_velocity"][s:s + self.seq_len + 1].astype(np.float32)
            ang = f[k]["angular_velocity"][s:s + self.seq_len + 1].astype(np.float32)
            x = np.concatenate([pos, ori, lin, ang], axis=1).astype(np.float32)  # (T+1, 13)
            u = f[k]["thrusts_applied"][s:s + self.seq_len].astype(np.float32)  # (T, 8)

            # ★ 新增：读取 label_make.py 写入的 labels/regime_step（每步一个 regime id）
            # 只取与当前序列对齐的前 T 个步长
            # reg = f[k]["labels"]["regime_step"][s:s + self.seq_len].astype(np.int64)  # (T,)
            reg = f[k]["labels"]["regime_step"][s:s + self.seq_len].astype(np.int64) - 1  # (T,)

        x_t, x_t1 = x[:-1], x[1:]
        if self.std is not None:
            x_t = torch.from_numpy(x_t);
            x_t1 = torch.from_numpy(x_t1);
            u = torch.from_numpy(u)
            x_t = self.std.transform_states(x_t).numpy()
            x_t1 = self.std.transform_states(x_t1).numpy()
            u = self.std.transform_actions(u).numpy()

        return {
            "x": torch.from_numpy(x_t).float(),  # (T, 13)
            "x_next": torch.from_numpy(x_t1).float(),  # (T, 13)
            "u": torch.from_numpy(u).float(),  # (T, 8)
            # ★ 新增：回传给训练循环用于 CE 监督（worldmodel_MoE_3reg_1.py 中使用）
            "regime_step": torch.from_numpy(reg).long(),  # (T,)
        }


class BalancedBatchSampler(BatchSampler):
    """Simple balanced sampler over episodes (best-effort)."""
    def __init__(self, dataset:H5SeqDataset, batch_size:int, seed:int=42):
        self.dataset=dataset; self.batch_size=int(batch_size); self.rng=random.Random(seed)
        self.all_indices=list(range(len(dataset))); self._shuffle_pool()
        self.num_batches=(len(self.all_indices)+self.batch_size-1)//self.batch_size
    def _shuffle_pool(self): self.rng.shuffle(self.all_indices); self.pool_ptr=0
    def __len__(self): return self.num_batches
    def __iter__(self):
        self._shuffle_pool()
        for _ in range(self.num_batches):
            batch=[]
            need=self.batch_size-len(batch)
            while need>0:
                if self.pool_ptr>=len(self.all_indices): self._shuffle_pool()
                idx=self.all_indices[self.pool_ptr]; self.pool_ptr+=1
                batch.append(idx); need-=1
            yield batch

def collate_pad(batch_list):
    out={}
    for key in batch_list[0].keys():
        out[key]=torch.stack([b[key] for b in batch_list], dim=0)
    return out

# --------------------------- Train loop (same style as run.py) ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file_path", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--k_consistency", type=int, default=15)
    p.add_argument("--regime_ce_weight", type=float, default=0.1)  ## 新加上
    p.add_argument("--lb_weight", type=float, default=0.01)  ## 新加
    p.add_argument("--entropy_weight", type=float, default=0.001)  ## 新加
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./checkpoints_moe3_20")
    p.add_argument("--no_standardize", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    dataset = H5SeqDataset(args.file_path, seq_len=args.seq_len, stride=args.stride, use_standardizer=(not args.no_standardize))
    if getattr(dataset, "std", None) is not None:
        try:
            std_path = os.path.join(args.save_dir, "standardizer.npz")
            dataset.std.save(std_path)
            print(f"[Info] Standardizer saved to: {std_path}")
        except Exception as e:
            print(f"[Warn] Failed to save standardizer: {e}")
    loader = DataLoader(dataset, batch_sampler=BalancedBatchSampler(dataset, batch_size=args.batch_size, seed=args.seed),
                        num_workers=0, collate_fn=collate_pad)

    # Model
    cfg = WMConfigMoE3(
        x_dim=13, u_dim=dataset[0]["u"].shape[-1],
        n_experts=3, h_dim=256, ff_hidden=256,
        k_consistency=args.k_consistency,
        device=args.device
    )
    model = MoEWorldModel3(cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda e: min(1.0, (e + 1) / 5))

    # Train
    for ep in range(1, args.epochs + 1):
        stats = train_one_epoch(model, optim, loader, cfg)
        scheduler.step()
        print(f"[Epoch {ep:03d}] NLL: {stats['nll']:.4f} | Cons: {stats['cons']:.4f}")
        if ep % 50 == 0:
            ckpt_path = os.path.join(args.save_dir, f"moe3_epoch{ep:03d}.pt")
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "cfg": cfg.__dict__,
            }, ckpt_path)
            print(f"[Info] checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    main()