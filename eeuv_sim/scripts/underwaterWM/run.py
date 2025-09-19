# run.py
# 命令行训练脚本：读取 HDF5 -> DataLoader -> 训练 GRU 世界模型 -> 保存

import argparse
import os
import torch

from utils import set_seed
from dataloader import make_dataloader_from_hdf5
from worldModel import WMConfig, ROVGRUModel, train_one_epoch

def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--file_path", type=str, default="/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_trust1data10.hdf5", required=True, help="HDF5 数据文件路径")
    p.add_argument("--file_path", type=str, default="/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_trust05data10_11.hdf5", required=True, help="HDF5 数据文件路径")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--k_consistency", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./checkpoints/20250819_1934")
    p.add_argument("--no_standardize", action="store_true", help="关闭标准化")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) DataLoader
    loader, std, u_dim = make_dataloader_from_hdf5(
        file_path=args.file_path,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pad_last=True,
        use_standardizer=(not args.no_standardize),
    )
    if std is not None:
        std_path = os.path.join(args.save_dir, "standardizer.npz")
        std.save(std_path)
        print(f"[Info] Standardizer saved to: {std_path}")

    # 2) Model & Optimizer
    cfg = WMConfig(
        x_dim=13,
        u_dim=u_dim,
        h_dim=256,
        ff_hidden=256,
        k_consistency=args.k_consistency,
        device=args.device
    )
    model = ROVGRUModel(cfg).to(cfg.device)
    # optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # run.py
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda e: min(1.0, (e + 1) / 5)  # 前 5 个 epoch 从 0.2,0.4,...,1.0
    )


    # 3) Train
    for ep in range(1, args.epochs + 1):

        stats = train_one_epoch(model, optim, loader, cfg)
        scheduler.step()
        print(f"[Epoch {ep:03d}] NLL: {stats['nll']:.4f} | Cons: {stats['cons']:.4f}")

        if ep % 50 == 0:

            # 保存 checkpoint
            ckpt_path = os.path.join(args.save_dir, f"model_epoch{ep:03d}.pt")
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optim.state_dict(),
                "cfg": cfg.__dict__,
            }, ckpt_path)
            print(f"[Info] checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    main()
