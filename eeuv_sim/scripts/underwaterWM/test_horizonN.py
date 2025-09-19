#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate fixed-horizon H-step prediction error for an underwater world model.

Usage example:
python test_horizonN.py \
  --file data/your_episode.h5 \
  --ckpt runs/exp1/best.pt \
  --episode ep_0001 \
  --horizon 10 \
  --save_fig ./horizon10_pos_error.png
"""

import os
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import Standardizer, quat_normalize_np
from worldModel import WMConfig, ROVGRUModel, rollout


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fixed-horizon H-step prediction error (position bias / error).")
    p.add_argument("--file", type=str, required=True, help="HDF5 路径")
    p.add_argument("--ckpt", type=str, required=True, help="训练好的 checkpoint 路径（.pt）")
    p.add_argument("--std", type=str, default=None, help="standardizer.npz 路径；默认取 ckpt 同目录")
    p.add_argument("--episode", type=str, default=None, help="要评估的 episode 名称（不填则取第一个）")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--horizon", type=int, default=10, help="固定地平线步数 H（向后预测 H 步）")
    p.add_argument("--save_fig", type=str, default="./horizon_pos_error.png", help="误差曲线输出路径")
    p.add_argument("--save_csv", type=str, default=None, help="可选：保存每个起点的误差到 CSV")
    return p.parse_args()


def load_episode(h5_path, ep_name=None):
    with h5py.File(h5_path, "r") as f:
        keys = sorted(list(f.keys()))
        if len(keys) == 0:
            raise RuntimeError("HDF5 中没有任何 episode")
        if ep_name is None:
            ep = keys[0]
        else:
            if ep_name not in f:
                raise KeyError(f"找不到指定 episode: {ep_name}. 可选: {keys[:5]}{'...' if len(keys)>5 else ''}")
            ep = ep_name
        g = f[ep]
        time = np.array(g["time"])                     # [N]
        pos  = np.array(g["position"])                 # [N,3]
        ori  = quat_normalize_np(np.array(g["orientation"]))  # [N,4] (wxyz)
        lv   = np.array(g["linear_velocity"])          # [N,3]
        av   = np.array(g["angular_velocity"])         # [N,3]
        thr  = np.array(g["thrusts"])                  # [N,u_dim]

        N = min(len(time), pos.shape[0], ori.shape[0], lv.shape[0], av.shape[0], thr.shape[0])
        time, pos, ori, lv, av, thr = time[:N], pos[:N], ori[:N], lv[:N], av[:N], thr[:N]
        x = np.concatenate([pos, ori, lv, av], axis=-1).astype(np.float32)  # [N,13]
        u = thr.astype(np.float32)                                          # [N,u_dim]
    return ep, time, x, u


def apply_standardize_whole_sequence(std: Standardizer, x: np.ndarray, u: np.ndarray):
    """将整条序列按训练时的统计做标准化；四元数只单位化不缩放。"""
    x_std = x.copy()
    x_std[:, 3:7] = quat_normalize_np(x_std[:, 3:7])
    # p(0:3), v(7:10), w(10:13) 做 z-score
    idx_scale = np.r_[np.arange(0, 3), np.arange(7, 13)]
    x_std[:, idx_scale] = (x_std[:, idx_scale] - std.x_mean[idx_scale]) / (std.x_std[idx_scale] + 1e-8)
    u_std = (u - std.u_mean) / (std.u_std + 1e-8)
    return x_std.astype(np.float32), u_std.astype(np.float32)


def invert_position_standardization(std: Standardizer, p_std: np.ndarray) -> np.ndarray:
    """把标准化空间的 position (M,3) 还原到物理单位。"""
    p_mean = std.x_mean[0:3]
    p_stdv = std.x_std[0:3] + 1e-8
    return p_std * p_stdv + p_mean


def main():
    args = parse_args()

    # 1) 加载 checkpoint 与 cfg
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise RuntimeError("checkpoint 中缺少 cfg 字段")
    cfg = WMConfig(**cfg_dict)
    cfg.device = args.device

    # 2) 加载 standardizer（必须与训练一致）
    std_path = args.std if args.std is not None else os.path.join(os.path.dirname(args.ckpt), "standardizer.npz")
    if not os.path.exists(std_path):
        raise FileNotFoundError(f"未找到 standardizer: {std_path}")
    std = Standardizer.load(std_path)

    # 3) 读取 episode
    ep, time, x_raw, u_raw = load_episode(args.file, args.episode)
    N = x_raw.shape[0]
    if N < 2:
        raise RuntimeError("该 episode 长度不足 2 步，无法做预测")
    H = int(args.horizon)
    if H < 1:
        raise ValueError("horizon 必须 >= 1")
    if N <= H:
        raise RuntimeError(f"该 episode 长度 {N} 不足以做 H={H} 步预测（需要 N > H）")

    # 4) 标准化整条序列
    x_std, u_std = apply_standardize_whole_sequence(std, x_raw, u_raw)

    # 5) 构建模型
    model = ROVGRUModel(cfg).to(cfg.device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # 6) 逐个起点做“固定地平线 H 步自由 rollout”
    # 起点 t 的预测末端是 t+H，所以 t 的范围是 [0, N-1-H]
    T0 = N - 1 - H + 1  # 可用起点数
    pos_err_norm = np.zeros(T0, dtype=np.float32)     # |Δp| at horizon H
    pos_err_vec  = np.zeros((T0, 3), dtype=np.float32)  # Δp 向量（bias 用）
    t_eval = np.zeros(T0, dtype=np.float64)           # 对应的时间戳（取 t+H 的时刻）

    with torch.no_grad():
        for i, t in enumerate(range(0, N - H)):
            # 起点：真实 x_t（标准化域）
            x0 = torch.from_numpy(x_std[t:t+1]).to(cfg.device)            # (1,13)
            # 控制序列：u[t : t+H]
            u_seq = torch.from_numpy(u_std[None, t:t+H, :]).to(cfg.device)  # (1,H,u_dim)

            # 自由 rollout H 步，得到 (1, H+1, 13)；索引 H 处是 x_{t+H | t}
            x_hat_std = rollout(model, x0, u_seq)[0].detach().cpu().numpy()  # (H+1,13)
            p_pred_std = x_hat_std[H, 0:3][None, :]  # (1,3) 末端位置（标准化域）
            p_pred = invert_position_standardization(std, p_pred_std)[0]  # (3,)

            # 真实末端位置
            p_gt = x_raw[t + H, 0:3]  # (3,)

            # 误差
            dp = p_pred - p_gt
            pos_err_vec[i] = dp
            pos_err_norm[i] = np.linalg.norm(dp)
            t_eval[i] = float(time[t + H])

    # 7) 统计指标：bias（各轴平均偏差）、|err| 统计
    bias_xyz = pos_err_vec.mean(axis=0)  # (3,)
    mean_err = float(pos_err_norm.mean())
    med_err  = float(np.median(pos_err_norm))
    p95_err  = float(np.percentile(pos_err_norm, 95))

    print(f"\nEpisode: {ep}")
    print(f"Horizon H = {H} 步 | 起点数量 = {T0}")
    print("—— 10步末端位置偏差统计 ——" if H == 10 else "—— H步末端位置偏差统计 ——")
    print(f"Bias (mean Δp) [m]:  dx={bias_xyz[0]:+.4f}, dy={bias_xyz[1]:+.4f}, dz={bias_xyz[2]:+.4f}")
    print(f"|Δp| Mean [m]:      {mean_err:.4f}")
    print(f"|Δp| Median [m]:    {med_err:.4f}")
    print(f"|Δp| 95th [m]:      {p95_err:.4f}")

    # 8) 可选：保存每个起点的误差到 CSV
    if args.save_csv is not None:
        import csv
        head = ["t_index", "time", "err_x", "err_y", "err_z", "err_norm"]
        with open(args.save_csv, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(head)
            for i, t in enumerate(range(0, N - H)):
                w.writerow([t, t_eval[i], pos_err_vec[i,0], pos_err_vec[i,1], pos_err_vec[i,2], pos_err_norm[i]])
        print(f"[Saved] per-start H-step errors -> {args.save_csv}")

    # 9) 绘图（H 步末端 |Δp| 随时间）
    try:
        plt.figure(figsize=(10, 4.5))
        # 若时间戳单调匹配长度，横轴用 time[t+H]
        if t_eval.shape[0] == pos_err_norm.shape[0]:
            x_axis = t_eval
            plt.xlabel("Time at t+H (s)")
        else:
            x_axis = np.arange(pos_err_norm.shape[0])
            plt.xlabel("Start index t")
        plt.plot(x_axis, pos_err_norm, linewidth=2)
        plt.ylabel(f"|Δp| at horizon H={H} (m)")
        plt.title(f"Episode {ep} - {H}-step Ahead Position Error")
        plt.grid(True, alpha=0.3)
        out_path = args.save_fig
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Saved] H-step ahead position error curve -> {out_path}")
    except Exception as e:
        print(f"[Warn] 绘图失败：{e}")


if __name__ == "__main__":
    main()
