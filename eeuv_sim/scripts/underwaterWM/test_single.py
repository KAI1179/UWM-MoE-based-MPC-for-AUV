#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import Standardizer, quat_normalize_np
from worldModel import WMConfig, ROVGRUModel, rollout

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a single episode: print per-step GT vs Pred and plot position error.")
    p.add_argument("--file", type=str, required=True, help="HDF5 路径")
    p.add_argument("--ckpt", type=str, required=True, help="训练好的 checkpoint 路径（.pt）")
    p.add_argument("--std", type=str, default=None, help="standardizer.npz 路径；默认取 ckpt 同目录")
    p.add_argument("--episode", type=str, default=None, help="要评估的 episode 名称（不填则取第一个）")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_steps", type=int, default=None, help="只评估前 K 步（含起点），默认整条序列")
    p.add_argument("--save_fig", type=str, default="./single_episode_pos_error.png", help="误差曲线输出路径")
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
    # utils.Standardizer 里是逐块方法；这边做 whole-seq 应用
    x_std = x.copy()
    # 先保证四元数单位化
    x_std[:, 3:7] = quat_normalize_np(x_std[:, 3:7])
    idx_scale = np.r_[np.arange(0,3), np.arange(7,13)]
    x_std[:, idx_scale] = (x_std[:, idx_scale] - std.x_mean[idx_scale]) / (std.x_std[idx_scale] + 1e-8)
    u_std = (u - std.u_mean) / (std.u_std + 1e-8)
    return x_std.astype(np.float32), u_std.astype(np.float32)

def invert_position_standardization(std: Standardizer, p_std: np.ndarray) -> np.ndarray:
    """把标准化空间的 position (N,3) 还原到物理单位"""
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
    u_dim = u_raw.shape[1]
    if N < 2:
        raise RuntimeError("该 episode 长度不足 2 步，无法做预测")
    if args.max_steps is not None:
        # max_steps 定义为包含起点在内的预测长度 => 最多使用 K = max_steps-1 个动作
        K = max(1, min(args.max_steps - 1, N - 1))
        time = time[:K+1]; x_raw = x_raw[:K+1]; u_raw = u_raw[:K]  # u 用到 K 个
    else:
        K = N - 1
        u_raw = u_raw[:K]

    # 4) 标准化整条序列
    x_std, u_std = apply_standardize_whole_sequence(std, x_raw, u_raw)

    # 5) 构建模型 & rollout
    model = ROVGRUModel(cfg).to(cfg.device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    x0 = torch.from_numpy(x_std[0:1]).to(cfg.device)            # (1,13)
    u_seq = torch.from_numpy(u_std[None, :, :]).to(cfg.device)  # (1,K,u_dim)
    x_hat_std = rollout(model, x0, u_seq)[0].detach().cpu().numpy()  # (K+1,13)

    # 6) 反标准化位置用于物理量误差
    p_gt = x_raw[:, 0:3]                             # (K+1,3) 真值（物理单位）
    p_pred = invert_position_standardization(std, x_hat_std[:, 0:3])  # (K+1,3)
    pos_err = np.linalg.norm(p_pred - p_gt, axis=-1) # (K+1,)

    # 7) 打印逐步对比
    print(f"\nEpisode: {ep} | length: {K+1} steps")
    header = "Step |   Time(s)   |        GT position (x,y,z)        |       Pred position (x,y,z)       |  |err|"
    print(header)
    print("-"*len(header))
    for t in range(K+1):
        print(f"{t:4d} | {time[t]:10.4f} | "
              f"{p_gt[t,0]:8.4f},{p_gt[t,1]:8.4f},{p_gt[t,2]:8.4f} | "
              f"{p_pred[t,0]:8.4f},{p_pred[t,1]:8.4f},{p_pred[t,2]:8.4f} | "
              f"{pos_err[t]:7.4f}")

    # 8) 绘图（位置误差曲线）
    try:
        plt.figure(figsize=(9,4.5))
        if time.shape[0] == pos_err.shape[0]:
            x_axis = time
            plt.xlabel("Time (s)")
        else:
            x_axis = np.arange(pos_err.shape[0])
            plt.xlabel("Step")
        plt.plot(x_axis, pos_err, linewidth=2)
        plt.ylabel("Position error |Δp|")
        plt.title(f"Episode {ep} - Position Error")
        plt.grid(True, alpha=0.3)
        out_path = args.save_fig
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\n[Saved] position error curve -> {out_path}")
    except Exception as e:
        print(f"[Warn] 绘图失败：{e}")

if __name__ == "__main__":
    main()
