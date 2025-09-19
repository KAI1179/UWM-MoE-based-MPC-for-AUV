
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate teacher-forced single-step prediction error for the MoE(3) world model.

Usage example:
python test_single_teacherforce_MoE3.py \
  --file data/your_episode.h5 \
  --ckpt path/to/checkpoints_moe3_aligned/moe3_epoch200.pt \
  --episode episode_0001 \
  --save_fig ./single_episode_pos_error_moe3.png
"""
import os
import math
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from worldmodel_MoE_3reg_1 import WMConfigMoE3, MoEWorldModel3, rollout

# ---------------------- Standardizer (NPZ loader) ----------------------
class NPZStandardizer:
    def __init__(self, x_mean, x_std, u_mean, u_std):
        self.x_mean = x_mean.astype(np.float32)
        self.x_std  = (x_std + 1e-8).astype(np.float32)
        self.u_mean = u_mean.astype(np.float32)
        self.u_std  = (u_std + 1e-8).astype(np.float32)

    @staticmethod
    def load(path):
        z = np.load(path)
        # support both keys: x_mean/x_std and mean/std
        x_mean = z['x_mean'] if 'x_mean' in z else z['mean']
        x_std  = z['x_std']  if 'x_std'  in z else z['std']
        u_mean = z['u_mean']
        u_std  = z['u_std']
        return NPZStandardizer(x_mean, x_std, u_mean, u_std)

def quat_normalize_np(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return (q / norm).astype(np.float32)

def parse_args():
    p = argparse.ArgumentParser(description="MoE3: Evaluate a single episode with teacher-forced 1-step predictions.")
    p.add_argument("--file", type=str, default="/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_1_labeled.hdf5", help="HDF5 路径")
    p.add_argument("--ckpt", type=str, default="./checkpoints_moe3_aligned/moe3_epoch150.pt", help="训练好的 MoE checkpoint 路径（.pt）")
    p.add_argument("--std", type=str, default=None, help="standardizer.npz 路径；默认取 ckpt 同目录")
    p.add_argument("--episode", type=str, default=None, help="要评估的 episode 名称（不填则取第一个）")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_steps", type=int, default=None, help="只评估前 K 步（含起点），默认整条序列")
    p.add_argument("--save_fig", type=str, default="./checkpoints_moe3_aligned/single_episode_pos_error_moe3.png", help="误差曲线输出路径")
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
        ori  = quat_normalize_np(np.array(g["orientation"]))  # [N,4]
        lv   = np.array(g["linear_velocity"])          # [N,3]
        av   = np.array(g["angular_velocity"])         # [N,3]
        # 支持两种字段名
        if "thrusts" in g:
            thr  = np.array(g["thrusts"])              # [N,u_dim]
        else:
            thr  = np.array(g["thrusts_applied"])      # [N,u_dim]

        N = min(len(time), pos.shape[0], ori.shape[0], lv.shape[0], av.shape[0], thr.shape[0])
        time, pos, ori, lv, av, thr = time[:N], pos[:N], ori[:N], lv[:N], av[:N], thr[:N]
        x = np.concatenate([pos, ori, lv, av], axis=-1).astype(np.float32)  # [N,13]
        u = thr.astype(np.float32)                                          # [N,u_dim]
    return ep, time, x, u

def apply_standardize_whole_sequence(std: NPZStandardizer, x: np.ndarray, u: np.ndarray):
    x_std = x.copy()
    x_std[:, 3:7] = quat_normalize_np(x_std[:, 3:7])
    idx_scale = np.r_[np.arange(0,3), np.arange(7,13)]
    x_std[:, idx_scale] = (x_std[:, idx_scale] - std.x_mean[idx_scale]) / (std.x_std[idx_scale] + 1e-8)
    u_std = (u - std.u_mean) / (std.u_std + 1e-8)
    return x_std.astype(np.float32), u_std.astype(np.float32)

def invert_position_standardization(std: NPZStandardizer, p_std: np.ndarray) -> np.ndarray:
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
    cfg = WMConfigMoE3(**cfg_dict)
    cfg.device = args.device

    # 2) 加载 standardizer
    std_path = args.std if args.std is not None else os.path.join(os.path.dirname(args.ckpt), "standardizer.npz")
    if not os.path.exists(std_path):
        raise FileNotFoundError(f"未找到 standardizer: {std_path}")
    std = NPZStandardizer.load(std_path)

    # 3) 读取 episode
    ep, time, x_raw, u_raw = load_episode(args.file, args.episode)
    N = x_raw.shape[0]
    u_dim = u_raw.shape[1]
    if N < 2:
        raise RuntimeError("该 episode 长度不足 2 步，无法做预测")
    if args.max_steps is not None:
        K = max(1, min(args.max_steps - 1, N - 1))
        time = time[:K+1]; x_raw = x_raw[:K+1]; u_raw = u_raw[:K]
    else:
        K = N - 1
        u_raw = u_raw[:K]

    # 4) 标准化
    x_std, u_std = apply_standardize_whole_sequence(std, x_raw, u_raw)

    # 5) 构建模型 & teacher-forced single-step
    model = MoEWorldModel3(cfg).to(cfg.device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    with torch.no_grad():
        x_hat_std = np.empty_like(x_std)  # (K+1, 13)
        x_hat_std[0] = x_std[0]
        for t in range(K):
            x0_t = torch.from_numpy(x_std[t:t+1]).to(cfg.device)          # (1,13)
            u_seq_t = torch.from_numpy(u_std[None, t:t+1, :]).to(cfg.device)  # (1,1,u_dim)
            x_pair = rollout(model, x0_t, u_seq_t)[0].detach().cpu().numpy()  # (2,13)
            x_hat_std[t+1] = x_pair[1]

    # 6) 反标准化到物理位置 & 误差
    p_gt = x_raw[:, 0:3]
    p_pred = invert_position_standardization(std, x_hat_std[:, 0:3])
    pos_err = np.linalg.norm(p_pred - p_gt, axis=-1)

    # 7) 打印逐步误差
    print(f"\n[MoE3] Episode: {ep} | length: {K+1} steps")
    header = "Step |   Time(s)   |        GT position (x,y,z)        |       Pred position (x,y,z)       |  |err|"
    print(header)
    print("-"*len(header))
    for t in range(K+1):
        print(f"{t:4d} | {time[t]:10.4f} | "
              f"{p_gt[t,0]:8.4f},{p_gt[t,1]:8.4f},{p_gt[t,2]:8.4f} | "
              f"{p_pred[t,0]:8.4f},{p_pred[t,1]:8.4f},{p_pred[t,2]:8.4f} | "
              f"{pos_err[t]:7.4f}")

    # 8) 绘图
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
        plt.title(f"[MoE3] Episode {ep} - Teacher-forced 1-step Position Error")
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
