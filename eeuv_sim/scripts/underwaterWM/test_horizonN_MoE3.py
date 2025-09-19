
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate fixed-horizon H-step (default 10) free-rollout position error for the MoE(3) world model.

Usage example:
python test_horizonN_MoE3.py \
  --file data/your_episode.h5 \
  --ckpt path/to/checkpoints_moe3_aligned/moe3_epoch200.pt \
  --episode episode_0001 \
  --horizon 10 \
  --save_fig ./horizon10_pos_error_moe3.png
"""
import os
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
    p = argparse.ArgumentParser(description="MoE3: Evaluate fixed-horizon H-step free rollout position error.")
    p.add_argument("--file", type=str, default="/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_1_labeled.hdf5", help="HDF5 路径")
    p.add_argument("--ckpt", type=str, default="./checkpoints_moe3_aligned/moe3_epoch200.pt", help="训练好的 MoE checkpoint 路径（.pt）")
    p.add_argument("--std", type=str, default=None, help="standardizer.npz 路径；默认取 ckpt 同目录")
    p.add_argument("--episode", type=str, default=None, help="要评估的 episode 名称（不填则取第一个）")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--horizon", type=int, default=10, help="固定地平线步数 H（向后预测 H 步）")
    p.add_argument("--save_fig", type=str, default="./horizon_pos_error_moe3.png", help="误差曲线输出路径")
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
    if N < 2:
        raise RuntimeError("该 episode 长度不足 2 步，无法做预测")
    H = int(args.horizon)
    if H < 1:
        raise ValueError("horizon 必须 >= 1")
    if N <= H:
        raise RuntimeError(f"该 episode 长度 {N} 不足以做 H={H} 步预测（需要 N > H）")

    # 4) 标准化
    x_std, u_std = apply_standardize_whole_sequence(std, x_raw, u_raw)

    # 5) 构建模型
    model = MoEWorldModel3(cfg).to(cfg.device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # 6) 逐起点自由 rollout H 步
    T0 = N - 1 - H + 1
    pos_err_norm = np.zeros(T0, dtype=np.float32)
    pos_err_vec  = np.zeros((T0, 3), dtype=np.float32)
    t_eval = np.zeros(T0, dtype=np.float64)

    with torch.no_grad():
        for i, t in enumerate(range(0, N - H)):
            x0 = torch.from_numpy(x_std[t:t+1]).to(cfg.device)            # (1,13)
            u_seq = torch.from_numpy(u_std[None, t:t+H, :]).to(cfg.device)  # (1,H,u_dim)

            x_hat_std = rollout(model, x0, u_seq)[0].detach().cpu().numpy()  # (H+1,13)
            p_pred_std = x_hat_std[H, 0:3][None, :]
            p_pred = invert_position_standardization(std, p_pred_std)[0]

            p_gt = x_raw[t + H, 0:3]
            dp = p_pred - p_gt
            pos_err_vec[i] = dp
            pos_err_norm[i] = np.linalg.norm(dp)
            t_eval[i] = float(time[t + H])

    # 7) 统计指标
    bias_xyz = pos_err_vec.mean(axis=0)  # (3,)
    mean_err = float(pos_err_norm.mean())
    med_err  = float(np.median(pos_err_norm))
    p95_err  = float(np.percentile(pos_err_norm, 95))

    print(f"\n[MoE3] Episode: {ep}")
    print(f"Horizon H = {H} 步 | 起点数量 = {T0}")
    print("—— 10步末端位置偏差统计 ——" if H == 10 else "—— H步末端位置偏差统计 ——")
    print(f"Bias (mean Δp) [m]:  dx={bias_xyz[0]:+.4f}, dy={bias_xyz[1]:+.4f}, dz={bias_xyz[2]:+.4f}")
    print(f"|Δp| Mean [m]:      {mean_err:.4f}")
    print(f"|Δp| Median [m]:    {med_err:.4f}")
    print(f"|Δp| 95th [m]:      {p95_err:.4f}")

    # 8) 可选：保存 CSV
    if args.save_csv is not None:
        import csv
        head = ["t_index", "time", "err_x", "err_y", "err_z", "err_norm"]
        with open(args.save_csv, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(head)
            for i, t in enumerate(range(0, N - H)):
                w.writerow([t, t_eval[i], pos_err_vec[i,0], pos_err_vec[i,1], pos_err_vec[i,2], pos_err_norm[i]])
        print(f"[Saved] per-start H-step errors -> {args.save_csv}")

    # 9) 绘图
    try:
        plt.figure(figsize=(10, 4.5))
        if t_eval.shape[0] == pos_err_norm.shape[0]:
            x_axis = t_eval
            plt.xlabel("Time at t+H (s)")
        else:
            x_axis = np.arange(pos_err_norm.shape[0])
            plt.xlabel("Start index t")
        plt.plot(x_axis, pos_err_norm, linewidth=2)
        plt.ylabel(f"|Δp| at horizon H={H} (m)")
        plt.title(f"[MoE3] Episode {ep} - {H}-step Ahead Position Error")
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
