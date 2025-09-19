#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_h5_trajectory.py
---------------------
读取 test_PPO_path_following.py 生成的 HDF5，绘制“参考轨迹 vs 实际轨迹（位置曲线）”。
- 若 HDF5 含有 extras/{waypoints, ref_traj}，则直接使用；
- 否则用 PPO_PathFollowingEnv_1.DEFAULT_WAYPOINTS 重建三次样条轨迹。

用法：
    python3 plot_h5_trajectory.py --h5 ./logs/ppo_rov_log_xxx.h5 --out myplot.png
"""
import argparse
import h5py
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def build_ref_from_waypoints(wp: np.ndarray, M: int = 600):
    from scipy.interpolate import CubicSpline
    t = np.linspace(0, 1, len(wp), dtype=np.float32)
    csx = CubicSpline(t, wp[:,0], bc_type='clamped')
    csy = CubicSpline(t, wp[:,1], bc_type='clamped')
    csz = CubicSpline(t, wp[:,2], bc_type='clamped')
    tt = np.linspace(0, 1, M, dtype=np.float32)
    ref = np.stack([csx(tt), csy(tt), csz(tt)], axis=1).astype(np.float32)
    return ref

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="HDF5 路径")
    ap.add_argument("--out", default="traj.png", help="输出图片路径")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        pos = f["position"][:]
        if "extras" in f and "ref_traj" in f["extras"]:
            ref = f["extras/ref_traj"][:]
            wp  = f["extras/waypoints"][:]
        else:
            from PPO_PathFollowingEnv_1 import DEFAULT_WAYPOINTS
            wp = DEFAULT_WAYPOINTS
            ref = build_ref_from_waypoints(wp, M=600)

    fig = plt.figure(figsize=(7,5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ref[:,0], ref[:,1], ref[:,2], label="Reference", linewidth=2)
    ax.scatter(wp[:,0], wp[:,1], wp[:,2], label="Waypoints", s=25)
    ax.plot(pos[:,0], pos[:,1], pos[:,2], label="Actual", linewidth=2)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Reference vs Actual Trajectory")
    fig.tight_layout()
    fig.savefig(args.out)
    print("saved:", args.out)

if __name__ == "__main__":
    main()
