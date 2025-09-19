#!/usr/bin/env python3
import os, argparse, h5py, numpy as np

def median_dt(times):
    if len(times) < 2:
        return 0.1
    d = np.diff(times.astype(float))
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return 0.1
    return float(np.median(d))

def make_health_mask_gt(time, regime_id, fault_index, t_fail, n_thrusters=8, delay_s=0.3):
    """
    生成基础 GT 掩码：故障通道在 t >= t_fail + delay_s 置 0，其余为 1。
    非故障工况或 fault_index < 0 时，全 1。
    """
    N = len(time)
    gt = np.ones((N, n_thrusters), dtype=np.float32)

    if regime_id == 4 and (fault_index is not None) and (fault_index >= 0) and np.isfinite(t_fail):
        onset = t_fail + delay_s
        # 找到第一个 time >= onset 的索引
        idx = np.searchsorted(time.astype(float), onset, side="left")
        idx = int(np.clip(idx, 0, N))
        if idx < N:
            gt[idx:, fault_index] = 0.0
    return gt

def apply_min_dwell(binary_series, min_dwell_steps=5):
    """
    对单个二值序列应用“最小驻留”：移除短暂尖峰/短暂空洞，使得每段至少 min_dwell_steps。
    思路：扫描运行长度，长度 < min_dwell 的段被覆盖成邻近值。
    """
    x = binary_series.astype(np.int32).copy()  # 0/1
    N = len(x)
    if N == 0 or min_dwell_steps <= 1:
        return x.astype(np.float32)

    i = 0
    while i < N:
        j = i
        while j < N and x[j] == x[i]:
            j += 1
        run_len = j - i
        if run_len < min_dwell_steps:
            # 将这段覆盖为“前一个值”或“后一个值”，优先用前一个（若 i>0），否则用后一个
            fill_val = x[i-1] if i > 0 else (x[j] if j < N else x[i])
            x[i:j] = fill_val
            # 不前移 i，继续合并后的段
        i = j
    return x.astype(np.float32)

def hysteresis_min_dwell_mask(gt_mask, min_dwell_steps=5):
    """
    对 (N,8) 的 GT 掩码逐通道应用最小驻留（滞回可在证据层做，这里简化为驻留稳定化）。
    """
    N, M = gt_mask.shape
    out = np.zeros_like(gt_mask, dtype=np.float32)
    for m in range(M):
        out[:, m] = apply_min_dwell(gt_mask[:, m], min_dwell_steps=min_dwell_steps)
    return out

def compute_is_saturated(u_cmd, umax=20.0, delta=1.0):
    """|u_cmd| >= umax - delta 视为饱和。"""
    return (np.abs(u_cmd) >= (umax - delta)).astype(np.float32)

def compute_du_abs(u_cmd, u_app):
    return np.abs(u_cmd - u_app).astype(np.float32)

def compute_kappa(time, v_lin, u_app, win_steps=5, eps=1e-3):
    """
    kappa[t] = ||Δv|| / (||u_app|| + eps)，Δv 为滑窗内速度变化量的范数。
    v_lin: (N,3), u_app: (N,8)
    """
    N = len(time)
    kappa = np.zeros(N, dtype=np.float32)
    v = v_lin.astype(float)
    u = u_app.astype(float)
    for t in range(N):
        t0 = max(0, t - win_steps + 1)
        dv = v[t] - v[t0]
        num = np.linalg.norm(dv)
        den = np.linalg.norm(u[t]) + eps
        kappa[t] = float(num / den)
    return kappa

def copy_episode_with_labels(src_grp, dst_file, ep_idx,
                             delay_s=0.3, min_dwell_s=0.5,
                             umax=20.0, delta_sat=1.0, kappa_win_s=0.5):
    # 读取基础数据
    time = src_grp["time"][()]
    pos  = src_grp["position"][()]
    ori  = src_grp["orientation"][()]
    lin  = src_grp["linear_velocity"][()]
    ang  = src_grp["angular_velocity"][()]
    ucmd = src_grp["thrusts_cmd"][()]
    uapp = src_grp["thrusts_applied"][()]
    hmask= src_grp["health_mask"][()]
    reg  = src_grp["regime_id"][()]
    fidx = src_grp["fault_index"][()]
    tfail= src_grp["t_fail"][()]

    # 形状检查
    N = len(time)
    assert ucmd.shape == (N, 8) and uapp.shape == (N, 8), "thrust shape mismatch"

    # 估计 dt 与步长
    dt_med = median_dt(time)
    dwell_steps = max(1, int(round(min_dwell_s / max(dt_med, 1e-6))))
    kappa_win_steps = max(1, int(round(kappa_win_s / max(dt_med, 1e-6))))

    # episode 级标签（本数据集 reg/fidx/tfail 是定值；容错写法：取第一个）
    regime_id = int(reg[0]) if len(reg) > 0 else -1
    fault_idx = int(fidx[0]) if len(fidx) > 0 else -1
    t_fail    = float(tfail[0]) if len(tfail) > 0 else -1.0

    # 生成 health mask GT
    gt = make_health_mask_gt(time, regime_id, fault_idx, t_fail, n_thrusters=8, delay_s=delay_s)
    gt_hys = hysteresis_min_dwell_mask(gt, min_dwell_steps=dwell_steps)

    # 辅助特征
    is_sat = compute_is_saturated(ucmd, umax=umax, delta=delta_sat)
    du_abs = compute_du_abs(ucmd, uapp)
    kappa  = compute_kappa(time, lin, uapp, win_steps=kappa_win_steps, eps=1e-3)
    regime_step = np.full((N,), regime_id, dtype=np.int32)

    # 写入新文件：复制原数据 + 写 labels/*
    ep_name = f"episode_{ep_idx}"
    dst_grp = dst_file.create_group(ep_name)

    # 原始字段（保持与你的数据结构一致）
    dst_grp.create_dataset("time",              data=time, compression="gzip")
    dst_grp.create_dataset("position",          data=pos,  compression="gzip")
    dst_grp.create_dataset("orientation",       data=ori,  compression="gzip")
    dst_grp.create_dataset("linear_velocity",   data=lin,  compression="gzip")
    dst_grp.create_dataset("angular_velocity",  data=ang,  compression="gzip")
    dst_grp.create_dataset("thrusts_cmd",       data=ucmd, compression="gzip")
    dst_grp.create_dataset("thrusts_applied",   data=uapp, compression="gzip")
    dst_grp.create_dataset("health_mask",       data=hmask,compression="gzip")
    dst_grp.create_dataset("regime_id",         data=reg,  compression="gzip")
    dst_grp.create_dataset("fault_index",       data=fidx, compression="gzip")
    dst_grp.create_dataset("t_fail",            data=tfail,compression="gzip")

    # labels 子组
    lab = dst_grp.create_group("labels")
    lab.create_dataset("health_mask_gt",       data=gt,      compression="gzip")
    lab.create_dataset("health_mask_gt_hys",   data=gt_hys,  compression="gzip")
    lab.create_dataset("is_saturated",         data=is_sat,  compression="gzip")
    lab.create_dataset("du_abs",               data=du_abs,  compression="gzip")
    lab.create_dataset("kappa",                data=kappa,   compression="gzip")
    lab.create_dataset("regime_step",          data=regime_step, compression="gzip")

    # 元信息（写成 attributes，便于追踪）
    lab.attrs["delay_s"]       = float(delay_s)
    lab.attrs["min_dwell_s"]   = float(min_dwell_s)
    lab.attrs["umax"]          = float(umax)
    lab.attrs["delta_sat"]     = float(delta_sat)
    lab.attrs["dt_med"]        = float(dt_med)
    lab.attrs["kappa_win_s"]   = float(kappa_win_s)
    lab.attrs["kappa_win_steps"]= int(kappa_win_steps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path",
                    # default="/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes.hdf5",
                    # default="/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_10.hdf5",
                    default="/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_1.hdf5",
                    help="input HDF5 path")
    ap.add_argument("--out", dest="out_path", default="",
                    help="output HDF5 path (default: *_labeled.hdf5 next to input)")
    ap.add_argument("--delay", type=float, default=0.3, help="failure label delay (s)")
    ap.add_argument("--dwell", type=float, default=0.5, help="min dwell duration (s)")
    ap.add_argument("--umax",  type=float, default=20.0, help="thruster max (N)")
    ap.add_argument("--delta_sat", type=float, default=1.0, help="sat margin (N)")
    ap.add_argument("--kappa_win", type=float, default=0.5, help="kappa window (s)")
    args = ap.parse_args()

    in_path = args.in_path
    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    if args.out_path:
        out_path = args.out_path
    else:
        base, ext = os.path.splitext(in_path)
        out_path = base + "_labeled" + (ext if ext else ".hdf5")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # 开始处理
    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
        # 遍历 episodes
        ep_keys = sorted([k for k in fin.keys() if k.startswith("episode_")],
                         key=lambda s: int(s.split("_")[1]))
        print(f"[INFO] episodes found: {len(ep_keys)}")
        for k in ep_keys:
            ep_idx = int(k.split("_")[1])
            print(f"[INFO] labeling {k} ...")
            copy_episode_with_labels(
                fin[k], fout, ep_idx,
                delay_s=args.delay, min_dwell_s=args.dwell,
                umax=args.umax, delta_sat=args.delta_sat, kappa_win_s=args.kappa_win
            )
        # 文件级元信息
        fout.attrs["source_file"] = os.path.abspath(in_path)
        fout.attrs["label_maker"] = "label_make.py"
        fout.attrs["version"]     = "v1.0"
        fout.attrs["notes"]       = "Original data + labels/* per episode"

    print(f"[OK] Labeled file saved to: {out_path}")

if __name__ == "__main__":
    main()
