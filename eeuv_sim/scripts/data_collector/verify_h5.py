# verify_h5.py
import argparse, h5py, numpy as np, textwrap

FIELDS = [
    "time","position","orientation","linear_velocity","angular_velocity",
    "thrusts_cmd","thrusts_applied","health_mask","regime_id","fault_index","t_fail"
]

def ep_summary(ep):
    out = {}
    n = len(ep["time"])
    out["length"] = n
    t = ep["time"][()]
    out["t0"], out["tN"] = float(t[0]), float(t[-1])
    if len(t) > 1:
        dt = np.diff(t)
        out["dt_med"] = float(np.median(dt))
        out["hz_est"] = 1.0 / out["dt_med"]
    else:
        out["dt_med"] = np.nan; out["hz_est"] = np.nan

    # position bounds
    pos = ep["position"][()]            # (N,3)
    out["pos_min"] = pos.min(axis=0).tolist()
    out["pos_max"] = pos.max(axis=0).tolist()

    # thrust mismatch
    u_cmd = ep["thrusts_cmd"][()]       # (N,8)
    u_app = ep["thrusts_applied"][()]   # (N,8)
    out["mean_abs_mismatch"] = np.mean(np.abs(u_cmd - u_app), axis=0).round(3).tolist()
    out["mismatch_overall"] = float(np.mean(np.abs(u_cmd - u_app)))

    # regime / fault
    regime = int(ep["regime_id"][()][0]) if len(ep["regime_id"])>0 else -1
    out["regime_id"] = regime
    fault_idx = int(ep["fault_index"][()][0]) if len(ep["fault_index"])>0 else -1
    t_fail = float(ep["t_fail"][()][0]) if len(ep["t_fail"])>0 else -1.0
    out["fault_idx"] = fault_idx
    out["t_fail"] = t_fail

    # health mask last sample（是否保持为0）
    hm_last = ep["health_mask"][()][-1] if len(ep["health_mask"])>0 else np.ones(8)
    out["health_mask_last"] = hm_last.astype(float).round(1).tolist()
    return out

def main(h5_path):
    with h5py.File(h5_path, "r") as f:
        eps = sorted([k for k in f.keys() if k.startswith("episode_")],
                     key=lambda s: int(s.split("_")[1]))
        print(f"[INFO] Episodes found: {len(eps)}")
        for k in eps:
            ep = f[k]
            # 字段完整性
            missing = [fld for fld in FIELDS if fld not in ep]
            if missing:
                print(f"[WARN] {k} missing fields: {missing}")
            s = ep_summary(ep)
            print(textwrap.dedent(f"""
            == {k} ==
              samples: {s['length']}   span: {s['t0']:.2f} → {s['tN']:.2f}s   dt_med≈{s['dt_med']:.3f}s  (~{s['hz_est']:.1f} Hz)
              regime: {s['regime_id']}   fault_idx: {s['fault_idx']}   t_fail: {s['t_fail']:.2f}
              pos min: {np.array(s['pos_min']).round(2).tolist()}
              pos max: {np.array(s['pos_max']).round(2).tolist()}
              mean|u_cmd-u_app| per thruster: {s['mean_abs_mismatch']}
              overall mismatch: {s['mismatch_overall']:.3f}
              last health mask: {s['health_mask_last']}
            """).strip())

if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("h5", help="path to HDF5 (e.g., rov_data_regimes.hdf5)")
    # args = ap.parse_args()
    # main("/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes.hdf5")
    main("/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_1.hdf5")
