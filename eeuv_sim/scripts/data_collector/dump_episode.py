import argparse
import h5py
import numpy as np

def print_episode_steps(h5_path, ep_idx, max_steps=20):
    with h5py.File(h5_path, "r") as f:
        # key = f"episode_{}".format(ep_idx)
        key = "episode_{}".format(ep_idx)
        if key not in f:
            print(f"[ERROR] {key} not found in {h5_path}")
            return
        grp = f[key]
        N = len(grp["time"])
        print(f"Episode {ep_idx}: length={N}, regime={int(grp['regime_id'][0])}, "
              f"fault_idx={int(grp['fault_index'][0])}, t_fail={float(grp['t_fail'][0])}")
        steps_to_show = min(N, max_steps)
        for i in range(steps_to_show):
            t   = float(grp["time"][i])
            pos = grp["position"][i].tolist()
            ori = grp["orientation"][i].tolist()
            lin = grp["linear_velocity"][i].tolist()
            ang = grp["angular_velocity"][i].tolist()
            u_c = grp["thrusts_cmd"][i].tolist()
            u_a = grp["thrusts_applied"][i].tolist()
            mask = grp["health_mask"][i].tolist()
            print(f"\nStep {i:04d}  t={t:.2f}s")
            print(f"  pos={pos}")
            print(f"  ori(wxyz)={ori}")
            print(f"  v_b={lin}, ω_b={ang}")
            print(f"  u_cmd={u_c}")
            print(f"  u_app={u_a}")
            print(f"  mask={mask}")

if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("h5", help="Path to HDF5 file")
    # ap.add_argument("--ep", type=int, default=0, help="Episode index to print")
    # ap.add_argument("--max_steps", type=int, default=20, help="Max steps to show")
    # args = ap.parse_args()
    # print_episode_steps("/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes.hdf5", 7, 600)
    print_episode_steps("/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_1.hdf5", 3, 600)
