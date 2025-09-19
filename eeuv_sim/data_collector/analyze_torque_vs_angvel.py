import h5py
import numpy as np
import matplotlib.pyplot as plt

def analyze_torque_response(h5_path, episode_idx=0, plot=True):
    with h5py.File(h5_path, "r") as f:
        grp = f[f"episode_{episode_idx}"]

        wrench = grp["wrench"][:]               # (N, 6)
        angular_vel = grp["angular_velocity"][:]  # (N, 3)

    tau_t = wrench[:-1, 3:]        # Torque: Tx, Ty, Tz (N-1, 3)
    omega_t = angular_vel[:-1]     # Angular velocity at time t (N-1, 3)
    omega_tp1 = angular_vel[1:]    # Angular velocity at time t+1
    domega = omega_tp1 - omega_t   # Δangular velocity (N-1, 3)

    print(f"\n🔍 Correlation between torque (τₜ) and Δangular velocity (ωₜ₊₁ - ωₜ):")
    for i, axis in enumerate(['x', 'y', 'z']):
        corr = np.corrcoef(tau_t[:, i], domega[:, i])[0, 1]
        print(f"  {axis.upper()}-axis: correlation = {corr:.4f}")

    if plot:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        for i, axis in enumerate(['x', 'y', 'z']):
            axs[i].plot(tau_t[:, i], label=f'Torque {axis}')
            axs[i].plot(domega[:, i], label=f'Δω {axis}')
            axs[i].set_ylabel(axis)
            axs[i].legend()
            axs[i].grid(True)
        plt.suptitle("Torque vs ΔAngular Velocity (per axis)")
        plt.xlabel("Time Step")
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    analyze_torque_response("/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data.hdf5", episode_idx=0)
