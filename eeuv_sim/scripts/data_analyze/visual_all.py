import h5py
import numpy as np


def read_hdf5_detailed(file_path):
    with h5py.File(file_path, 'r') as f:
        episodes = list(f.keys())
        data = {}

        for episode in episodes:
            episode_data = f[episode]

            # 读取所有字段
            time_data = np.array(episode_data["time"])
            position_data = np.array(episode_data["position"])  # [N, 3]
            orientation_data = np.array(episode_data["orientation"])  # [N, 4] (w, x, y, z)
            linear_vel_data = np.array(episode_data["linear_velocity"])  # [N, 3]
            angular_vel_data = np.array(episode_data["angular_velocity"])  # [N, 3]
            thrusts_data = np.array(episode_data["thrusts"])  # [N, ?] 你动作维度多少就是多少
            wrench_data = np.array(episode_data["wrench"])  # [N, 6] (force + torque)

            data[episode] = {
                "time": time_data,
                "position": position_data,
                "orientation": orientation_data,
                "linear_velocity": linear_vel_data,
                "angular_velocity": angular_vel_data,
                "thrusts": thrusts_data,
                "wrench": wrench_data,
            }

            print(f"Episode {episode}:")
            for i in range(len(time_data)):
                print(f" Step {i}:")
                print(f"   Time: {time_data[i]:.4f}s")
                print(
                    f"   Position: x={position_data[i, 0]:.3f}, y={position_data[i, 1]:.3f}, z={position_data[i, 2]:.3f}")
                ori = orientation_data[i]
                print(f"   Orientation (w,x,y,z): {ori[0]:.4f}, {ori[1]:.4f}, {ori[2]:.4f}, {ori[3]:.4f}")
                lin_v = linear_vel_data[i]
                print(f"   Linear Velocity: x={lin_v[0]:.3f}, y={lin_v[1]:.3f}, z={lin_v[2]:.3f}")
                ang_v = angular_vel_data[i]
                print(f"   Angular Velocity: x={ang_v[0]:.3f}, y={ang_v[1]:.3f}, z={ang_v[2]:.3f}")
                thrust = thrusts_data[i]
                print(f"   Thrusts: {thrust}")
                wrench = wrench_data[i]
                print(f"   Wrench (Fx,Fy,Fz,Tx,Ty,Tz): {wrench}")

            print("-" * 40)

        return data


def main():
    # file_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data.hdf5'
    file_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_trust1data10.hdf5'
    # file_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_trust02data5.hdf5'
    # file_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/comparison_methods/results/mpc_rov_log.h5'
    data = read_hdf5_detailed(file_path)


if __name__ == "__main__":
    main()
