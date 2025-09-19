import h5py
import numpy as np
import matplotlib.pyplot as plt


def analyze_thrusts(file_path):
    all_thrusts = []

    with h5py.File(file_path, 'r') as f:
        for episode_key in f.keys():
            thrusts = np.array(f[episode_key]["thrusts"])  # shape: [T, 8]
            all_thrusts.append(thrusts)

    # 将所有数据拼接起来 [N_total_steps, 8]
    all_thrusts = np.vstack(all_thrusts)  # shape: [total_steps, 8]

    # 计算每个推力通道的统计量
    min_vals = np.min(all_thrusts, axis=0)
    max_vals = np.max(all_thrusts, axis=0)
    mean_vals = np.mean(all_thrusts, axis=0)
    std_vals = np.std(all_thrusts, axis=0)

    # 打印统计信息
    print("Thrust Statistics (per channel):")
    for i in range(8):
        print(f"  Thrust {i}: Min={min_vals[i]:.2f}, Max={max_vals[i]:.2f}, Mean={mean_vals[i]:.2f}, Std={std_vals[i]:.2f}")

    # 可视化 - 箱线图
    plt.figure(figsize=(10, 6))
    plt.boxplot([all_thrusts[:, i] for i in range(8)], labels=[f'T{i}' for i in range(8)])
    plt.title("Distribution of Each Thrust Channel")
    plt.xlabel("Thrust Channel")
    plt.ylabel("Thrust Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # file_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data.hdf5'
    # file_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_random.hdf5'
    file_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_trust1data10.hdf5'
    analyze_thrusts(file_path)


if __name__ == "__main__":
    main()
