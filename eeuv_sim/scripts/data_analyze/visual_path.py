import h5py
import numpy as np
import matplotlib.pyplot as plt


# 读取HDF5文件
def read_hdf5(file_path):
    with h5py.File(file_path, 'r+') as f:
        episodes = list(f.keys())  # 获取所有的episode
        data = {}
        for episode in episodes:
            episode_data = f[episode]
            data[episode] = {
                "time": np.array(episode_data["time"]),
                "position": np.array(episode_data["position"]),
            }
        return data


# 可视化ROV轨迹
def plot_trajectory(data):
    # 设定五种不同的颜色
    # colors = ['b', 'g', 'r', 'c', 'm']
    colors = [
        'b', 'g', 'r', 'c', 'm', 'y', 'k',  # 基本颜色
        '#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#33FFF5', '#A133FF'
    ]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制每条轨迹
    for episode_idx, (episode_key, episode_data) in enumerate(data.items()):
        if episode_idx >= len(colors):  # 如果轨迹多于五条，就循环使用颜色
            color = colors[episode_idx % len(colors)]
        else:
            color = colors[episode_idx]

        time = episode_data["time"]
        position = episode_data["position"]

        # 提取位置数据
        x = position[:, 0]
        y = position[:, 1]
        z = position[:, 2]

        # 绘制轨迹
        ax.plot(x, y, z, label=f"Episode {episode_idx}", color=color)

        if episode_idx == 10:
            break

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title("ROV Trajectories")
    ax.legend()
    plt.show()


# 主函数
def main():
    # file_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data.hdf5'
    file_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_trust1data10.hdf5'


    # 读取HDF5文件
    data = read_hdf5(file_path)

    # 可视化所有轨迹
    plot_trajectory(data)


if __name__ == "__main__":
    main()
