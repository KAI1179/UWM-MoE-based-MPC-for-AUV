import h5py
import numpy as np


# 读取HDF5文件并显示每条轨迹每个step的时间戳
def read_hdf5(file_path):
    with h5py.File(file_path, 'r+') as f:
        episodes = list(f.keys())  # 获取所有的episode
        data = {}

        # 遍历所有的episode并读取数据
        for episode in episodes:
            episode_data = f[episode]
            time_data = np.array(episode_data["time"])  # 获取时间戳
            data[episode] = {
                "time": time_data,
                "position": np.array(episode_data["position"]),
            }

            # 输出每个step的时间戳
            print(f"Episode {episode}:")
            for idx, time_stamp in enumerate(time_data):
                print(f"  Step {idx}: Time = {time_stamp}")

        return data


# 主函数
def main():
    file_path = '/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data.hdf5'

    # 读取HDF5文件并查看每个step的时间戳
    data = read_hdf5(file_path)


if __name__ == "__main__":
    main()
