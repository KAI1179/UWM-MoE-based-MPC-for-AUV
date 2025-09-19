import os
import h5py
import rclpy
from data_collector_diff import ROVDataCollector
from action_sampler import SmoothActionSampler

def main():
    rclpy.init()
    # node = ROVDataCollector(thrust_hz=1, data_hz=10)
    node = ROVDataCollector(thrust_hz=0.5, data_hz=10)
    sampler = SmoothActionSampler()
    # bounds = {
    #     "x": [-15, 15],
    #     "y": [-15, 15],
    #     "z": [-28, -0.2],
    # }
    bounds = {
        "x": [5, 35],
        "y": [-15, 15],
        "z": [-28, -0.2],
    }

    num_episodes = 660
    # output_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_trust1data10.hdf5"
    # output_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_trust1data10_1.hdf5"
    # output_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_trust05data10.hdf5"
    output_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_trust05data10_11.hdf5"
    # output_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_trust02data5.hdf5"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as f:
        for i in range(num_episodes):
            node.get_logger().info(f"Collecting episode {i}...")
            data = node.collect_episode(max_steps=600, bounds=bounds, sampler=sampler)
            node.save_to_hdf5(data, f, i)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
