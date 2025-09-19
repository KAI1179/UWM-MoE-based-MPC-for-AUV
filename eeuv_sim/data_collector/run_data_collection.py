import os
import h5py
import rclpy
from data_collector import ROVDataCollector
from action_sampler import SmoothActionSampler

def main():
    rclpy.init()
    node = ROVDataCollector(hz=10)
    # node.debug_reset_and_thrust_test(thrust_value=15.0, wait_time=0.5)
    # exit()
    sampler = SmoothActionSampler()
    bounds = {
        "x": [-15, 15],
        "y": [-15, 15],
        "z": [-28, -0.2],
    }

    num_episodes = 1
    output_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data.hdf5"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as f:
        for i in range(num_episodes):
            node.get_logger().info(f"Collecting episode {i}...")
            data = node.collect_episode(max_steps=500, bounds=bounds, sampler=sampler)
            node.save_to_hdf5(data, f, i)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
