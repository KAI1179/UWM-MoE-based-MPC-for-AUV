import os
import h5py
import rclpy
from data_collector_regimes import ROVDataCollectorRegimes

def main():
    rclpy.init()
    node = ROVDataCollectorRegimes(thrust_hz=1.0, data_hz=10.0)

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

    # regimen_plan = [
    #     (1, 120),
    #     (2, 120),
    #     (3, 120),
    #     (4, 120),
    # ]

    regimen_plan = [
        (1, 360),
        (2, 360),
        (3, 360),
    ]

    output_path = os.environ.get(
        "ROV_REGIME_DATA_H5",
        # "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_1.hdf5"
        # "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_10.hdf5"
        # "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_1.hdf5"
        # "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_8.hdf5"
        "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_18.hdf5"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, "w") as f:
        ep = 0
        for regime_id, num_eps in regimen_plan:
            for i in range(num_eps):
                node.get_logger().info(f"Collecting regime {regime_id} episode {i}...")
                data = node.collect_episode(regime_id=regime_id, max_steps=600, bounds=bounds)
                node.save_to_hdf5(data, f, ep)
                ep += 1

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
