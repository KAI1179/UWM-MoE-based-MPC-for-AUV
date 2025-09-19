import time
import numpy as np
import h5py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from gazebo_msgs.msg import EntityState
from thruster_to_wrench import ThrusterWrenchCalculator
import random
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from eeuv_sim.srv import ResetToPose
from std_srvs.srv import Empty


class ROVDataCollector(Node):
    def __init__(self, hz=10):
        super().__init__('rov_data_collector')
        self.rate_hz = hz
        self.publisher = self.create_publisher(Float32MultiArray, '/ucat/thruster_cmd', 1)
        self.reset_pub = self.create_publisher(Bool, '/ucat/reset', 1)
        self.state_sub = self.create_subscription(EntityState, '/ucat/state', self.state_callback, 1)
        self.state = None
        self.start_time = None
        self.dt = 1.0 / self.rate_hz
        self.fast_forward = 1.0
        self.time_optimize_value = 0.0
        self.dt_optimize_gain = 0.05

        self.thru_to_wrench = ThrusterWrenchCalculator('/home/xukai/ros2_ws/src/eeuv_sim/data/dynamics/BlueDynamics.yaml')

        self.reset_client = self.create_client(ResetToPose, '/reset_to_pose')
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('set_entity_state service not available!')
            return False


    def state_callback(self, msg):
        self.state = msg


    # def debug_reset_and_thrust_test(self, thrust_value=15.0, wait_time=0.5):
    #     self.get_logger().info("=== Debug: Reset + Thrust Test ===")
    #
    #     # Step 1: Reset Environment
    #     success = self.reset_environment()
    #     if not success:
    #         self.get_logger().error("Reset failed.")
    #         return
    #
    #     # Step 2: Unpause Gazebo physics
    #     self.get_logger().info("Unpausing Gazebo physics...")
    #     unpause_client = self.create_client(Empty, '/unpause_physics')
    #     if unpause_client.wait_for_service(timeout_sec=2.0):
    #         future = unpause_client.call_async(Empty.Request())
    #         rclpy.spin_until_future_complete(self, future)
    #         self.get_logger().info("Gazebo physics unpaused.")
    #     else:
    #         self.get_logger().warn("Unpause service unavailable.")
    #
    #     # Step 3: Wait for state update
    #     self.state = None
    #     for _ in range(10):
    #         rclpy.spin_once(self, timeout_sec=0.1)
    #         if self.state is not None:
    #             break
    #     if self.state is None:
    #         self.get_logger().error("No state received after reset.")
    #         return
    #
    #     # Record state before thrust
    #     pos0 = self.state.pose.position
    #     vel0 = self.state.twist.linear
    #     self.get_logger().info(f"Initial Position: x={pos0.x:.3f}, y={pos0.y:.3f}, z={pos0.z:.3f}")
    #     self.get_logger().info(f"Initial Velocity: x={vel0.x:.3f}, y={vel0.y:.3f}, z={vel0.z:.3f}")
    #
    #     # Step 4: Apply thrust
    #     action = np.ones(8) * thrust_value
    #     msg = Float32MultiArray(data=action.tolist())
    #     self.publisher.publish(msg)
    #     self.get_logger().info(f"Sent thrust: {thrust_value} on all channels")
    #
    #     # Step 5: Wait for ROV to respond
    #     time.sleep(wait_time)
    #
    #     # Step 6: Read state after thrust
    #     rclpy.spin_once(self, timeout_sec=0.2)
    #     if self.state is not None:
    #         pos1 = self.state.pose.position
    #         vel1 = self.state.twist.linear
    #         self.get_logger().info(f"After {wait_time}s -> Position: x={pos1.x:.3f}, y={pos1.y:.3f}, z={pos1.z:.3f}")
    #         self.get_logger().info(f"After {wait_time}s -> Velocity: x={vel1.x:.3f}, y={vel1.y:.3f}, z={vel1.z:.3f}")
    #
    #         dx = pos1.x - pos0.x
    #         dy = pos1.y - pos0.y
    #         dz = pos1.z - pos0.z
    #         self.get_logger().info(f"Δ Position: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
    #     else:
    #         self.get_logger().warn("No state received after thrust.")

    def reset_environment(self):
        """
        Resets the environment: clears sim state, sets random ROV position.
        """
        req = ResetToPose.Request()
        # 0. 清除历史状态
        # reset_msg = Bool()
        # reset_msg.data = True
        # self.reset_pub.publish(reset_msg)

        # 1. 随机位置（在合法边界内，保持在中心附近）
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        z = random.uniform(25, 5)  # 注意 z 是负值，水下方向为负

        # 2. 姿态可保持固定或小范围扰动
        # qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0  # 无旋转
        # 若希望扰动：可以随机 yaw
        import tf_transformations
        yaw = random.uniform(-3.14, 3.14)
        # qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0, 0, yaw)

        req.x = x
        req.y = y
        req.z = z
        req.roll = 0.0
        req.pitch = 0.0
        req.yaw = yaw

        future = self.reset_client.call_async(req)
        # rclpy.spin_until_future_complete(self, future)

        # # 3. 设置实体状态
        # state = EntityState()
        # state.name = 'Blue'  # 替换为你 model 的名字
        # state.pose.position.x = x
        # state.pose.position.y = y
        # state.pose.position.z = z
        # state.pose.orientation.x = qx
        # state.pose.orientation.y = qy
        # state.pose.orientation.z = qz
        # state.pose.orientation.w = qw
        #
        # # 4. 等待服务可用并调用
        # client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        # if not client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().error('set_entity_state service not available!')
        #     return False

        # req = SetEntityState.Request()
        # req.state = state
        # future = client.call_async(req)
        # rclpy.spin_until_future_complete(self, future)
        self.get_logger().info(f'Reset ROV to: x={x:.2f}, y={y:.2f}, z={z:.2f}')
        # if future.result() is not None:
        #     self.get_logger().info(f'Reset ROV to: x={x:.2f}, y={y:.2f}, z={z:.2f}')
        # else:
        #     self.get_logger().error('Failed to set ROV pose!')

        # 5. 稍作等待
        # time.sleep(3.0)
        self.state = None  # 清空旧状态
        timeout = time.time() + 3.0  # 最多等待3秒
        while self.state is None and time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.state is None:
            self.get_logger().warn("No updated state received after reset!")

        return True

    def wait_time_optimizer(self, start_time, end_time):
        # 计算时间误差
        dt_error = (self.dt / self.fast_forward) - (end_time - start_time)

        # 如果时间误差小于 0.1，更新 time_optimize_value
        if dt_error < 0.1:
            self.time_optimize_value += self.dt_optimize_gain * dt_error
            self.time_optimize_value = np.clip(self.time_optimize_value, -0.1, 0.1)  # 限制时间优化值的更新范围


    # def step(self, action):
    #     msg = Float32MultiArray(data=action.tolist())
    #     self.publisher.publish(msg)

    def step(self, action):
        # 假设每个 step 需要时间控制
        start_time = time.perf_counter()

        # 执行推力命令或控制逻辑
        msg = Float32MultiArray(data=action.tolist())
        self.publisher.publish(msg)

        try:
            time.sleep((self.dt / self.fast_forward) + self.time_optimize_value)
        except Exception as e:
            self.get_logger().error(f"Invalid sleep time: {e}")
            time.sleep(self.dt / self.fast_forward)  # 在出现异常时使用默认时间

        # 获取结束时间并优化等待时间
        end_time = time.perf_counter()
        self.wait_time_optimizer(start_time, end_time)

        # 调整等待时间，确保时间间隔


    def collect_episode(self, max_steps=500, bounds=None, sampler=None):
        _ = self.reset_environment()
        data = {
            "time": [],
            "position": [],
            "orientation": [],
            "linear_velocity": [],
            "angular_velocity": [],
            "thrusts": [],
            "wrench": [],
        }

        rate = self.create_rate(self.rate_hz)
        step = 0

        rclpy.spin_once(self, timeout_sec=0.09)

        if self.state is not None:
            pos = self.state.pose.position
            ori = self.state.pose.orientation
            lin = self.state.twist.linear
            ang = self.state.twist.angular
        data["position"].append([pos.x, pos.y, pos.z])
        data["orientation"].append([ori.w, ori.x, ori.y, ori.z])
        data["linear_velocity"].append([lin.x, lin.y, lin.z])
        data["angular_velocity"].append([ang.x, ang.y, ang.z])
        data["thrusts"].append([0.0] * 8)  # or last sampled
        data["wrench"].append([0.0] * 6)
        self.start_time = time.perf_counter()

        while rclpy.ok() and step < max_steps:
            rclpy.spin_once(self, timeout_sec=0.09)
            if self.state is None:
                continue

            # 判断是否越界
            x, y, z = pos.x, pos.y, pos.z
            if bounds and not (bounds['x'][0] <= x <= bounds['x'][1] and
                               bounds['y'][0] <= y <= bounds['y'][1] and
                               bounds['z'][0] <= z <= bounds['z'][1]):
                self.get_logger().info(f'Boundary hit: x={x}, y={y}, z={z}')
                break

            action = sampler.smooth_sample()

            wrench = self.thru_to_wrench.compute_wrench(action)

            pos = self.state.pose.position
            ori = self.state.pose.orientation
            lin = self.state.twist.linear
            ang = self.state.twist.angular

            now = time.perf_counter() - self.start_time
            data["time"].append(now)
            data["position"].append([x, y, z])
            data["orientation"].append([ori.w, ori.x, ori.y, ori.z])
            data["linear_velocity"].append([lin.x, lin.y, lin.z])
            data["angular_velocity"].append([ang.x, ang.y, ang.z])
            data["thrusts"].append(action.tolist())
            data["wrench"].append(wrench.tolist())

            print("Timestep: {}".format(step))
            self.step(action)
            step += 1
            # rate.sleep()

        return data

    def save_to_hdf5(self, data, h5file, episode_idx):
        grp = h5file.create_group(f"episode_{episode_idx}")
        for k, v in data.items():
            grp.create_dataset(k, data=np.array(v), compression="gzip")