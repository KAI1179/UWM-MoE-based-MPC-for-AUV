import time
import numpy as np
import h5py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
# from thruster_to_wrench import ThrusterWrenchCalculator
from thruster_wrench_exchange import ThrusterWrenchCalculator
from tf_transformations import euler_from_quaternion

import random
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from eeuv_sim.srv import ResetToPose
from std_srvs.srv import Empty
from rclpy.task import Future


class ROVDataCollector(Node):
    def __init__(self, thrust_hz=2, data_hz=10):
        super().__init__('rov_data_collector')
        self.thrust_hz = thrust_hz
        self.data_hz = data_hz

        self.thrust_dt = 1.0 / self.thrust_hz
        self.data_dt = 1.0 / self.data_hz

        self.thrust_to_data_per = self.data_hz / self.thrust_hz ## per 5 data sample, 1 action sample

        self.publisher = self.create_publisher(Float32MultiArray, '/ucat/thruster_cmd', 1)
        self.reset_pub = self.create_publisher(Bool, '/ucat/reset', 1)
        self.state_sub = self.create_subscription(EntityState, '/ucat/state', self.state_callback, 1)
        self.state = None
        self.start_time = None
        # self.dt = 1.0 / self.rate_hz
        self.fast_forward = 1.0
        self.time_optimize_value = 0.0
        self.dt_optimize_gain = 0.05

        self.current_action = None

        self.thru_to_wrench = ThrusterWrenchCalculator('/home/xukai/ros2_ws/src/eeuv_sim/data/dynamics/BlueDynamics.yaml')

        self.reset_client = self.create_client(ResetToPose, '/reset_to_pose')
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('set_entity_state service not available!')
            # return False

        self.last_state_time = 0.0
        self.current_state_time = 0.0

    def state_callback(self, msg):
        self.state = msg
        # self.last_state_time = time.time()
        self.current_state_time = time.time()


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
        x = random.uniform(10, 30)
        y = random.uniform(-10, 10)
        z = random.uniform(20, 5)  # 注意 z 是负值，水下方向为负

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
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(f'Reset ROV to: x={x:.2f}, y={y:.2f}, z={z:.2f}')


        # 5. 稍作等待
        # time.sleep(3.0)
        self.state = None  # 清空旧状态
        timeout = time.time() + 3.0  # 最多等待3秒
        while self.state is None and time.time() < timeout:
            # rclpy.spin_once(self, timeout_sec=0.1)
            rclpy.spin_once(self)
        if self.state is None:
            self.get_logger().warn("No updated state received after reset!")

        return True

    def wait_time_optimizer(self, start_time, end_time):
        # 计算时间误差
        dt_error = (self.data_dt / self.fast_forward) - (end_time - start_time)

        # 如果时间误差小于 0.1，更新 time_optimize_value
        if dt_error < 0.1:
            self.time_optimize_value += self.dt_optimize_gain * dt_error
            self.time_optimize_value = np.clip(self.time_optimize_value, -0.1, 0.1)  # 限制时间优化值的更新范围

    def step(self, action):
        # 假设每个 step 需要时间控制
        # start_time = time.perf_counter()

        # 执行推力命令或控制逻辑
        msg = Float32MultiArray(data=action.tolist())
        self.publisher.publish(msg)


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

        rate = self.create_rate(self.data_hz)
        step = 0

        # rclpy.spin_once(self, timeout_sec=0.09)

        # rclpy.spin_once(self)


        self.last_state_time = self.current_state_time
        timeout = time.time() + 0.09

        while self.last_state_time == self.current_state_time and time.time() < timeout:
        # while self.last_state_time == last_seen and time.time() < timeout:
            self.last_state_time = self.current_state_time
            rclpy.spin_once(self)



        # if self.state is None:
        #     self.get_logger().error("Failed to get initial state.")
        #     return None

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
        action = np.zeros(8)

        while rclpy.ok() and step < max_steps:
            # rclpy.spin_once(self, timeout_sec=0.1)
            # rclpy.spin_once(self)

            # last_seen = self.last_state_time
            # timeout = time.time() + 0.05
            # while self.last_state_time == last_seen and time.time() < timeout:
            #     rclpy.spin_once(self)

            timeout = time.time() + 0.05
            while self.last_state_time == self.current_state_time and time.time() < timeout:
                self.last_state_time = self.current_state_time
                rclpy.spin_once(self)

            # self.state = rclpy.wait
            if self.state is None:
                continue
            self.last_state_time = self.current_state_time

            # 判断是否越界
            x, y, z = pos.x, pos.y, pos.z
            if bounds and not (bounds['x'][0] <= x <= bounds['x'][1] and
                               bounds['y'][0] <= y <= bounds['y'][1] and
                               bounds['z'][0] <= z <= bounds['z'][1]):
                self.get_logger().info(f'Boundary hit: x={x}, y={y}, z={z}')
                break
            step += 1
            start_time = time.perf_counter()
            if step % self.thrust_to_data_per == 1:
                action = sampler.random_sample()
                # self.step(action)
                # wrench = self.thru_to_wrench.compute_wrench(action)
                attitude_tmp = euler_from_quaternion(
                    [self.state.pose.orientation.x, self.state.pose.orientation.y, self.state.pose.orientation.z,
                     self.state.pose.orientation.w])
                wrench = np.concatenate(self.thru_to_wrench.compute_wrench_world(action, attitude_tmp))

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

            try:
                time.sleep((self.data_dt / self.fast_forward) + self.time_optimize_value)
            except Exception as e:
                self.get_logger().error(f"Invalid sleep time: {e}")
                time.sleep(self.data_dt / self.fast_forward)  # 在出现异常时使用默认时间

            # 获取结束时间并优化等待时间
            end_time = time.perf_counter()
            self.wait_time_optimizer(start_time, end_time)

            # rate.sleep()

        return data

    def save_to_hdf5(self, data, h5file, episode_idx):
        grp = h5file.create_group(f"episode_{episode_idx}")
        for k, v in data.items():
            grp.create_dataset(k, data=np.array(v), compression="gzip")