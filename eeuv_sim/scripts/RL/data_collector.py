import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float32MultiArray

import numpy as np
import h5py
import time
import os

MAX_STEPS = 500
EPISODES = 1  # Set >1 when ready for full run
SAMPLE_DT = 0.1  # 10 Hz
OUTPUT_FILE = 'rov_dataset.hdf5'


class ROVDataCollector(Node):
    def __init__(self):
        super().__init__('rov_data_collector')

        self.state_sub = self.create_subscription(
            Odometry, '/ucat/odom', self.odom_callback, 10)

        self.action_sub = self.create_subscription(
            Float32MultiArray, '/ucat/thruster_output', self.action_callback, 10)

        self.timer = self.create_timer(SAMPLE_DT, self.timer_callback)

        # Initialize episode-related storage
        self.episode_id = 0
        self.reset_episode()

        # File setup
        if os.path.exists(OUTPUT_FILE):
            self.dataset = h5py.File(OUTPUT_FILE, 'a')
        else:
            self.dataset = h5py.File(OUTPUT_FILE, 'w')

        self.get_logger().info('ROV data collector initialized.')

    def reset_episode(self):
        self.state_buffer = []
        self.action_buffer = []
        self.time_buffer = []

        self.current_action = np.zeros(8)
        self.current_state = None
        self.start_time = None
        self.step_counter = 0
        self.done = False

    def odom_callback(self, msg):
        # Extract state
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular

        # 19D state
        state = np.array([
            position.x, position.y, position.z,
            orientation.x, orientation.y, orientation.z, orientation.w,
            linear.x, linear.y, linear.z,
            angular.x, angular.y, angular.z
        ])
        self.current_state = state

    def action_callback(self, msg):
        # Assume 8 thrusters
        self.current_action = np.array(msg.data[:8])

    def timer_callback(self):
        if self.done or self.current_state is None:
            return

        now = time.time()
        if self.start_time is None:
            self.start_time = now

        rel_time = now - self.start_time

        # Save trajectory
        self.state_buffer.append(self.current_state.copy())
        self.action_buffer.append(self.current_action.copy())
        self.time_buffer.append(rel_time)
        self.step_counter += 1

        # Early stop if out of region
        x, y, z = self.current_state[0], self.current_state[1], self.current_state[2]
        if abs(x) > 15.0 or abs(y) > 15.0 or z > -0.2 or z < -28 or self.step_counter >= MAX_STEPS:
            self.finish_episode()

    def finish_episode(self):
        self.done = True
        group = self.dataset.create_group(f'episode_{self.episode_id}')
        group.create_dataset('time', data=np.array(self.time_buffer))
        group.create_dataset('state', data=np.array(self.state_buffer))
        group.create_dataset('action', data=np.array(self.action_buffer))

        self.get_logger().info(f'[Episode {self.episode_id}] Recorded {len(self.time_buffer)} steps.')

        self.episode_id += 1
        if self.episode_id >= EPISODES:
            self.get_logger().info('All episodes finished. Shutting down.')
            self.dataset.close()
            rclpy.shutdown()
        else:
            self.reset_episode()


def main(args=None):
    rclpy.init()
    node = ROVDataCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
