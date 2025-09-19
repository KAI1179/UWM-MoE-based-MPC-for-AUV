#!/usr/bin/python3
"""
Script to calculate thruster movement for glider AUV in Gazebo
@author: Yuya Hamamatsu
@contact: yuya.hamamatsu@taltech.ee
"""
import os
import yaml
import math
import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import WrenchStamped, Twist
from std_msgs.msg import Float32MultiArray, Bool
from gazebo_msgs.msg import EntityState

from tf_transformations import euler_from_quaternion, quaternion_from_euler

class MoveThruster(Node):
    def __init__(self):
        super().__init__('moveThruster')
        self.declare_parameter('robot_model', 'UCAT')
        self.robot_model = self.get_parameter('robot_model').value

        if self.robot_model == 'UCAT':
            self.declare_parameter('yaml_dynamics', 'UCATDynamics.yaml')
        elif self.robot_model == 'LAUV':
            self.declare_parameter('yaml_dynamics', 'LAUVDynamics.yaml')

        yaml_dynamics = self.get_parameter('yaml_dynamics').value
        parameters_from_yaml = os.path.join(
                get_package_share_directory('eeuv_sim'),
                'data', 'dynamics',
                yaml_dynamics
                )
        self.declare_parameter('rl_setting_yaml', 'rl_setting.yaml')
        yaml_rl = self.get_parameter('rl_setting_yaml').value
        parameters_rl = os.path.join(
                get_package_share_directory('eeuv_sim'),
                'data', 'rl_setting',
                yaml_rl
                )
        with open(parameters_from_yaml, 'r') as file:
            dynamics_parameters = yaml.load(file, Loader=yaml.FullLoader)
        with open(parameters_rl, 'r') as file:
            rl_parameters = yaml.load(file, Loader=yaml.FullLoader)

        self.fast_forward = rl_parameters["rl"]["fast_forward"]

        _COG = dynamics_parameters["dynamics"]["geometry"]["COG"]
        _COB = dynamics_parameters["dynamics"]["geometry"]["COB"]
        self.cog = _COG


        self.thruster_dynamics_params = dynamics_parameters["thrusterDynamics"]

        self.number_of_thrusters = self.thruster_dynamics_params["NumberOfThrusters"]
        self.thruster_positions = self.thruster_dynamics_params["thrusterPositions"]
        self.thruster_Directions = self.thruster_dynamics_params["thrusterDirections"]
        self.thruster_limits = self.thruster_dynamics_params["thrustlimits"]

        self.dt = 0.1
        self.cmd_array = np.zeros(self.number_of_thrusters)

        self.thruster_cmd = np.zeros(self.number_of_thrusters)
        self.reset_flag = False
        self.attitude = [0, 0, 0]

        self.wrench_pub = self.create_publisher(WrenchStamped, '/ucat/force_thrust', 10)

        self.reset_flag_sub = self.create_subscription(Bool, '/ucat/reset', self.reset_flag_callback, 10)
        self.thruster_cmd_sub = self.create_subscription(Float32MultiArray, '/ucat/thruster_cmd', self.thruster_cmd_callback, 10)
        self.state_sub = self.create_subscription(EntityState, '/ucat/state', self.state_callback, 10)

        self.create_timer(self.dt / self.fast_forward, self.move_thruster)

    def reset_flag_callback(self, msg):
        """
        Callback function for reset flag.
        """
        self.reset_flag = msg.data

    def thruster_cmd_callback(self, msg):
        """
        Callback function for thruster command.
        """
        self.thruster_cmd = msg.data
        self.cmd_array = np.array(self.thruster_cmd.tolist())
        # self._logger.info(f'Thruster command: {self.cmd_array}')
        # self._logger.info(f'Raw msg: {self.thruster_cmd.tolist()}')

    def state_callback(self, msg):
        """
        Callback function for state.
        """
        self.state = msg
        self.attitude = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

    def move_thruster(self):
        """
        Function to move thrusters.
        """
        # self._logger.info('Moving thrusters')
        # self._logger.info(f'Thruster command: {self.cmd_array}')
        # self._logger.info(f'Attitude: {self.thruster_limits}')
        wrench_msg = WrenchStamped()
        if self.number_of_thrusters != len(self.cmd_array):
            self.wrench_pub.publish(wrench_msg)
            return

        for i in range(self.number_of_thrusters):
            if self.cmd_array[i] > self.thruster_limits[i][1]:
                self.cmd_array[i] = self.thruster_limits[i][1]
            elif self.cmd_array[i] < self.thruster_limits[i][0]:
                self.cmd_array[i] = self.thruster_limits[i][0]
            force, moment = self.calculate_wrench(self.thruster_positions, self.thruster_Directions, self.cmd_array, self.cog, self.attitude)

        # print("------------------------------")
        # print("total force", force)
        # print("total moment", moment)
        # print("------------------------------")

        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.wrench.force.x = force[0]
        wrench_msg.wrench.force.y = force[1]
        wrench_msg.wrench.force.z = force[2]
        wrench_msg.wrench.torque.x = moment[0]
        wrench_msg.wrench.torque.y = moment[1]
        wrench_msg.wrench.torque.z = moment[2]

        self.wrench_pub.publish(wrench_msg)

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """
        Function to convert Euler angles to a rotation matrix.

        Parameters: roll (float)
        roll (float): Roll angle in degrees
        pitch (float): pitch angle in degrees
        yaw (float): Yaw angle in degrees

        Returns: rotation_matrix: rotation matrix
        rotation_matrix: rotation matrix (3x3 numpy array).
        """
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(roll_rad), -math.sin(roll_rad)],
            [0, math.sin(roll_rad), math.cos(roll_rad)]
        ])

        Ry = np.array([
            [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
            [0, 1, 0],
            [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
        ])

        Rz = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0],
            [0, 0, 1]
        ])

        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        return rotation_matrix

    def calculate_wrench(self, thruster_positions, thruster_orientations, thruster_forces, center_of_mass, rpy):
        """
        Calculate the resultant wrench (force and moment) in the world frame from multiple thrusters.

        Parameters:
        - thruster_positions: List of thruster positions in the body frame (Nx3 array).
        - thruster_orientations: List of thruster orientations in the body frame (Nx3 array, should be unit vectors).
        - thruster_forces: List of thruster forces (N array).
        - center_of_mass: Position of the center of mass in the body frame (3,)
        - rpy: Roll, pitch, yaw angles representing the current orientation (3,)

        Returns:
        - total_force: Resultant force vector in the world frame (3,)
        - total_moment: Resultant moment vector in the world frame (3,)
        """
        # Calculate the rotation matrix from roll, pitch, yaw
        rotation_matrix = self.euler_to_rotation_matrix(rpy[0], rpy[1], rpy[2])

        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        #self._logger.info(f'thrsuter orientation: {thruster_orientations}')

        for i in range(len(thruster_positions)):
            # Convert thruster direction to the body frame
            direction_body = np.array(thruster_orientations[i])

            # Calculate force in the body frame
            force_body = thruster_forces[i] * direction_body

            # Calculate position relative to the center of mass in the body frame
            position_body = np.array(thruster_positions[i]) - np.array(center_of_mass)

            # Calculate moment in the body frame
            # Clockwise rotation is positive
            moment_pitch = position_body[0] * force_body[2] - position_body[2] * force_body[0]
            moment_yaw = position_body[1] * force_body[0] - position_body[0] * force_body[1]
            moment_roll = position_body[2] * force_body[1] - position_body[1] * force_body[2]

            moment_body = np.array([moment_roll, moment_pitch, moment_yaw])

            #Print debugging information
            # print(f"Thruster {i + 1}:")
            # print(f"  Position (Body): {thruster_positions[i]}")
            # print(f"  Position relative to CoM (Body): {position_body}")
            # print(f"  Orientation (Body): {thruster_orientations[i]}")
            # print(f"  Force (Body): {force_body}")
            # print(f"  Moment (Body): {moment_body}")

            # Convert force and moment to the world frame
            force_world = np.dot(rotation_matrix, force_body)
            moment_world = np.dot(rotation_matrix, moment_body)

            # Sum up the forces and moments
            total_force += force_world
            total_moment += moment_world

            if np.isnan(total_force).any() or np.isnan(total_moment).any():
                self._logger.error('NaN detected in the calculated wrench.')
                return np.zeros(3), np.zeros(3)

        return total_force, total_moment

def main(args=None):
    rclpy.init(args=args)
    move_thruster = MoveThruster()
    rclpy.spin(move_thruster)
    move_thruster.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()