#!/usr/bin/python3
"""
Script to calculate wing movement for glider AUV in Gazebo
@author: Yuya Hamamatsu 
@contact: yuya.hamamatsu@taltech.ee
"""
import os
import yaml
import math
import numpy as np

import rclpy
from rclpy import Future
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import WrenchStamped, Twist
from std_msgs.msg import Bool, Float32MultiArray
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState

from tf_transformations import euler_from_quaternion, quaternion_from_euler

from wingDynamics import WingDynamics

class MoveWing(Node):
    def __init__(self):
        super().__init__('moveWing')
        self.declare_parameter('robot_model', 'UCAT')
        self.robot_model = self.get_parameter('robot_model').value
        
        if self.robot_model == 'UCAT':
            self.declare_parameter('yaml_dynamics', 'UCATDynamics.yaml')
        elif self.robot_model == 'LAUV':
            self.declare_parameter('yaml_dynamics', 'LAUVDynamics.yaml')

        yaml_dynamics = self.get_parameter('yaml_dynamics').value
        parameters_from_yaml = os.path.join(
                get_package_share_directory('uw_gazebo'),
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
            self.wing_dynamics_parameters = yaml.load(file, Loader=yaml.FullLoader)
        with open(parameters_rl, 'r') as file:
            rl_parameters = yaml.load(file, Loader=yaml.FullLoader)
        
        self.wing_dynamics = WingDynamics(self.wing_dynamics_parameters["wingDynamics"])
        self.fast_forward = self.fast_forward = rl_parameters["rl"]["fast_forward"]

        self.get_logger().info('Wing movement node has been initialized')

        self.dt = 0.1

        self.wing_angle_state_pub = self.create_publisher(Float32MultiArray, '/ucat/wing_angle_state', 10)
        self.wrench_pub = self.create_publisher(WrenchStamped, '/ucat/force_wings', 10)

        self.create_timer(self.dt / self.fast_forward, self.move_wing)
        self.reset_flag_sub = self.create_subscription(Bool,'/ucat/reset', self.reset_flag_callback, 10)
        self.wing_angle_sub = self.create_subscription(Float32MultiArray, '/ucat/wing_angle_cmd', self.wing_angle_callback, 10)
        self.state_sub = self.create_subscription(EntityState, "/ucat/state", self.state_callback, 10)

        self.state = EntityState()
        self.twist = Twist()
        self.wing_angle = Float32MultiArray()

        self.vel_world = np.zeros(3)
        self.attitude = np.zeros(3)
        
        self.number_of_wings = self.wing_dynamics_parameters["wingDynamics"]["NumberOfWings"]
        self.wing_area = self.wing_dynamics_parameters["wingDynamics"]["WingArea"]
        self.wing_positions = self.wing_dynamics_parameters["wingDynamics"]["WingPositions"]
        self.wing_direction = self.wing_dynamics_parameters["wingDynamics"]["WingMountingAngle"]
        self.wing_movement_direction = self.wing_dynamics_parameters["wingDynamics"]["WingMovementDirection"]
        self.wing_limit = self.wing_dynamics_parameters["wingDynamics"]["WingMovementLimits"]

        self.wing_changing_rate = self.wing_dynamics_parameters["wingDynamics"]["WingChangingRate"]

        self.wing_angle_list = [0.0] * self.number_of_wings
        self.wing_angle_state = Float32MultiArray()
        self.wing_angle_state_list = [0.0] * self.number_of_wings

    def reset_flag_callback(self, msg):
        """
        Callback function for reset flag.
        """
        self.reset_flag = msg.data
        if self.reset_flag:
            self.wing_angle_state_list = [0.0] * self.number_of_wings
            self.vel_world = np.zeros(3)
            self.attitude = np.zeros(3)

        

    def state_callback(self, msg):
        if not math.isnan(msg.pose.orientation.w):
            self.state = msg
            self.twist = msg.twist
            self.attitude = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            self.vel_world = np.array([self.twist.linear.x, self.twist.linear.y, self.twist.linear.z])
    
    def wing_angle_callback(self, msg):
        self.wing_angle = msg
        self.wing_angle_list = self.wing_angle.data

    # def reset_joints(self, msg):
    #     self.wing_dynamics.reset()
    #     self.get_logger().info('Resetting wing angles')

    def update_wing_angle(self):
        array = []
        if len(self.wing_angle_list) != self.number_of_wings:
            return
        for i in range(self.number_of_wings):
            e = self.wing_angle_list[i] - self.wing_angle_state_list[i]
            sign = 1 if e > 0 else -1
            omega = self.wing_changing_rate[i]
            thera = self.wing_angle_state_list[i] + sign * omega * self.dt
            if thera > self.wing_limit[i][1]:
                thera = self.wing_limit[i][1]
            elif thera < self.wing_limit[i][0]:
                thera = self.wing_limit[i][0]
            array.append(thera)
        self.wing_angle_state.data = array
        self.wing_angle_state_pub.publish(self.wing_angle_state)
        self.wing_angle_state_list = array

    def move_wing(self):
        self.update_wing_angle()
        wrenchs = WrenchStamped()
        for i in range(self.number_of_wings):
            # self._logger.info(f'Wing {i} angle: {self.wing_angle_state_list[i]}')
            # self._logger.info(f'Wing {i} position {self.wing_positions[i]}')
            # self._logger.info(f'Wing {i} direction {self.wing_direction[i]}')
            # self._logger.info(f'Wing {i} movement direction {self.wing_movement_direction[i]}')
            # self._logger.info(f'Wing {i} area {self.wing_area[i]}')
            # self._logger.info(f'Wing {i} attitude {self.attitude}')
            wrench = self.wing_dynamics.calculate_wrench(self.vel_world, self.wing_area[i], self.wing_positions[i], self.attitude, self.wing_direction[i], self.wing_angle_state_list[i], self.wing_movement_direction[i])
            # check nan
            # self._logger.info(f'Wing {i} wrench: {wrench}')
            wrenchs.wrench.force.x += wrench[0]
            wrenchs.wrench.force.y += wrench[1]
            wrenchs.wrench.force.z += wrench[2]
            wrenchs.wrench.torque.x += wrench[3]
            wrenchs.wrench.torque.y += wrench[4]
            wrenchs.wrench.torque.z += wrench[5]

        self.wrench_pub.publish(wrenchs)


def main(args=None):
    rclpy.init(args=args)
    move_wings = MoveWing()
    rclpy.spin(move_wings)
    move_wings.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()