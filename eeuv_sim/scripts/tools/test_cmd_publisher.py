#!/usr/bin/python3
"""
Script to publish MultiArray message to control thrusters and Wing in Gazebo
@author: Yuya Hamamatsu 
@contact: yuya.hamamatsu@taltech.ee

"""

import numpy as np

import argparse
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

class TestCmdPublisher(Node):
    def __init__(self):
        super().__init__('test_cmd_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, '/ucat/thruster_cmd', 10)
        self.publisher_wing = self.create_publisher(Float32MultiArray, '/ucat/wing_angle_cmd', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        #list = [3.0, -3.0, -3.0, 3.0, 0.0, -3.0, 0, 0.0]
        list = [5.0]
        list_w = [-40, 0, 40, 0] 

        self.cmd = list
        self.cmd_wing = list_w

    def timer_callback(self):
        msg = Float32MultiArray()
        msg_w = Float32MultiArray()
        for i in range(len(self.cmd)):
            msg.data.append(self.cmd[i])
        for i in range(len(self.cmd_wing)):
            msg_w.data.append(self.cmd_wing[i])

        self.publisher_.publish(msg)
        self.publisher_wing.publish(msg_w)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.get_logger().info('Publishing Wing: "%s"' % msg_w.data)
        self.i += 0.1

def main(args=None):
    rclpy.init(args=args)
    node = TestCmdPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    