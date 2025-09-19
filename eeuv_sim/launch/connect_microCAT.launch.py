#!/usr/bin/python3
"""
Launching hardware and software interfaces for μ-CAT simulation

@author: Roza Gkliva
@contact: roza.gkliva@taltech.ee
"""

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ucat_hardware_interface = Node(
        package='eeuv_sim',  
        executable='microcatHardwareInterface.py',
        name='ros_arduino_interface',
        output='screen',
        parameters=[
            {'serial_port': '/dev/ttyACM0'}
        ]
    )


    return LaunchDescription([
        ucat_hardware_interface
    ])