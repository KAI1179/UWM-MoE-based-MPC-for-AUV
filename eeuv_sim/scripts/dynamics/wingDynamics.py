#!/usr/bin/python3
"""
Script to calculate wing dynamics for glider AUV in Gazebo
@author: Yuya Hamamatsu 
@contact: yuya.hamamatsu@taltech.ee
"""

import math
import numpy as np

class WingDynamics():
    def __init__(self, dynamics_parameters):
        """
            self:
            dynamics_parameters: dictionary variable containg fin dynamics params
        """
        self.c_d = dynamics_parameters["C_D"]
        self.c_l = dynamics_parameters["C_L"]
        self.wing_area = dynamics_parameters["WingArea"]


        alpha_mount = dynamics_parameters["WingMountingAngle"]
        self.alpha_mount_cos = np.cos(np.deg2rad(alpha_mount))
        self.alpha_mount_sin = np.sin(np.deg2rad(alpha_mount))


        self.FxFins = np.zeros(4)
        self.FyFins = np.zeros(4)
        self.FzFins = np.zeros(4)

        self.TxFins = np.zeros(4)
        self.TyFins = np.zeros(4)
        self.TzFins = np.zeros(4)

        
    def _calculate_forces(self, velocity_body, wing_area, angle_of_attack, air_density=998):
        """
        Functions to calculate lift and drag.
        
        Parameters: velocity_body (np.array)
        velocity_body (np.array): Velocity vector in body coordinates (m/s).
        wing_area (float): Wing area in square meters
        angle_of_attack (float): angle of attack in radians
        air_density (float): Air density in kg/m^3. Default is 1.225 kg/m^3 at standard sea level.
        
        Returns: air_density (float): air density in kg/m^3.
        (lift, drag): Tuple of lift and drag in newtons.
        """
        speed = np.linalg.norm(velocity_body)
        
        cl_0 = 0.0  # zero lift angle of attack
        cd_0 = self.c_d[1]  # zero lift drag coefficient
        
        # https://www.sciencedirect.com/science/article/pii/S0029801820305758
        cl = self.c_l[0] * angle_of_attack + self.c_l[0] * angle_of_attack * abs(angle_of_attack) + cl_0
        cd = self.c_d[0] * angle_of_attack * angle_of_attack + cd_0
        
        q = 0.5 * air_density * (speed ** 2)
        
        lift = cl * q * wing_area
        drag = cd * q * wing_area
        
        return lift, drag

    def calculate_wrench(self, velocity_world, wing_area, distance_to_cg, body_orientation, default_wing_angle, control_angle, axis='pitch', air_density=998):
        """
        Function to calculate forces and moments (wrenches) applied to a fuselage based on wing angle of attack, velocity, and wing area.
        
        Parameters: velocity_world (np.array): velocity in world coordinates (m/s).
        velocity_world (np.array): velocity in world coordinates (m/s).
        wing_area (float): wing area in square meters
        distance_to_cg (np.array): distance (x, y, z) in meters from the wing's center of lift to the plane's center of gravity.
        body_orientation (tuple): body orientation (roll, pitch, yaw) in degrees relative to the world coordinate system.
        default_wing_angle (tuple): Default wing angle (roll, pitch, yaw) in degrees.
        control_angle (float): operating angle in degrees.
        axis (str): 'pitch' or 'yaw'. Specifies the axis of operation.
        air_density (float): Air density in kg/m^3. Default is 1.225 kg/m^3 at standard sea level.
        
        Returns: The air density of the airframe.
        wrench: Tuple of forces and moments applied to the airframe in Newton Newton meters.
        """
        
        # Transform velocity to body coordinates
        rotation_matrix_world_to_body = self._euler_to_rotation_matrix(*body_orientation).T
        velocity_body = np.dot(rotation_matrix_world_to_body, velocity_world)
        control_angle_rad = math.radians(control_angle)
        default_wing_angle_rad = np.radians(default_wing_angle)
        
        if np.linalg.norm(velocity_body) > 0:
            if axis == 'pitch':
                angle_of_attack = math.atan2(velocity_body[2], velocity_body[0]) + default_wing_angle_rad + control_angle_rad
            elif axis == 'roll':
                angle_of_attack = math.atan2(velocity_body[2], velocity_body[1]) + default_wing_angle_rad + control_angle_rad
            elif axis == 'yaw':
                angle_of_attack = math.atan2(velocity_body[1], velocity_body[0]) + default_wing_angle_rad + control_angle_rad
            else:
                raise ValueError("Axis must be 'pitch' or 'yaw'")
        else:
            angle_of_attack = default_wing_angle_rad + control_angle_rad
        
        lift, drag = self._calculate_forces(velocity_body, wing_area, angle_of_attack, air_density)
        
        # normalize velocity
        if np.linalg.norm(velocity_body) > 0:
            velocity_dir_body = velocity_body / np.linalg.norm(velocity_body)
        else:
            velocity_dir_body = np.array([0, 0, 0])
        
        # calculate lift direction
        if np.linalg.norm(velocity_body) > 0:
            if axis == 'pitch':
                lift_direction_body = np.array([-velocity_dir_body[2], 0, velocity_dir_body[0]])
            elif axis == 'roll':
                lift_direction_body = np.array([0, -velocity_dir_body[2], velocity_dir_body[1]])
            elif axis == 'yaw':
                lift_direction_body = np.array([-velocity_dir_body[1], velocity_dir_body[0], 0])
            lift_direction_body /= np.linalg.norm(lift_direction_body)
        else:
            lift_direction_body = np.array([0, 0, 0])

        # print("------------------------------")
        # print("DYANAMICS")
        # print("lift_direction_body", lift_direction_body)
        # print("velocity_dir_body", velocity_dir_body)
        # print("distance_to_cg", distance_to_cg)
        # print("lift", lift)
        # print("drag", drag)
        # print("body_orientation", body_orientation)
        # print("rotation_matrix_world_to_body", rotation_matrix_world_to_body)
        # print("velocity_world", velocity_world)
        # print("velocity_body", velocity_body)
        # print("--------------------------------")

        
        lift_vector = lift * lift_direction_body
        drag_vector = -drag * velocity_dir_body
        force_body = lift_vector + drag_vector
        
        moment_body = np.cross(distance_to_cg, force_body)
        
        rotation_matrix_body_to_world = self._euler_to_rotation_matrix(*body_orientation)
        force_world = np.dot(rotation_matrix_body_to_world, force_body)
        moment_world = np.dot(rotation_matrix_body_to_world, moment_body)
        
        # return 6-DOF wrench list
        wrench = np.concatenate((force_world, moment_world))

        return wrench

    def _euler_to_rotation_matrix(self, roll, pitch, yaw):
        """
        Function to convert Euler angles to a rotation matrix.
        
        Parameters: roll (float): roll angle in degrees.
        roll (float): Roll angle in degrees
        pitch (float): pitch angle in degrees
        yaw (float): yaw angle in degrees
        
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

