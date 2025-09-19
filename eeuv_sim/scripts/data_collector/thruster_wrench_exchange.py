#!/usr/bin/env python3
"""
Compute wrench from thruster configuration loaded via YAML.
Author: Adapted from moveThruster.py
Enhanced: Supports forward and inverse mapping between thrusters and wrench (body/world)

compute_wrench(thrusts)
    从 thruster forces 计算 wrench（body frame）
compute_wrench_world(thrusts, rpy)
    从 thruster forces 计算 wrench（world frame）
compute_thruster_forces_from_wrench(wrench_body)
    从 wrench（body）反求 thruster forces
compute_thruster_forces_from_wrench_world(wrench_world, rpy)	从
    wrench（world）反求 thruster forces
set_center_of_gravity(cog)
    设置重心偏移
euler_to_rotation_matrix(roll, pitch, yaw)
    姿态转旋转矩阵
"""

import numpy as np
import yaml
import os


class ThrusterWrenchCalculator:
    def __init__(self, yaml_path):
        """
        Initialize the calculator by loading thruster configuration from YAML.
        :param yaml_path: Full path to YAML file (e.g., UCATDynamics.yaml)
        """
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        try:
            self.thruster_params = data['thrusterDynamics']
            self.number_of_thrusters = self.thruster_params['NumberOfThrusters']
            self.thruster_positions = self.thruster_params['thrusterPositions']
            self.thruster_directions = self.thruster_params['thrusterDirections']
            self.thrust_limits = self.thruster_params['thrustlimits']
        except KeyError as e:
            raise KeyError(f"Missing required thrusterDynamics field in YAML: {e}")

        self.center_of_gravity = np.zeros(3)

    def set_center_of_gravity(self, cog):
        """
        Optionally set center of gravity offset.
        :param cog: list or np.array of shape (3,)
        """
        if len(cog) != 3:
            raise ValueError("CoG must be a 3-element vector.")
        self.center_of_gravity = np.array(cog)

    def compute_wrench(self, thrusts):
        """
        Compute wrench vector (force and moment) in body frame.
        :param thrusts: list or np.array of length N (thruster forces)
        :return: np.array of shape (6,) [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        if len(thrusts) != self.number_of_thrusters:
            raise ValueError(f"Expected {self.number_of_thrusters} thruster inputs, got {len(thrusts)}")

        force_total = np.zeros(3)
        torque_total = np.zeros(3)

        for i in range(self.number_of_thrusters):
            p = np.array(self.thruster_positions[i]) - self.center_of_gravity
            d = np.array(self.thruster_directions[i])
            t = max(self.thrust_limits[i][0], min(thrusts[i], self.thrust_limits[i][1]))

            f = d * t
            tau = np.cross(p, f)

            force_total += f
            torque_total += tau

        return np.concatenate([force_total, torque_total])

    # def compute_wrench_world(self, thrusts, rpy):
    #     """
    #     Compute wrench vector (force and moment) in world frame.
    #     :param thrusts: list or np.array of length N
    #     :param rpy: list or np.array of shape (3,) => [roll, pitch, yaw]
    #     :return: (3,) force and (3,) moment in world frame
    #     """
    #     if len(thrusts) != self.number_of_thrusters:
    #         raise ValueError(f"Expected {self.number_of_thrusters} thruster inputs, got {len(thrusts)}")
    #
    #     rotation_matrix = self.euler_to_rotation_matrix(*rpy)
    #
    #     total_force_world = np.zeros(3)
    #     total_moment_world = np.zeros(3)
    #
    #     for i in range(self.number_of_thrusters):
    #         position_body = np.array(self.thruster_positions[i]) - self.center_of_gravity
    #         direction_body = np.array(self.thruster_directions[i])
    #         t = max(self.thrust_limits[i][0], min(thrusts[i], self.thrust_limits[i][1]))
    #
    #         force_body = t * direction_body
    #
    #         # Moment in body
    #         moment_pitch = position_body[0] * force_body[2] - position_body[2] * force_body[0]
    #         moment_yaw   = position_body[1] * force_body[0] - position_body[0] * force_body[1]
    #         moment_roll  = position_body[2] * force_body[1] - position_body[1] * force_body[2]
    #         moment_body = np.array([moment_roll, moment_pitch, moment_yaw])
    #
    #         # Rotate to world
    #         force_world = rotation_matrix @ force_body
    #         moment_world = rotation_matrix @ moment_body
    #
    #         total_force_world += force_world
    #         total_moment_world += moment_world
    #
    #     return total_force_world, total_moment_world

    def compute_wrench_world(self, thrusts, rpy):
        """
        Compute wrench vector (force and moment) in world frame.
        :param thrusts: list or np.array of length N
        :param rpy: list or np.array of shape (3,) => [roll, pitch, yaw] (in radians)
        :return: (3,) force and (3,) moment in world frame
        """
        if len(thrusts) != self.number_of_thrusters:
            raise ValueError(f"Expected {self.number_of_thrusters} thruster inputs, got {len(thrusts)}")
        if len(rpy) != 3:
            raise ValueError("RPY must be a 3-element vector.")

        # Body-to-world rotation
        rotation_matrix = self.euler_to_rotation_matrix(*rpy)

        total_force_world = np.zeros(3)
        total_moment_world = np.zeros(3)

        for i in range(self.number_of_thrusters):
            # Position relative to CoG in body frame
            position_body = np.array(self.thruster_positions[i]) - self.center_of_gravity
            # Direction in body frame
            direction_body = np.array(self.thruster_directions[i])
            # Clamp thrust to limits
            t = np.clip(thrusts[i], self.thrust_limits[i][0], self.thrust_limits[i][1])

            # Force in body frame
            force_body = t * direction_body
            # Moment in body frame using right-hand rule
            moment_body = np.cross(position_body, force_body)

            # Transform to world frame
            force_world = rotation_matrix @ force_body
            moment_world = rotation_matrix @ moment_body

            total_force_world += force_world
            total_moment_world += moment_world

        return total_force_world, total_moment_world

    def compute_thruster_forces_from_wrench(self, wrench_body):
        """
        Compute thruster forces that best approximate the given wrench (in body frame).
        Uses least-squares solution to handle redundancy or underactuation.
        :param wrench_body: np.array of shape (6,) = [Fx, Fy, Fz, Tx, Ty, Tz]
        :return: np.array of shape (N,) thruster forces
        """
        if len(wrench_body) != 6:
            raise ValueError("Wrench must be a 6-element vector.")

        T = np.zeros((6, self.number_of_thrusters))

        for i in range(self.number_of_thrusters):
            p = np.array(self.thruster_positions[i]) - self.center_of_gravity
            d = np.array(self.thruster_directions[i])

            T[0:3, i] = d
            T[3:6, i] = np.cross(p, d)

        thruster_forces, _, _, _ = np.linalg.lstsq(T, wrench_body, rcond=None)

        for i in range(self.number_of_thrusters):
            thruster_forces[i] = np.clip(thruster_forces[i], self.thrust_limits[i][0], self.thrust_limits[i][1])

        return thruster_forces

    def compute_thruster_forces_from_wrench_world(self, wrench_world, rpy):
        """
        Given a desired wrench in world frame, compute the corresponding thruster forces.
        Internally converts wrench to body frame, then solves using least squares.

        :param wrench_world: (6,) numpy array [Fx, Fy, Fz, Tx, Ty, Tz] in world frame
        :param rpy: (3,) roll, pitch, yaw
        :return: (N,) numpy array of thruster forces
        """
        if len(wrench_world) != 6:
            raise ValueError("Wrench must be a 6-element vector.")
        if len(rpy) != 3:
            raise ValueError("RPY must be a 3-element vector.")

        R = self.euler_to_rotation_matrix(*rpy)
        R_inv = R.T

        force_body = R_inv @ wrench_world[:3]
        torque_body = R_inv @ wrench_world[3:]
        wrench_body = np.concatenate([force_body, torque_body])

        return self.compute_thruster_forces_from_wrench(wrench_body)

    @staticmethod
    def euler_to_rotation_matrix(roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) to rotation matrix.
        :param roll: rotation about x-axis
        :param pitch: rotation about y-axis
        :param yaw: rotation about z-axis
        :return: 3x3 rotation matrix
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ])
