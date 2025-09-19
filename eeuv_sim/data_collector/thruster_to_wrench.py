#!/usr/bin/env python3
"""
Compute wrench from thruster configuration loaded via YAML.
Author: Adapted from moveThruster.py
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
            t = thrusts[i]

            # Optional: Clamp thrust to limits
            t = max(self.thrust_limits[i][0], min(t, self.thrust_limits[i][1]))

            f = d * t
            tau = np.cross(p, f)

            force_total += f
            torque_total += tau

        return np.concatenate([force_total, torque_total])
