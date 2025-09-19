#!/usr/bin/env python3
"""
Compute thruster forces required to achieve a desired wrench (force + torque).
Author: Adapted to match ThrusterWrenchCalculator style
"""

import numpy as np
import yaml
import os


class WrenchToThrusterAllocator:
    def __init__(self, yaml_path):
        """
        Load thruster configuration from a YAML file.
        :param yaml_path: Path to the same YAML file used in ThrusterWrenchCalculator
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
        self.TAM = self._build_thruster_allocation_matrix()

    def set_center_of_gravity(self, cog):
        """
        Set the center of gravity and rebuild the TAM.
        :param cog: 3-element list or array
        """
        if len(cog) != 3:
            raise ValueError("Center of gravity must be a 3-element vector.")
        self.center_of_gravity = np.array(cog)
        self.TAM = self._build_thruster_allocation_matrix()

    def _build_thruster_allocation_matrix(self):
        """
        Build the 6xN thruster allocation matrix (TAM).
        :return: numpy array of shape (6, N)
        """
        TAM = np.zeros((6, self.number_of_thrusters))
        for i in range(self.number_of_thrusters):
            pos = np.array(self.thruster_positions[i]) - self.center_of_gravity
            dir = np.array(self.thruster_directions[i])
            torque = np.cross(pos, dir)
            TAM[0:3, i] = dir
            TAM[3:6, i] = torque
        return TAM

    def allocate_thrust(self, desired_wrench):
        """
        Compute required thruster forces using pseudoinverse of TAM.
        :param desired_wrench: 6-element list or array [Fx, Fy, Fz, Tx, Ty, Tz]
        :return: numpy array of N thruster forces
        """
        if len(desired_wrench) != 6:
            raise ValueError("Desired wrench must be a 6-element vector.")

        desired_wrench = np.array(desired_wrench)

        # Compute least-squares solution
        pseudo_inv = np.linalg.pinv(self.TAM)
        raw_thrusts = pseudo_inv.dot(desired_wrench)

        # Clamp thrusts within limits
        clamped_thrusts = np.zeros_like(raw_thrusts)
        for i, t in enumerate(raw_thrusts):
            min_t, max_t = self.thrust_limits[i]
            clamped_thrusts[i] = max(min_t, min(max_t, t))

        return clamped_thrusts
