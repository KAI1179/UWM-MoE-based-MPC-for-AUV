#!/usr/bin/python3

"""
@author: Walid Remmas (Adapted from ROS1 node by Christian Meurer)
@contact: walid.remmas@taltech.ee

"""

import rclpy
from rclpy.node import Node

import numpy as np


class AUVDynamics():
    """
    Class to compute and simulate the dynamics of AUVs
    M - system inertia matrix
    C - Coriolis-centripedal matrix
    D - damping matrix
    G - vector of gravitational/buoyancy forces and moments
    J - Jacobian matrix transforms [velocity, angular velocity] (body -> world).
        Inverse Jacobian implements (world -> body).
    """

    def __init__(self, dynamics_parameters):
        """
            self:
            dynamic_parameters: dictionary variable containg dynamic params
        """
        # Read AUV dynamics parameters from dictionary
        # ---------------------------------------------------
        self.params = dynamics_parameters
        # ---------------------------------------------------

        self.dt = 1

        ### init matrices ###
        self.M = np.zeros((6, 6))
        self.M_inv = np.zeros((6, 6))
        self.C = np.zeros((6, 6))
        self.CRB = np.zeros((6, 6))
        self.CAdded = np.zeros((6, 6))
        self.D = np.zeros((6, 6))
        self.G = np.zeros(6)
        self.L = np.zeros((6, 6))
        self.J = np.zeros((6, 6))
        self.J_inv = np.zeros((6, 6))

        self.Jd = np.zeros((6, 6))
        self.Jd_inv = np.zeros((6, 6))


        ### inertial parameters ###
        self.mass = self.params["inertial"]["mass"]  # mass in kg
        self.Ib = self._get_inertia_matrix()

        ### buoyancy related ###
        self.RG = np.array(self.params["geometry"]["COG"]) # center of gravity (body frame)
        self.RB = np.array(self.params["geometry"]["COB"]) # center of buoyancy (body frame)
        self.We = self.mass * 9.81  # gravity
        self.Bo = self.We * 1.0  # buoyancy

        self._init_mass_matrix()

        ### drag coefficients ###
        self.Xu = self.params["drag"]["Xu"]
        self.Xuu = self.params["drag"]["Xuu"]
        self.Yv = self.params["drag"]["Yv"]
        self.Yvv = self.params["drag"]["Yvv"]
        self.Zw = self.params["drag"]["Zw"]
        self.Zww = self.params["drag"]["Zww"]
        self.Kp = self.params["drag"]["Kp"]
        self.Kpp = self.params["drag"]["Kpp"]
        self.Mq = self.params["drag"]["Mq"]
        self.Mqq = self.params["drag"]["Mqq"]
        self.Nr = self.params["drag"]["Nr"]
        self.Nrr = self.params["drag"]["Nrr"]

        print("Dynamics parameters loaded")


    def _get_inertia_matrix(self):
        """
        Moment of inertia U-CAT
        """
        Ixx = self.params["inertial"]["Ixx"]
        Ixy = self.params["inertial"]["Ixy"]
        Ixz = self.params["inertial"]["Ixz"]
        Iyx = self.params["inertial"]["Iyx"]
        Iyy = self.params["inertial"]["Iyy"]
        Iyz = self.params["inertial"]["Iyz"]

        Izx = self.params["inertial"]["Izx"]
        Izy = self.params["inertial"]["Izy"]
        Izz = self.params["inertial"]["Izz"]

        return np.array([
            [Ixx, -Ixy, -Ixz],
            [-Iyx, Iyy, -Iyz],
            [-Izx, -Izy, Izz]])

    def _S(self, k):
        """
        Cross product matrix based on vector v (skew symmetric matrix).
        v * _S(k) is equivalent to the cross product of vectors v and k.
        """
        return np.array([
                        [0, -k[2], k[1]],
                        [k[2], 0, -k[0]],
                        [-k[1], k[0], 0]])

    def _init_mass_matrix(self):
        self.M11 = self.mass * np.eye(3)
        self.M12 = -self.mass * self._S(self.RG)
        self.M21 = -self.M12

        # rigid body inertia matrix
        self.MRB = np.zeros((6, 6))
        self.MRB[0:3, 0:3] = self.M11
        self.MRB[0:3, 3:6] = self.M12
        self.MRB[3:6, 0:3] = self.M21
        self.MRB[3:6, 3:6] = self.Ib

        # added mass - strip theory
        Xud = self.params["inertial"]["added_mass"]["Xud"]
        Yvd = self.params["inertial"]["added_mass"]["Yvd"]
        Zwd = self.params["inertial"]["added_mass"]["Zwd"]
        Kpd = self.params["inertial"]["added_mass"]["Kpd"]
        Mqd = self.params["inertial"]["added_mass"]["Mqd"]
        Nrd = self.params["inertial"]["added_mass"]["Nrd"]
        Mwd = 0.0  # -1.0154
        self.MAdded = -np.diag([Xud, Yvd, Zwd, Kpd, Mqd, Nrd])
        self.MAdded[2, 4] = Mwd
        self.MAdded[4, 2] = Mwd

        self.M = self.MAdded + self.MRB
        self.M_inv = np.linalg.inv(self.M)

    def _update_centripetal_matrix(self, velocity):
        v1 = np.transpose(velocity[0:3])  # linear velocity
        v2 = np.transpose(velocity[3:6])  # angular velocity

        C12 = -self.mass * self._S(v1) - self.mass * \
            np.dot(self._S(v2), self._S(self.RG))
        C21 = -self.mass * self._S(v1) + self.mass * \
            np.dot(self._S(self.RG), self._S(v2))
        C22 = -self._S(np.dot(self.Ib, v2))

        self.CRB[0:3, 3:6] = C12
        self.CRB[3:6, 0:3] = C21
        self.CRB[3:6, 3:6] = C22

        A11 = self.MAdded[0:3, 0:3]
        A12 = self.MAdded[0:3, 3:6]
        A21 = self.MAdded[3:6, 0:3]
        A22 = self.MAdded[3:6, 3:6]

        CA12 = -self._S(np.dot(A11, v1) + np.dot(A12, v2))
        CA22 = -self._S(np.dot(A21, v1) + np.dot(A22, v2))

        self.CAdded[0:3, 3:6] = CA12
        self.CAdded[3:6, 0:3] = CA12
        self.CAdded[3:6, 3:6] = CA22

        self.C = self.CRB + self.CAdded

    def _update_damping_matrix(self, velocity):

        self.D[0, 0] = -self.Xuu * abs(velocity[0]) - self.Xu
        self.D[1, 1] = -self.Yvv * abs(velocity[1]) - self.Yv
        self.D[2, 2] = -self.Zww * abs(velocity[2]) - self.Zw
        self.D[3, 3] = -self.Kpp * abs(velocity[3]) - self.Kp
        self.D[4, 4] = -self.Mqq * abs(velocity[4]) - self.Mq
        self.D[5, 5] = -self.Nrr * abs(velocity[5]) - self.Nr

    def _update_gravity_vector(self, position):
        phi = position[3]  # roll
        theta = position[4]  # pitch

        self.G[0] = (self.We - self.Bo) * np.sin(theta)
        self.G[1] = -(self.We - self.Bo) * np.cos(theta) * np.sin(phi)
        self.G[2] = -(self.We - self.Bo) * np.cos(theta) * np.cos(phi)

        self.G[3] = ((self.RG[1] * self.We - self.RB[1] * self.Bo) *
                     np.cos(theta) * np.cos(phi) -
                     (self.RG[2] * self.We - self.RB[2] * self.Bo) * np.cos(theta) *
                     np.sin(phi))

        self.G[4] = (-(self.RG[2] * self.We - self.RB[2] * self.Bo) *
                     np.sin(theta) -
                     (self.RG[0] * self.We - self.RB[0] * self.Bo) * np.cos(theta) *
                     np.cos(phi))

        self.G[5] = ((self.RG[0] * self.We - self.RB[0] * self.Bo) *
                     np.cos(theta) * np.sin(phi) +
                     (self.RG[1] * self.We - self.RB[1] * self.Bo) * np.sin(phi))


    def _update_jacobian_matrix(self, position):
        phi = position[3]  # roll
        theta = position[4]  # pitch
        psi = position[5]  # yaw

        xrot = np.array([[1.0, 0.0, 0.0],
                         [0.0, np.cos(phi), -np.sin(phi)],
                         [0.0, np.sin(phi), np.cos(phi)]])

        yrot = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(theta), 0.0, np.cos(theta)]])

        zrot = np.array([[np.cos(psi), -np.sin(psi), 0.0],
                         [np.sin(psi), np.cos(psi), 0.0],
                         [0.0, 0.0, 1.0]])

        ROT = np.dot(np.dot(zrot, yrot), xrot)
        ROT_inv = np.linalg.inv(ROT)

        T = np.array([
            [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
            [0.0, np.cos(phi), -np.sin(phi)],
            [0.0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])
        T_inv = np.linalg.inv(T)

        self.J[0:3, 0:3] = ROT
        self.J[3:6, 3:6] = T

        self.J_inv[0:3, 0:3] = ROT_inv
        self.J_inv[3:6, 3:6] = T_inv

    def update(self, position, velocity):
        """
        Update M, C, D, G, J matrices based on current state.

        position = [x, y, z, roll, pitch, yaw] in world frame
        velocity = [vx, vy, vz, v_roll, v_pitch, v_yaw] in body fixed frame
        """
        self._update_centripetal_matrix(velocity)
        self._update_damping_matrix(velocity)
        self._update_gravity_vector(position)
        self._update_jacobian_matrix(position)

    def get_vel(self, vel):
        """ Given the current velocity and the previous position computes the p_dot """
        p_dot = np.zeros(6)
        p_dot[0:3] = np.dot(self.J[0:3, 0:3], vel[0:3])
        p_dot[3:6] = np.dot(self.J[3:6, 3:6], vel[3:6])
        return p_dot

    def get_acc(self, vel, U):
        M_inv = self.M_inv
        C = self.C
        D = self.D
        G = self.G
        return np.dot(M_inv, -np.dot(C, vel) - np.dot(D, vel) - G + U)
