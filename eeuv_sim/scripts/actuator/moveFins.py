#!/usr/bin/python3
"""
Node to control the joints states on Gazebo
@author: Walid Remmas
@contact: walid.remmas@taltech.ee
@date: 08-01-2023
"""

import sys
import rclpy
from rclpy import Future
from rclpy.node import Node

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
import tf_transformations

import numpy as np
from ament_index_python.packages import get_package_share_directory
import os
import yaml

from std_msgs.msg import Bool
from geometry_msgs.msg import WrenchStamped
from finDynamics import FinDynamics
from eeuv_sim_interfaces.msg import Flippers, Flipper, FinForces, FlippersFeedback, FlippersFeedbacks

class MoveFins(Node):
    """Class that allows to move the fins to a desired orientation on Gazebo
    """

    def __init__(self):
        super().__init__('move_fins')

        self.declare_parameter('robot_model')
        self.robot_model = self.get_parameter('robot_model').value

        # Get dynamic parameters from YAML file
        # ---------------------------------------------------
        self.declare_parameter('yaml_dynamics')
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

        # Load parameters from YAML file
        with open(parameters_from_yaml, 'r') as file:
            dynamics_parameters = yaml.load(file, Loader=yaml.FullLoader)
        with open(parameters_rl, 'r') as file:
            rl_parameters = yaml.load(file, Loader=yaml.FullLoader)
        # ---------------------------------------------------
        ### initialize fin dynamics classes ###
        self.finDynamics = FinDynamics(dynamics_parameters["finDynamics"])

        self.fast_forward = rl_parameters["rl"]["fast_forward"]

        self.srv_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        # Wait for the service to be available
        while not self.srv_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.dt = 0.01 # seconds
        if self.robot_model == "UCAT":
            self.fin_positions = {"FR_fin_link": [0.232, -0.096, -0.002],
                                  "BR_fin_link": [-0.226, -0.096, -0.002],
                                  "BL_fin_link": [-0.22, 0.096, -0.002],
                                  "FL_fin_link": [0.226, 0.096, -0.002]}

            self.fin_angles = {"FR_fin_link": [0.0, 3.14159, -2.6179938],
                                  "BR_fin_link": [0.0, 0.0, -3.66519],
                                  "BL_fin_link": [3.14159, 0.0, -2.61799388],
                                  "FL_fin_link": [0.0, 0.0, -0.523599]}
            self.gazeboDirections = [-1, -1, -1, 1]
        else:
            self.fin_positions = {"FL_fin_link": [0.110, 0.065, 0.0],
                                  "BL_fin_link": [-0.110, 0.065, 0.0],
                                  "BR_fin_link": [-0.110, -0.065, 0.0],
                                  "FR_fin_link": [0.110, -0.065, 0.0]}

            self.fin_angles = {"FL_fin_link": [0.0, 0.0, -0.52],
                                  "BL_fin_link": [0.0, 0.0, 0.52],
                                  "BR_fin_link": [3.14159, 0.0, -0.52],
                                  "FR_fin_link": [3.14159, 0.0, 0.52]}
            self.gazeboDirections = [1, -1, -1, 1]


        ### ROS messages ###
        self.flippers_msg = Flippers()
        self.wrench_msg = WrenchStamped()

        self.wrench_debug_msg = FinForces()

        ### init fin kinematics variables ###
        self.amplitude = np.zeros(4)
        self.frequency = np.zeros(4)
        self.zeroDirection = np.zeros(4)
        self.phaseOffset = np.zeros(4)

        self.omega = 0

        self.finMappingDirection = self.finDynamics.finMappingDirection

        self.pos =  np.zeros(4)
        self.initialState = self.finDynamics.initialState
        self.vel = [0.0, 0.0, 0.0, 0.0]

        ### CPG parameters ###
        self.phi = np.zeros(4)      # position of the fin
        self.phi_d = np.zeros(4)    # velocity of the fin
        self.A_d = np.zeros(4)      # derivative of amplitude
        self.A = np.zeros(4)        # amplitude
        self.zd_d = np.zeros(4)     # derivative of zero direction
        self.zd = np.zeros(4) # zero direction
        self.k_amp = 15.0  # rate of convergence for amplitude
        self.k_zd = 7.0  # rate of convergence for zero direction
        
        self.use_cpg = False
        self.max_accereation = 64 # rad/s^2 
        self.max_velocity = np.pi * 16
        self.e_prev = 0

        self.wrench_pub = self.create_publisher(WrenchStamped, '/ucat/force_fins', 10)
        self.wrench_debug_pub = self.create_publisher(FinForces, '/ucat/force_fins_debug', 10)
        self.flipper_feedback_pub = self.create_publisher(FlippersFeedbacks, '/hw/flippers_feedback', 10)

        self.move_fins_timer = self.create_timer(self.dt / self.fast_forward, self.updateJointStates)

        self.sub_flippers = self.create_subscription(Flippers,'/hw/flippers_cmd', self.flippersCallback, 10)
        self.reset_flag_sub = self.create_subscription(Bool,'/ucat/reset', self.reset_joints, 10)

    def normalizeAngle(self, a):
        """
        Function to normalize an angle between -2Pi and 2Pi
        @param: self
        @param: a [radians]- angle in
        @result: returns normalized angle
        """
        while (a > np.pi):
            a -= 2 * np.pi
        while (a < -np.pi):
            a += 2 * np.pi
        return a

    def flippersCallback(self, msg):
        """
        Callback function to get flippers data from the WrenchDriver
        node
        @param: self
        @param: msg [message containing an array of Flipper (1x4)]
        @result: /
        """
        self.flippers_msg = msg
        self.ok = True
        for i in range(0, 4):
            self.amplitude[i] = msg.flippers[i].amplitude 
            self.frequency[i] = msg.flippers[i].frequency
            self.zeroDirection[i] = msg.flippers[i].zero_direction
            self.phaseOffset[i] = msg.flippers[i].phase_offset

    def flipperMotorPDSimulator(self, target, current, dt):
        """
        Function to simulate a PD controller for the flipper motor for target angle
        """
        Kp = 10.0
        Kd = 0.0
        target2 = target + 2 * np.pi
        target3 = target - 2 * np.pi
        _e = min(abs(target - current), abs(target2 - current), abs(target3 - current))
        if abs(target - current) == _e:
            e = target - current
            #print("target - current")
        elif abs(target2 - current) == _e:
            e = target2 - current
            #print("target2 - current")
        else:
            e = target3 - current
            #print("target3 - current")
        #print(target, current)
        #print("e: ", e, "e_prev: ", self.e_prev, "dt: ", dt)
        de = (e - self.e_prev) / dt
        self.e_prev = e
        return Kp * e #+ Kd * de
    

    def getFinParams(self, newAmplitudes, newZeroDirections, frequency, phaseOffset):
        
        dt_cpg = 0.01

        nextPos = np.zeros(4)
        targetPos = np.zeros(4)
        finVelocity = np.zeros(4)
        finAcceleration = np.zeros(4)

        self.omega = 2.0 * np.pi * self.frequency
        phase = self.phaseOffset
        

        for i in range(0,4):
            # using short names for less crowded code
            # nzd: new zero direction
            # na: new amplitude
            p = self.pos[i]
            prev_v = self.vel[i]
            v_lis = []
            a_lis = []
            for j in range(0, int(self.dt / dt_cpg)):
                if self.robot_model == 'UCAT':
                    nzd = self.zeroDirection[i]
                elif self.robot_model == 'microcat':
                    nzd = -self.zeroDirection[i]
                na = self.amplitude[i] / 2.0
                self.phi[i] += dt_cpg * (self.omega[i])
                ### CPG Oscillator ###
                # -------------------------------------------------------------------------------------
                if self.use_cpg:
                    self.A_d[i] += dt_cpg * (self.k_amp * ((self.k_amp/4.0) * (na-self.A[i]) - self.A_d[i]))
                    self.A[i] += dt_cpg * self.A_d[i]
                    self.zd_d[i] += dt_cpg * (self.k_zd * ((self.k_zd/4.0) * (nzd-self.zd[i]) - self.zd_d[i]))
                    self.zd[i] += dt_cpg * self.zd_d[i]
                    p = (self.zd[i] * self.finMappingDirection[i] + self.A[i]*np.cos(self.phi[i]+phase[i]) + self.initialState[i])
                    self.normalizeAngle(p) 
                    # computing fins velocity
                    firstPart = self.A_d[i] * np.cos(self.phi[i]+ phase[i])
                    secondPart = - self.omega[i] * self.A[i] * np.sin(self.phi[i] + phase[i])
                    v_lis.append(self.zd_d[i] * self.finMappingDirection[i] + firstPart + secondPart)

                    # Computing fins acceleration
                    A_dd = self.k_amp * ((self.k_amp/4.0) * (na-self.A[i]) - self.A_d[i])
                    firstPart_d = A_dd *  np.cos(self.phi[i]+phase[i])  - self.omega[i] * self.A_d[i] * np.sin(self.phi[i]+phase[i])
                    secondPart_d = - self.omega[i] * self.A_d[i] * np.sin(self.phi[i]+phase[i]) - self.omega[i]**2 * self.A[i] * np.cos(self.phi[i]+phase[i])
                    zd_dd = self.k_zd * ((self.k_zd/4.0) * (nzd-self.zd[i]) - self.zd_d[i])
                    a_lis.append(zd_dd * self.finMappingDirection[i] + firstPart_d + secondPart_d)
                # -------------------------------------------------------------------------------------
                else:
                    targetPos[i] = (nzd * self.finMappingDirection[i] + na * np.sign(np.sin(self.phi[i] + phase[i])) + self.initialState[i])
                    if abs(targetPos[i] - p) < 0.001:
                        v = 0
                    else:
                        v = self.flipperMotorPDSimulator(targetPos[i], p, dt_cpg)
                    if abs(v) > self.max_velocity:
                        v = self.max_velocity * np.sign(v)
                    p = p + v * dt_cpg
                    p = self.normalizeAngle(p)                   
                    v_lis.append(v)
                    a = (v - prev_v) / dt_cpg
                    if abs(a) > self.max_accereation:
                        a = self.max_accereation * np.sign(a)
                    a_lis.append(a)
                    prev_v = v
            nextPos[i] = p
            finVelocity[i] = np.mean(v_lis)
            finAcceleration[i] = np.mean(a_lis)

            #print("fin n: ", i, "targetPos: ", targetPos[i], "currentPos: ", self.pos[i], "na: ", na, "vel: ", finVelocity[i], "acc: ", finAcceleration[i])


        return nextPos, finVelocity, finAcceleration

    def send_request(self, entity_name, angle):
        # Create the request message
        req = SetEntityState.Request()
        new_orientation_state = EntityState()
        new_orientation_state.name = entity_name
        new_orientation_state.reference_frame = "base_link"

        new_orientation_state.pose.position.x = self.fin_positions[entity_name][0]
        new_orientation_state.pose.position.y = self.fin_positions[entity_name][1]
        new_orientation_state.pose.position.z = self.fin_positions[entity_name][2]

        orientation = tf_transformations.quaternion_from_euler(
                                        self.fin_angles[entity_name][0],
                                        self.fin_angles[entity_name][1] + angle,
                                        self.fin_angles[entity_name][2])

        new_orientation_state.pose.orientation.x = orientation[0]
        new_orientation_state.pose.orientation.y = orientation[1]
        new_orientation_state.pose.orientation.z = orientation[2]
        new_orientation_state.pose.orientation.w = orientation[3]

        req.state = new_orientation_state

        future_set_entity_state = self.srv_client.call_async(req)
        future_set_entity_state.add_done_callback(self._callback_set_entity_state_response)

    def _callback_set_entity_state_response(self, future:Future):
        try:
            response = future.result()
            #self.get_logger().info(f"Gazebo robot returns: {response}.")
        except Exception as e:
            self.get_logger().error(f"Call gazebo_set_entity_state encounter error: {e}")

    def updateJointStates(self):
        dt_cpg = 0.01
        self.flipper_feedbacks = FlippersFeedbacks()
        self.pos, self.vel, acc = self.getFinParams(self.amplitude,
                                        self.zeroDirection,
                                        self.frequency,
                                        self.phaseOffset)
        

        wrench = self.finDynamics.getWrench(self.pos, self.vel, acc)

        self.wrench_msg.header.stamp = self.get_clock().now().to_msg()
        self.wrench_msg.wrench.force.x = wrench[0]
        self.wrench_msg.wrench.force.y = wrench[1]
        self.wrench_msg.wrench.force.z = wrench[2]
        self.wrench_msg.wrench.torque.x = wrench[3]
        self.wrench_msg.wrench.torque.y = wrench[4]
        self.wrench_msg.wrench.torque.z = wrench[5]
        self.wrench_pub.publish(self.wrench_msg)

        self.wrench_debug_msg.header.stamp = self.get_clock().now().to_msg()
        for i in range(4):
            self.wrench_debug_msg.fx[i] = self.finDynamics.FxFins[i]
            self.wrench_debug_msg.fy[i] = self.finDynamics.FyFins[i]
            self.wrench_debug_msg.fz[i] = self.finDynamics.FzFins[i]
            self.wrench_debug_msg.tx[i] = self.finDynamics.TxFins[i]
            self.wrench_debug_msg.ty[i] = self.finDynamics.TyFins[i]
            self.wrench_debug_msg.tz[i] = self.finDynamics.TzFins[i]

        self.flipper_feedbacks.flipper_fr.angular_vel = self.vel[0]
        self.flipper_feedbacks.flipper_fr.angular_pos = self.pos[0]
        self.flipper_feedbacks.flipper_fr.angular_acc = acc[0]
        self.flipper_feedbacks.flipper_br.angular_vel = self.vel[1]
        self.flipper_feedbacks.flipper_br.angular_pos = self.pos[1]
        self.flipper_feedbacks.flipper_br.angular_acc = acc[1]
        self.flipper_feedbacks.flipper_bl.angular_vel = self.vel[2]
        self.flipper_feedbacks.flipper_bl.angular_pos = self.pos[2]
        self.flipper_feedbacks.flipper_bl.angular_acc = acc[2]
        self.flipper_feedbacks.flipper_fl.angular_vel = self.vel[3]
        self.flipper_feedbacks.flipper_fl.angular_pos = self.pos[3]
        self.flipper_feedbacks.flipper_fl.angular_acc = acc[3]
        self.flipper_feedback_pub.publish(self.flipper_feedbacks)

        self.wrench_debug_pub.publish(self.wrench_debug_msg)

        phi = np.zeros(4)
        for i in range(4):
            phi[i] =  self.gazeboDirections[i] * self.pos[i] + self.initialState[i]
        #print(phi)

        self.send_request("FR_fin_link", phi[0])
        self.send_request("BR_fin_link", phi[1])
        self.send_request("BL_fin_link", phi[2])
        self.send_request("FL_fin_link", phi[3])

    def reset_joints(self, msg):
        phi = np.zeros(4)
        for i in range(4):
            phi[i] =  self.initialState[i]

        self.send_request("FR_fin_link", phi[0])
        self.send_request("BR_fin_link", phi[1])
        self.send_request("BL_fin_link", phi[2])
        self.send_request("FL_fin_link", phi[3])

def main(args=None):
    rclpy.init(args=args)
    move_fins = MoveFins()
    rclpy.spin(move_fins)
    move_fins.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
