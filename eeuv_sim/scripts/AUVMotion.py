#!/usr/bin/python3

"""
Node to update the robot's pose in Gazebo based on AUV dyanmics
@author: Yuya Hamamatsu, Walid Remmas
@contact: yuya.hamamatsu@taltech.ee
"""

import rclpy
from rclpy.node import Node
from rclpy import Future
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Bool, Float32
from gazebo_msgs.srv import SetEntityState, SpawnEntity, DeleteEntity
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import WrenchStamped, Pose, Point, Quaternion, Twist, Vector3

from tf_transformations import quaternion_from_euler
import numpy as np
import yaml
import os

from eeuv_sim.srv import ResetToPose
import random

from AUVDynamics import AUVDynamics


class AUVMotion(Node):
    """
    Class to update the AUV dynamics based on actuators forces
    and updates the robot's pose in Gazebo using the ApplyLinkWrench service
    """
    def __init__(self, obscales=False):
        super().__init__('auv_motion')
        self.with_obstacles = obscales

        # Get dynamic parameters from YAML file
        # ---------------------------------------------------

        self.declare_parameter('robot_model', 'UCAT')
        self.robot_model = self.get_parameter('robot_model').value
        
        if self.robot_model == 'UCAT':
            self.declare_parameter('yaml_dynamics', 'UCATDynamics.yaml')
            self._logger.info("UCAT model is selected")
        elif self.robot_model == 'LAUV':
            self.declare_parameter('yaml_dynamics', 'LAUVDynamics.yaml')
            self._logger.info("LAUV model is selected")
        elif self.robot_model == 'Blue':
            self.declare_parameter('yaml_dynamics', 'BlueDynamics.yaml')
            self._logger.info("Blue model is selected")

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
            self.dynamics_parameters = yaml.load(file, Loader=yaml.FullLoader)
        with open(parameters_rl, 'r') as file:
            rl_parameters = yaml.load(file, Loader=yaml.FullLoader)
        # ---------------------------------------------------
        self.dt = 0.01
        self.fast_forward = rl_parameters["rl"]["fast_forward"]

        
        self.srv_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        # Wait for the service to be available
        while not self.srv_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        
        self.default_dynamics_parameters = self.dynamics_parameters.copy()
        self.dynamics = AUVDynamics(self.dynamics_parameters["dynamics"])
        self.pos = np.zeros(6)
        self.pos[2] = 5.0
        self.vel = np.zeros(6)
        self.acc = np.zeros(6)
        self.U = np.zeros(6)
        self.depth_vel_hist = [0] * 10

        self.reset_flag_sub = self.create_subscription(Bool,'/ucat/reset', self.reset_callback, 10)
        self.auv_dynamics_timer = self.create_timer(self.dt / self.fast_forward, self.updateDynamics)

        if self.robot_model == 'LAUV':
            self.wrench_from_wings_sub = self.create_subscription(WrenchStamped,'/ucat/force_wings', self.WrenchfromWings, 10)
            self.wrench_from_wings = WrenchStamped()
            self.wrench_from_thrusters_sub = self.create_subscription(WrenchStamped,'/ucat/force_thrust', self.WrenchfromThrusters, 10)
            self.wrench_from_thrusters = WrenchStamped()
            self.get_body_wrench_timer = self.create_timer(self.dt / self.fast_forward, self.get_body_wrench)
        elif self.robot_model == 'Blue':
            self.wrench_from_thrusters_sub = self.create_subscription(WrenchStamped,'/ucat/force_thrust', self.WrenchfromThrusters, 10)
            self.wrench_from_thrusters = WrenchStamped()
            self.get_body_wrench_timer = self.create_timer(self.dt / self.fast_forward, self.get_body_wrench_thrust)
            self.reset_to_pose_srv = self.create_service(ResetToPose, '/reset_to_pose', self.reset_to_pose_callback)
        else:
            self.wrench_from_fins_sub = self.create_subscription(WrenchStamped,'/ucat/force_fins', self.onWrench, 10)

        # Publisher for RL agent
        self.state_publisher = self.create_publisher(EntityState, "/ucat/state", 10)
        self.is_domain_randomization = False
        if self.is_domain_randomization == True:
            self.setRandomizeParams()

        self.previous_reset_time = 0

        # Obscales

        self.obstacles_geometry = [0.0,0.0,-2.5,5.0,5.0,1.2]
        self.obstacles = []
        if self.with_obstacles:
            self.obstacles = [
                {"name": "obstacle_0000", "x": self.obstacles_geometry[0], "y": self.obstacles_geometry[1], "z": self.obstacles_geometry[2], "size_x": self.obstacles_geometry[3], "size_y": self.obstacles_geometry[4], "size_z": self.obstacles_geometry[5]}
            ]
            for obstacle in self.obstacles:
                self.create_obstacle(obstacle["name"], obstacle['x'], obstacle['y'], obstacle['z'], obstacle['size_x'] -0.6, obstacle['size_y'] -0.6, obstacle['size_z']-0.6)

        self.body_buffer = 0.05
        self.obstacles_name_list = ["obstacle_0000"]

    def setRandomizeParams(self):
        print("randomizing the dynamics parameters")
        self.COG_x = self.default_dynamics_parameters["dynamics"]["geometry"]["COG"][0] 
        self.COG_y = self.default_dynamics_parameters["dynamics"]["geometry"]["COG"][1] 
        self.COG_z = self.default_dynamics_parameters["dynamics"]["geometry"]["COG"][2] 
        self.COB_x = self.default_dynamics_parameters["dynamics"]["geometry"]["COB"][0] 
        self.COB_y = self.default_dynamics_parameters["dynamics"]["geometry"]["COB"][1] 
        self.COB_z = self.default_dynamics_parameters["dynamics"]["geometry"]["COB"][2]
        self.mass = self.default_dynamics_parameters["dynamics"]["inertial"]["mass"] 
        self.Ixx = self.default_dynamics_parameters["dynamics"]["inertial"]["Ixx"] 
        self.Iyy = self.default_dynamics_parameters["dynamics"]["inertial"]["Iyy"] 
        self.Izz = self.default_dynamics_parameters["dynamics"]["inertial"]["Izz"] 

    def generate_random_obstacles(self):
        # Clear existing obstacles
        for _name in self.obstacles_name_list:
            req = DeleteEntity.Request()
            req.name = _name #obstacle['name']
            self.delete_client.call_async(req)

        self.obstacles.clear()

        # Generate new obstacles with random positions and sizes
        num_obstacles =  1 # random.randint(1, 5)  # Random number of obstacles
        for i in range(num_obstacles):
            name = f'obstacle_{random.randint(1000,9999)}'
            r_x = random.uniform(-2.0, 2.0)
            r_y = random.uniform(-2.0, 2.0)
            r_z = random.uniform(-0.5, 0.5)
            r_size_x = random.uniform(-0.2, 2.0)
            r_size_y = random.uniform(-0.2, 2.0)
            r_size_z = random.uniform(-0.2, 0.5)
            x = self.obstacles_geometry[0] + r_x
            y = self.obstacles_geometry[1] + r_y
            z = self.obstacles_geometry[2] + r_z
            size_x = self.obstacles_geometry[3] + r_size_x
            size_y = self.obstacles_geometry[4] + r_size_y
            size_z = self.obstacles_geometry[5] + r_size_z
            self.obstacles.append({
                'name': name,
                'x': x,
                'y': y,
                'z': z,
                'size_x': size_x,
                'size_y': size_y,
                'size_z': size_z
            })
            self.obstacles_name_list.append(name)
            self.create_obstacle(name, x, y, z, size_x -0.6, size_y-0.6, size_z-1.0)
        
        # number of the name of the obstacles to keep 5 
        if len(self.obstacles_name_list) > 5:
            self.obstacles_name_list = self.obstacles_name_list[-5:]

    def create_obstacle(self, name, x, y, z, size_x, size_y, size_z):
        # Create the request message for obstacle
        sdf = f"""
        <?xml version="1.0" ?>
        <sdf version="1.6">
        <model name={name}>
            <static>true</static>
            <link name="link">
            <pose>0 0 0 0 0 0</pose>
            <collision name="collision">
                <geometry>
                <box>
                    <size>{size_x} {size_y} {size_z}</size>
                </box>
                </geometry>
                <surface>
                <contact>
                    <collide_without_contact>true</collide_without_contact>
                </contact>
                <bounce>
                    <restitution_coefficient>0.0</restitution_coefficient>
                </bounce>
                <friction>
                    <ode>
                    <mu>0.0</mu>
                    <mu2>0.0</mu2>
                    </ode>
                </friction>
                </surface>
            </collision>
            <visual name="visual">
                <geometry>
                <box>
                    <size>{size_x} {size_y} {size_z}</size>
                </box>
                </geometry>
                <material>
                <ambient>1 0 0 1</ambient>
                <diffuse>1 0 0 1</diffuse>
                </material>
            </visual>
            </link>
        </model>
        </sdf>
        """

        req = SpawnEntity.Request()
        req.name = name
        req.xml = sdf
        req.robot_namespace = 'obstacle'
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z

        # self.obstacles.append({
        #     'name': name,
        #     'x': x,
        #     'y': y,
        #     'z': z,
        #     'size_x': size_x,
        #     'size_y': size_y,
        #     'size_z': size_z
        # })

        future = self.spawn_client.call_async(req)
        future.add_done_callback(self._callback_spawn_entity)

    def onWrench(self, msg):
        self.U[0] = msg.wrench.force.x
        self.U[1] = msg.wrench.force.y
        self.U[2] = msg.wrench.force.z
        self.U[3] = msg.wrench.torque.x
        self.U[4] = msg.wrench.torque.y
        self.U[5] = msg.wrench.torque.z


    def WrenchfromWings(self, msg):
        self.wrench_from_wings = msg


    def WrenchfromThrusters(self, msg):
        self.wrench_from_thrusters = msg


    def get_body_wrench(self):
        self.U[0] = self.wrench_from_wings.wrench.force.x + self.wrench_from_thrusters.wrench.force.x
        self.U[1] = self.wrench_from_wings.wrench.force.y + self.wrench_from_thrusters.wrench.force.y
        self.U[2] = self.wrench_from_wings.wrench.force.z + self.wrench_from_thrusters.wrench.force.z
        self.U[3] = self.wrench_from_wings.wrench.torque.x + self.wrench_from_thrusters.wrench.torque.x
        self.U[4] = self.wrench_from_wings.wrench.torque.y + self.wrench_from_thrusters.wrench.torque.y
        self.U[5] = self.wrench_from_wings.wrench.torque.z + self.wrench_from_thrusters.wrench.torque.z
        # self._logger.info(f'U: {self.U}')
        # self._logger.info(f'wrench from wings: {self.wrench_from_wings}')
        # self._logger.info(f'wrench from thrusters: {self.wrench_from_thrusters}')

    def get_body_wrench_thrust(self):
        self.U[0] = self.wrench_from_thrusters.wrench.force.x
        self.U[1] = self.wrench_from_thrusters.wrench.force.y
        self.U[2] = self.wrench_from_thrusters.wrench.force.z
        self.U[3] = self.wrench_from_thrusters.wrench.torque.x
        self.U[4] = self.wrench_from_thrusters.wrench.torque.y
        self.U[5] = self.wrench_from_thrusters.wrench.torque.z
        #self._logger.info(f'U: {self.U}')
        #self._logger.info(f'wrench from thrusters: {self.wrench_from_thrusters}')



    def updateDynamics(self):
        self.dynamics.update(self.pos, self.vel)
        self.acc = self.dynamics.get_acc(self.vel, self.U)
        # Update body velocity
        self.vel += self.acc * self.dt

        # calculate global frame velocity
        pos_dot = self.dynamics.get_vel(self.vel)
        # Update Position
        self.pos += pos_dot * self.dt

        # Avoid going above surface
        if self.pos[2] < 0.02:
            self.pos[2] = 0.02
            self.vel[2] = 0.0

        self.update_state()

    def update_state(self):
        # Create the request message
        p_x = self.pos[0]
        p_y = -self.pos[1]
        p_z = -self.pos[2]

        collision_detected = False

        for obstacle in self.obstacles:
            x_min = obstacle['x'] - obstacle['size_x'] / 2
            x_max = obstacle['x'] + obstacle['size_x'] / 2
            y_min = obstacle['y'] - obstacle['size_y'] / 2
            y_max = obstacle['y'] + obstacle['size_y'] / 2
            z_min = obstacle['z'] - obstacle['size_z'] / 2
            z_max = obstacle['z'] + obstacle['size_z'] / 2

            # Check if inside obstacle
            if x_min <= p_x <= x_max and y_min <= p_y <= y_max and z_min <= p_z <= z_max:
                collision_detected = True
                distances = {
                    'left': abs(p_x - x_min),
                    'right': abs(p_x - x_max),
                    'bottom': abs(p_y - y_min),
                    'top': abs(p_y - y_max),
                    'back': abs(p_z - z_min),
                    'front': abs(p_z - z_max)
                }

                # Find the minimum distance to determine the closest face
                closest_face = min(distances, key=distances.get)

                if closest_face == 'left':
                    p_x = x_min - self.body_buffer
                    self.vel[0] = 0.0
                    self.acc[0] = 0.0
                    print("Collision with left face")
                elif closest_face == 'right':
                    p_x = x_max + self.body_buffer
                    self.vel[0] = 0.0
                    self.acc[0] = 0.0
                    print("Collision with right face")
                elif closest_face == 'bottom':
                    p_y = y_min - self.body_buffer
                    self.vel[1] = 0.0
                    self.acc[1] = 0.0
                    print("Collision with bottom face")
                elif closest_face == 'top':
                    p_y = y_max + self.body_buffer
                    self.vel[1] = 0.0
                    self.acc[1] = 0.0
                    print("Collision with top face")
                elif closest_face == 'back':
                    p_z = z_min - self.body_buffer
                    self.vel[2] = 0.0
                    self.acc[2] = 0.0
                    print("Collision with back face")
                elif closest_face == 'front':
                    p_z = z_max + self.body_buffer
                    self.vel[2] = 0.0
                    self.acc[2] = 0.0
                    print("Collision with front face")

        req = SetEntityState.Request()
        new_orientation_state = EntityState()
        new_orientation_state.name = "base_link"
        new_orientation_state.reference_frame = "world"

        self.pos[0] = p_x
        self.pos[1] = -p_y
        self.pos[2] = -p_z

        new_orientation_state.pose.position.x = p_x
        new_orientation_state.pose.position.y = p_y
        new_orientation_state.pose.position.z = p_z

        orientation = quaternion_from_euler(
                                        self.pos[3],
                                        -self.pos[4],
                                        -self.pos[5])

        new_orientation_state.pose.orientation.x = orientation[0]
        new_orientation_state.pose.orientation.y = orientation[1]
        new_orientation_state.pose.orientation.z = orientation[2]
        new_orientation_state.pose.orientation.w = orientation[3]

        new_orientation_state.twist.linear.x = self.vel[0]
        new_orientation_state.twist.linear.y = self.vel[1]
        new_orientation_state.twist.linear.z = self.vel[2]
        new_orientation_state.twist.angular.x = self.vel[3]
        new_orientation_state.twist.angular.y = self.vel[4]
        new_orientation_state.twist.angular.z = self.vel[5]
        
        # check nan
        if np.isnan(self.pos).any() or np.isnan(self.vel).any() or np.isnan(self.acc).any():
            self.get_logger().error("Nan value detected in AUV state!")
            return
        
        self.state_publisher.publish(new_orientation_state)

        req.state = new_orientation_state

        future_set_entity_state = self.srv_client.call_async(req)
        future_set_entity_state.add_done_callback(self._callback_set_entity_state_response)

    def reset_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds
        if current_time - self.previous_reset_time > 10000:
            self.previous_reset_time = current_time
            self.reset()

    def reset(self):
        self.pos = np.zeros(6)
        self.pos[2] = 5

        self.vel = np.zeros(6)
        self.acc = np.zeros(6)
        self.U = np.zeros(6)

        if self.is_domain_randomization == True:
            print("randomizing the dynamics parameters")
            self.dynamics_parameters["dynamics"]["geometry"]["COG"][0] = self.COG_x + np.random.uniform(-0.01, 0.01)
            self.dynamics_parameters["dynamics"]["geometry"]["COG"][1] = self.COG_y + np.random.uniform(-0.01, 0.01)
            self.dynamics_parameters["dynamics"]["geometry"]["COG"][2] = self.COG_z
            self.dynamics_parameters["dynamics"]["geometry"]["COB"][0] = self.COB_x
            self.dynamics_parameters["dynamics"]["geometry"]["COB"][1] = self.COB_y
            self.dynamics_parameters["dynamics"]["geometry"]["COB"][2] = self.COB_z
            self.dynamics_parameters["dynamics"]["inertial"]["mass"] = self.mass + np.random.uniform(0, 3)
            self.dynamics_parameters["dynamics"]["inertial"]["Ixx"] = self.Ixx + np.random.uniform(0.0, 0.01)
            self.dynamics_parameters["dynamics"]["inertial"]["Iyy"] = self.Iyy + np.random.uniform(0.0, 0.01)
            self.dynamics_parameters["dynamics"]["inertial"]["Izz"] = self.Izz + np.random.uniform(0.0, 0.01)
            print(f"default {self.default_dynamics_parameters}")
            print(f"randomized {self.dynamics_parameters}")
            self.dynamics = AUVDynamics(self.dynamics_parameters["dynamics"])

        reseted_state = EntityState()
        req = SetEntityState.Request()
        reseted_state.pose.position.x = self.pos[0]
        reseted_state.pose.position.y = -self.pos[1]
        reseted_state.pose.position.z = -self.pos[2]

        orientation = quaternion_from_euler(
                                        self.pos[3],
                                        -self.pos[4],
                                        -self.pos[5])

        reseted_state.pose.orientation.x = orientation[0]
        reseted_state.pose.orientation.y = orientation[1]
        reseted_state.pose.orientation.z = orientation[2]
        reseted_state.pose.orientation.w = orientation[3]

        reseted_state.twist.linear.x = self.vel[0]
        reseted_state.twist.linear.y = self.vel[1]
        reseted_state.twist.linear.z = self.vel[2]
        reseted_state.twist.angular.x = self.vel[3]
        reseted_state.twist.angular.y = self.vel[4]
        reseted_state.twist.angular.z = self.vel[5]

        self.get_logger().info("reset auv motion!")

        req.state = reseted_state

        depth_vel = Float32()
        depth_vel.data = 0.0

        self.state_publisher.publish(reseted_state)
        future_set_entity_state = self.srv_client.call_async(req)
        future_set_entity_state.add_done_callback(self._callback_set_entity_state_response)


        if self.with_obstacles:
            self.generate_random_obstacles()


    def _callback_set_entity_state_response(self,future:Future):
        try:
            response = future.result()
            #self.get_logger().info(f"Gazebo robot returns: {response}.")
        except Exception as e:
            self.get_logger().error(f"Call gazebo_set_entity_state encounter error: {e}")

    def _callback_spawn_entity(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Spawn status: {response.status_message}")
        except Exception as e:
            self.get_logger().error(f"Failed to spawn entity: {e}")

    def reset_to_pose_callback(self, request, response):
       
        self.pos = np.array([
            request.x,
            request.y,
            request.z,  
            request.roll,
            request.pitch, 
            request.yaw
        ])
        self.vel = np.zeros(6)
        self.acc = np.zeros(6)
        self.U = np.zeros(6)

        reseted_state = EntityState()
        reseted_state.name = "Blue"
        reseted_state.pose.position.x = self.pos[0]
        reseted_state.pose.position.y = -self.pos[1]
        reseted_state.pose.position.z = -self.pos[2]

        q = quaternion_from_euler(self.pos[3], -self.pos[4], -self.pos[5])
        reseted_state.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        reseted_state.twist.linear = Vector3()
        reseted_state.twist.angular = Vector3()

        # 
        self.state_publisher.publish(reseted_state)

        # 
        req = SetEntityState.Request()
        req.state = reseted_state
        future = self.srv_client.call_async(req)
        print(future)
        # rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response.success = True
            response.message = "ROV reset successful"
        else:
            response.success = False
            response.message = "Failed to call set_entity_state"

        return response


def main(args=None):
    rclpy.init(args=args)
    auv_motion = AUVMotion()
    rclpy.spin(auv_motion)
    auv_motion.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
