#!/usr/bin/python3

"""
Node to conduct reinforcement learning to control U-CAT in Gazebo
@author: Yuya Hamamatsu 
@contact: yuya.hamamatsu@taltech.ee
"""
import time
import random
import numpy as np
import argparse
import pandas as pd

import yaml
import os

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import gym
from gym import spaces

from std_msgs.msg import Bool, Float32MultiArray
from gazebo_msgs.msg import EntityState

from tf_transformations import quaternion_from_euler


class LearningLocomotion(gym.Env):
    def __init__(self, fix_brake_fins, log_data_path):
        self.rosnode = SendActionROS2()
        super(LearningLocomotion, self).__init__()
        self.fix_fin_brake = fix_brake_fins
        self.log_data_path = log_data_path

        # logs setup
        self.df = pd.DataFrame(columns=["timestep", "n_episode", "broken_condition", "is_sucess", "success_rate", "reward"])
        self.n_episode = 0
        self.active_episode_log = 10
        self.success_rate = 0.0
        self.success_list = []

        self.rosnode.declare_parameter('rl_setting_yaml', 'rl_setting.yaml')
        yaml_rl = self.rosnode.get_parameter('rl_setting_yaml').value
        parameters_rl = os.path.join(
                get_package_share_directory('eeuv_sim'),
                'data', 'rl_setting',
                yaml_rl
                )
        with open(parameters_rl, 'r') as file:
            rl_parameters = yaml.load(file, Loader=yaml.FullLoader)
         

        alive_n_fins = 1
        self.movable_fins = [1, 1, 1, 1] # 1 if movable, 0 if not / FR, BR, BL, FL
        self.exception_fins = [[0,1,0,1], [0,1,0,0], [0,0,0,1]]

        if fix_brake_fins:
            self.movable_fins = fix_brake_fins
        else:
            self.init_fault_env()


        self.observation_dimansion = 9
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -5.0]),
                                        high=np.array([1.0, 1.0, 1.0, 1.0, 5.0]), dtype=np.float32)  # Wing angle x4 (degree), Thruster x1 (N)
        self.dt = 0.5

        self.observation_space = spaces.Box(low=-float("inf"),
                                            high=float("inf"),
                                            shape=(self.observation_dimansion,), 
                                            dtype=np.float32) 
        
        self.reset_msg = Bool()
        self.wings_cmd = Float32MultiArray()
        self.wings_cmd.data = [0.0, 0.0, 0.0, 0.0]
        self.thruster_cmd = Float32MultiArray()
        self.thruster_cmd.data = [0.0]

        #self.goal = [5, 0]
        #self.init_distance = self.__distance([0,0], self.goal)

        self.timesteps = 0 
        self.timestep_over = 200
        self.dt_optimize_gain = 0.05
        self.time_optimize_value = 0.0

        self.fast_forward = rl_parameters["rl"]["fast_forward"]

        self.is_send_reset = False

        self.depth_history = []
        self.stack_d_len = 30
        self.stack_depth = 0.1

    def brake_random_fins(self):
        self.movable_fins = []
        for _ in range(4):
            random_number = random.randint(0, 1)  
            self.movable_fins.append(random_number)
        

    def init_fault_env(self):
        self.brake_random_fins()
        while sum(self.movable_fins) == 0 or sum(self.movable_fins) == 4: # or self.movable_fins in self.exception_fins:
            self.brake_random_fins()

    def calc_log_data(self, reward, is_sucess):
        if self.timesteps == 0:
            return
        self.success_list.append(is_sucess)
        self.success_list = self.success_list[-100:]
        self.success_rate = sum(self.success_list) / len(self.success_list)
        self.df = pd.concat([self.df, pd.DataFrame([[self.timesteps, self.n_episode, str(self.movable_fins), is_sucess, self.success_rate, reward]], columns=self.df.columns)], ignore_index=True)
        if self.n_episode % self.active_episode_log == 0:
            self.df.to_csv(self.log_data_path, index=False)
            print(f"Saved log data at {self.log_data_path}")
        self.n_episode += 1


    def reset(self):
        self.reset_msg.data = True
        self.timesteps = 0
        if not self.fix_fin_brake:
            self.init_fault_env()
        if not self.is_send_reset:
            self.rosnode.reset_pub.publish(self.reset_msg)
            self.is_send_reset = True
        obs = np.zeros(self.observation_dimansion)

        #self.pre_dist = self.init_distance
        print("RESET!")
        return obs

    def calc_uprightness_reward(self, state):
        # calc R from quertanion
        R = np.array([[1 - 2 * (state.pose.orientation.y**2 + state.pose.orientation.z**2), 2 * (state.pose.orientation.x * state.pose.orientation.y - state.pose.orientation.z * state.pose.orientation.w), 2 * (state.pose.orientation.x * state.pose.orientation.z + state.pose.orientation.y * state.pose.orientation.w)],
                        [2 * (state.pose.orientation.x * state.pose.orientation.y + state.pose.orientation.z * state.pose.orientation.w), 1 - 2 * (state.pose.orientation.x**2 + state.pose.orientation.z**2), 2 * (state.pose.orientation.y * state.pose.orientation.z - state.pose.orientation.x * state.pose.orientation.w)],
                        [2 * (state.pose.orientation.x * state.pose.orientation.z - state.pose.orientation.y * state.pose.orientation.w), 2 * (state.pose.orientation.y * state.pose.orientation.z + state.pose.orientation.x * state.pose.orientation.w), 1 - 2 * (state.pose.orientation.x**2 + state.pose.orientation.y**2)]])
        
        gravity_vector = np.array([0, 0, -1]) 

        robot_up_vector = np.dot(R, gravity_vector)
        dot_product = np.dot(robot_up_vector, gravity_vector)
        cos_theta = dot_product / (np.linalg.norm(robot_up_vector) * np.linalg.norm(gravity_vector))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        reward = np.pi / 2 - angle

        # If uprightness is OK add reward

        return reward


    def step(self, action):
        start_time = time.perf_counter()
        self.pub_action(action)
    
        rclpy.spin_once(self.rosnode)


        try:
            time.sleep((self.dt / self.fast_forward ) + self.time_optimize_value) #0.0304 / self.fast_forward
        except:
            print("invalid time")
            time.sleep((self.dt / self.fast_forward ))

        state = self.rosnode.get_state()
        reward = self.reward_function(state)
        info = {}
        print(action)

        is_stacking = 0
        d = state.pose.position.z
        self.depth_history.append(d)
        if len(self.depth_history) > self.stack_d_len:
            self.depth_history.pop(0)
        # calc diff of max and min
        diff = abs(max(self.depth_history) - min(self.depth_history))
        if diff < self.stack_depth:
            is_stacking = 1
            print("STACKING")
        
        observation = [#state.pose.position.x, state.pose.position.y, 
                       state.twist.linear.z, 
                       state.pose.orientation.w, state.pose.orientation.x, state.pose.orientation.y, state.pose.orientation.z,
                       state.twist.angular.x, state.twist.angular.y, state.twist.angular.z,
                       is_stacking] # TODO update  

        #print(observation)
        over = self.is_episode_over(state, reward)
        episode_done = True if over and self.timesteps >50 else False

        self.timesteps += 1
        end_time = time.perf_counter()
        self.wait_time_optimizer(start_time, end_time)
        self.is_send_reset = False
        return observation, reward, episode_done, info
    
    
    def reward_function(self, state):
        upr = self.calc_uprightness_reward(state)
        depth_speed = - state.twist.linear.z
        reward = depth_speed * -2.5
        #angular_stability_penalty = (np.abs(state.twist.angular.x) *0.8 + np.abs(state.twist.angular.y)*0.8 + np.abs(state.twist.angular.z)) * 0.05
        linear_stability_penalty = (np.abs(state.twist.linear.x) + np.abs(state.twist.linear.y)) * 0.1
        uprightness_reward = (upr - 1.57) * 0.25
        
        #reward -= angular_stability_penalty
        #reward -= linear_stability_penalty
        reward += uprightness_reward
        print(uprightness_reward)
        if state.pose.position.z > -0.20:
            dist = (state.pose.position.x **2 + state.pose.position.y **2) ** 0.5
            dist_penalty = dist * 4
            time_penalty = (self.timesteps / self.timestep_over) * 20
            reward += 500 - (dist_penalty + time_penalty) 
        #self.pre_dist = dist
        print("timestep: " + str(self.timesteps))
        print("z vel: " + str(depth_speed))
        print(reward)
        return reward
 
    def is_episode_over(self, state, reward):
        cond_goal = state.pose.position.z > -0.20
        cond_timeover = self.timesteps > self.timestep_over
        
        cond = cond_goal or cond_timeover
        if cond:
            print(f"total reward {reward}")
            self.calc_log_data(reward, cond_goal)

        return cond
        
    def __distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def pub_action(self, action):
        for i in range(0, 4):
            if self.movable_fins[i] == 0:
                self.wings_cmd.data[i] = 0.0
                continue
            self.wings_cmd.data[i] = action[i] * 40.0 
        self.rosnode.wing_pub.publish(self.wings_cmd)

        self.thruster_cmd.data[0] = action[-1]
        self.rosnode.thrust_pub.publish(self.thruster_cmd)


    def wait_time_optimizer(self, start_time, end_time):
        dt_error = (self.dt / self.fast_forward) - (end_time - start_time)
        if dt_error < 0.1:
            self.time_optimize_value += self.dt_optimize_gain * dt_error
        

class SendActionROS2(Node):
    def __init__(self):
        super().__init__('send_action')
        self.state = EntityState()
        self.fin_angles = Float32MultiArray()

        self.reset_pub = self.create_publisher(Bool, '/ucat/reset', 10)
        self.thrust_pub = self.create_publisher(Float32MultiArray, '/ucat/thruster_cmd', 10)
        self.wing_pub = self.create_publisher(Float32MultiArray, '/ucat/wing_angle_cmd', 10)
        self.state_sub = self.create_subscription(EntityState, "/ucat/state", self.state_update_callback, 10)
        

    def reset_state(self):
        self.state = EntityState()

    def state_update_callback(self, msg):
        self.state = msg

    def get_state(self):
        return self.state
    
        
if __name__ == "__main__":
    from stable_baselines3.sac.policies import MlpPolicy
    from stable_baselines3 import A2C, PPO
    from sb3_contrib import RecurrentPPO
    parser = argparse.ArgumentParser(description='Train the model for U-CAT in Gazebo using RL')
    parser.add_argument('--model_path', type=str, help='Path to save the model')
    parser.add_argument('-c', "--continue_training", type=str, help='Continue training the model')
    parser.add_argument('-t', "--transfer_modoel", type=str, help='Transfer first layer weights from the model')

    tp = lambda x:list(map(int, x.split('.')))
    parser.add_argument("-f", "--fix_fins", type=tp, help="Fix which fin broken. 1 for movable, 0 for fixed as 1.1.1.1")

    args, _ = parser.parse_known_args()
    if args.fix_fins:
        str_brake = "".join(map(str, args.fix_fins))
    else:
        str_brake = "random"
    
    home_dir = os.path.expanduser("~")
    t = time.localtime()
    default_model_path = os.path.join(f"{home_dir}", "lauv_models", f"FT_LAUV_{str_brake}_{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.zip")
    log_data_path = os.path.join(f"{home_dir}", "lauv_models", f"FT_LAUV_{str_brake}_{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}_{t.tm_sec}.csv")



    # Use the provided video path or the default one
    model_path = args.model_path if args.model_path else default_model_path
    model_path_abs = os.path.abspath(model_path)

    print(f"Model will be saved at {model_path_abs}")

    rclpy.init()
    env = LearningLocomotion(args.fix_fins, log_data_path)

    if args.continue_training:
        model = RecurrentPPO.load(args.continue_training, env, verbose=1, tensorboard_log="./u-catlog/")
        print(f"Loaded model from {args.continue_training}")
    else:
        policy_kwargs = dict(
            lstm_hidden_size=64,
            n_lstm_layers=3,  
            net_arch=[64]
        )
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log="./u-catlog/", clip_range=0.20, policy_kwargs=policy_kwargs)
        #model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./u-catlog/", clip_range=0.20)

    if args.transfer_modoel:
        pre_train_model = RecurrentPPO.load(args.transfer_modoel)
        lstm_layer_actor = pre_train_model.policy.lstm_actor
        actor_first_layer_weights = lstm_layer_actor.weight_ih_l0.detach().clone()
        actor_first_layer_bias = lstm_layer_actor.bias_ih_l0.detach().clone()
        actor_second_layer_weights = lstm_layer_actor.weight_hh_l0.detach().clone()
        actor_second_layer_bias = lstm_layer_actor.bias_hh_l0.detach().clone()

        lstm_layer_critic = pre_train_model.policy.lstm_critic
        critic_first_layer_weights = lstm_layer_critic.weight_ih_l0.detach().clone()
        critic_first_layer_bias = lstm_layer_critic.bias_ih_l0.detach().clone()
        critic_second_layer_weights = lstm_layer_critic.weight_hh_l0.detach().clone()
        critic_second_layer_bias = lstm_layer_critic.bias_hh_l0.detach().clone()

        model.policy.lstm_actor.weight_ih_l0.data = actor_first_layer_weights
        model.policy.lstm_actor.bias_ih_l0.data = actor_first_layer_bias
        model.policy.lstm_critic.weight_ih_l0.data = critic_first_layer_weights
        model.policy.lstm_critic.bias_ih_l0.data = critic_first_layer_bias
        model.policy.lstm_actor.weight_hh_l0.data = actor_second_layer_weights
        model.policy.lstm_actor.bias_hh_l0.data = actor_second_layer_bias
        model.policy.lstm_critic.weight_hh_l0.data = critic_second_layer_weights
        model.policy.lstm_critic.bias_hh_l0.data = critic_second_layer_bias
        print(f"Transferred first layer weights from {args.transfer_modoel}")
        
        # Not freezing the first layer
        model.policy.lstm_actor.weight_ih_l0.requires_grad = True
        model.policy.lstm_actor.bias_ih_l0.requires_grad = True
        model.policy.lstm_critic.weight_ih_l0.requires_grad = True
        model.policy.lstm_critic.bias_ih_l0.requires_grad = True
        model.policy.lstm_actor.weight_hh_l0.requires_grad = True
        model.policy.lstm_actor.bias_hh_l0.requires_grad = True
        model.policy.lstm_critic.weight_hh_l0.requires_grad = True
        model.policy.lstm_critic.bias_hh_l0.requires_grad = True

    total_timesteps = 30000000
    save_interval = 5000

    for i in range(total_timesteps // save_interval):
        model.learn(total_timesteps=save_interval)
        model.save(model_path_abs)

        print(f"Saved model at step {(i+1) * save_interval}")
    
    model.save(model_path_abs)

