#!/usr/bin/python3

import rclpy
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from hros_actuation_servomotor_msgs.msg import ServoMotorState, ServoMotorGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing scara joint angles.
from control_msgs.msg import JointTrajectoryControllerState
from ros2pkg.api import get_prefix_path

import numpy as np
import sys

import tensorflow as tf

import argparse
import copy
import time

sys.path.append('/media/erle/Datos/RISTO_NN/baselines')
from baselines.agent.scara_arm.agent_scara_real import AgentSCARAROS
from baselines import benc,logger
from baselines.common import set_global_seeds, tf_util as U


from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import joblib

import multiprocessing

import os

class ScaraJntsEnv(AgentSCARAROS):

    # agent_scara.AgentSCARAROS.__init__(self, 'tests.xml')

    def __init__(self):
        print("I am in init function")
        # Too much hard coded stuff in here, especially the joint names and the motor names.
        # TODO: see with KDL we can fetch the base and the end-effector for the FK kinematics.
        # That way we eliminate all of the parameters. In here ideally we should only have the end goal and the names of the topics, regarding ROS

        # Topics for the robot publisher and subscriber.
        JOINT_PUBLISHER = '/scara_controller/command'
        JOINT_SUBSCRIBER = '/scara_controller/state'
        # where should the agent reach, in this case the middle of the O letter in H-ROS
        EE_POS_TGT = np.asmatrix([0.3325683, 0.0657366, 0.4868]) # center of O
        #EE_POS_TGT = np.asmatrix([0.3305805, -0.1326121, 0.4868]) # center of the H
        # EE_POS_TGT = np.asmatrix([0.3305805, -0.1326121, 0.4868]) # center of the H
        EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_POINTS = np.asmatrix([[0, 0, 0]])
        EE_VELOCITIES = np.asmatrix([[0, 0, 0]])

        MOTOR1_JOINT = 'motor1'
        MOTOR2_JOINT = 'motor2'
        MOTOR3_JOINT = 'motor3'
        # Set constants for links
        WORLD = "world"
        BASE = 'scara_e1_base_link'
        BASE_MOTOR = 'scara_e1_base_motor'
        #
        SCARA_MOTOR1 = 'scara_e1_motor1'
        SCARA_INSIDE_MOTOR1 = 'scara_e1_motor1_inside'
        SCARA_SUPPORT_MOTOR1 = 'scara_e1_motor1_support'
        SCARA_BAR_MOTOR1 = 'scara_e1_bar1'
        SCARA_FIXBAR_MOTOR1 = 'scara_e1_fixbar1'
        #
        SCARA_MOTOR2 = 'scara_e1_motor2'
        SCARA_INSIDE_MOTOR2 = 'scara_e1_motor2_inside'
        SCARA_SUPPORT_MOTOR2 = 'scara_e1_motor2_support'
        SCARA_BAR_MOTOR2 = 'scara_e1_bar2'
        SCARA_FIXBAR_MOTOR2 = 'scara_e1_fixbar2'
        #
        SCARA_MOTOR3 = 'scara_e1_motor3'
        SCARA_INSIDE_MOTOR3 = 'scara_e1_motor3_inside'
        SCARA_SUPPORT_MOTOR3 = 'scara_e1_motor3_support'
        SCARA_BAR_MOTOR3 = 'scara_e1_bar3'
        SCARA_FIXBAR_MOTOR3 = 'scara_e1_fixbar3'
        #
        SCARA_RANGEFINDER = 'scara_e1_rangefinder'
        EE_LINK = 'ee_link'
        JOINT_ORDER = [MOTOR1_JOINT, MOTOR2_JOINT, MOTOR3_JOINT]
        LINK_NAMES = [BASE, BASE_MOTOR,
                      SCARA_MOTOR1, SCARA_INSIDE_MOTOR1, SCARA_SUPPORT_MOTOR1, SCARA_BAR_MOTOR1, SCARA_FIXBAR_MOTOR1,
                      SCARA_MOTOR2, SCARA_INSIDE_MOTOR2, SCARA_SUPPORT_MOTOR2, SCARA_BAR_MOTOR2, SCARA_FIXBAR_MOTOR2,
                      SCARA_MOTOR3, SCARA_INSIDE_MOTOR3, SCARA_SUPPORT_MOTOR3,
                      EE_LINK]
        # Set end effector constants
        INITIAL_JOINTS = np.array([0, 0, 0])
        # where is your urdf? We load here the 4 joints.... In the agent_scara we need to generalize it for joints depending on the input urdf

        prefix_path = get_prefix_path('scara_e1_description')

        if(prefix_path==None):
            print("I can't find scara_e1_description")
            sys.exit(0)

        TREE_PATH = prefix_path + "/share/scara_e1_description/urdf/scara_e1_3joints.urdf"

        reset_condition = {
            'initial_positions': INITIAL_JOINTS,
             'initial_velocities': []
        }

        STEP_COUNT = 2  # Typically 100.

        # Set the number of seconds per step of a sample.
        TIMESTEP = 0.01  # Typically 0.01.
        # Set the number of timesteps per sample.
        STEP_COUNT = 100  # Typically 100.
        # Set the number of samples per condition.
        SAMPLE_COUNT = 5  # Typically 5.
        # set the number of conditions per iteration.
        # Set the number of trajectory iterations to collect.
        ITERATIONS = 20  # Typically 10.
        slowness = 0.2

        m_joint_order = copy.deepcopy(JOINT_ORDER)
        m_link_names = copy.deepcopy(LINK_NAMES)
        m_joint_publishers = copy.deepcopy(JOINT_PUBLISHER)
        m_joint_subscribers = copy.deepcopy(JOINT_SUBSCRIBER)

        ee_pos_tgt = EE_POS_TGT
        ee_rot_tgt = EE_ROT_TGT

        # Initialize target end effector position
        ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)

        self.agent = {
            'type': AgentSCARAROS,
            'dt': TIMESTEP,
            'T': STEP_COUNT,
            'ee_points_tgt': ee_tgt,
            'joint_order': m_joint_order,
            'link_names': m_link_names,
            'slowness': slowness,
            'reset_conditions': reset_condition,
            'tree_path': TREE_PATH,
            'joint_publisher': m_joint_publishers,
            'joint_subscriber': m_joint_subscribers,
            'end_effector_points': EE_POINTS,
            'end_effector_velocities': EE_VELOCITIES,
            'num_samples': SAMPLE_COUNT,
            'goal_vel': 0.05,
        }

        AgentSCARAROS.__init__(self)

        # self.spec = {'timestep_limit': 5,
        # 'reward_threshold':  950.0,}
        self.max_episode_steps = 50

        parser = argparse.ArgumentParser(description='Run Gazebo benchmark.')
        parser.add_argument('--seed', help='RNG seed', type=int, default=0)
        parser.add_argument('--save_model_with_prefix',
                            help='Specify a prefix name to save the model with after every iters. Note that this will generate multiple files (*.data, *.index, *.meta and checkpoint) with the same prefix', default='')
        parser.add_argument('--restore_model_from_file',
                            help='Specify the absolute path to the model file including the file name upto .model (without the .data-00000-of-00001 suffix). make sure the *.index and the *.meta files for the model exists in the specified location as well', default='')
        args = parser.parse_args()

        env = self

        self.test_ppo2(env,num_timesteps=1, seed=args.seed, save_model_with_prefix=args.save_model_with_prefix, restore_model_from_file=args.restore_model_from_file)

    def _observation_callback1(self, message):
        self._observation_msg =  message

    def make_env():
        # env.init_time(slowness= 10, slowness_unit='sec', reset_jnts=False)
        env = bench.MonitorRobotics(env, logger.get_dir(), allow_early_resets=True)
        env.render()
        return env

    def test_ppo2(self,env, num_timesteps, seed, save_model_with_prefix, restore_model_from_file):
        ncpu = multiprocessing.cpu_count()
        if sys.platform == 'darwin': ncpu //= 2

        print("ncpu: ", ncpu)
        # ncpu = 1
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=ncpu,
                                inter_op_parallelism_threads=ncpu,
                                log_device_placement=False)
                                
        config.gpu_options.allow_growth = True #pylint: disable=E1101

        sess = tf.Session(config=config)

        sess.__enter__()
        env = DummyVecEnv([self.make_env])
        env = VecNormalize(env)

        nenvs = env.num_envs
        ob_space = env.observation_space
        ac_space = env.action_space
        nsteps = 1 # default
        nbatch = nenvs * nsteps
        nminibatches=4

        nbatch_train = nbatch // nminibatches
        vf_coef=0.5
        max_grad_norm=0.5
        ent_coef=0.0

        gamma=0.99
        lam=0.95

        dones = [False for _ in range(nenvs)]

        policy = MlpPolicy
        make_model = lambda : ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm)

        model = make_model()

        model.load('/home/rkojcev/baselines_networks/paper/data/GazeboModularScara3DOFv3Env_diff_times/ppo2/10000000_nsec/checkpoints/00410')

        runner = ppo2.Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
        obs = np.zeros((nenvs,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        # obs = np.zeros((nenvs,) + env.observation_space.shape, dtype=model.X.dtype.name)
        obs[:] = env.reset()
        print("Initial obs: ", obs)
        obs_extended = np.tile(obs, (nsteps,1))

        # print(obs_extended.shape)
        #
        done = False
        env.start_steps()
        while rclpy.ok():
            actions, values, states, neglogpacs = model.step(obs)
            obs, reward, done, info = env.step(actions)
        rclpy.shutdown()

if __name__ == '__main__':
  ScaraJntsEnv()
