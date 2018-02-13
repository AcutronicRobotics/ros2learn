import argparse
import tensorflow as tf

from mpi4py import MPI
from rl_algs.common import set_global_seeds, tf_util as U
import os.path as osp
import gym, logging
import numpy as np
from collections import deque
from gym import spaces
import rl_algs.common.misc_util
import sys
import shutil
import subprocess
import gym_gazebo
import time

import time
import gym_gazebo

import mlsh_code.rollouts_robotics_mult as rollouts
from mlsh_code.policy_network import Policy
from mlsh_code.subpolicy_network import SubPolicy
from mlsh_code.observation_network import Features
from mlsh_code.learner import Learner
import rl_algs.common.tf_util as U
import pickle

sys.path.append('/home/rkojcev/devel/baselines')
from baselines.agent.scara_arm.agent_scara_real_mlsh import AgentSCARAROS
from baselines.agent.utility.general_utils import get_ee_points, get_position

import rclpy
from rclpy.qos import QoSProfile, qos_profile_sensor_data
# from hros_actuation_servomotor_msgs.msg import ServoMotorState, ServoMotorGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing scara joint angles.
from control_msgs.msg import JointTrajectoryControllerState
from ros2pkg.api import get_prefix_path

import argparse
import copy
import time


import csv


# from gym import utils
# from gym.envs.mujoco import mujoco_env

# here we define the parameters necessary to launch
savename = 'ScaraTest'
replay_bool= 'True'
macro_duration = 5
# num_subs = 4
num_subs = 2
num_rollouts = 2500
warmup_time = 1 #1 # 30
train_time = 2 #2 # 200
force_subpolicy=None
store=True

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

replay = str2bool(replay_bool)
# args.replay = str2bool(args.replay)

RELPATH = osp.join(savename)
LOGDIR = osp.join('/tmp/results' if sys.platform.startswith('linux') else '/tmp', RELPATH) #osp.join('/root/results' if sys.platform.startswith('linux') else '/tmp', RELPATH)
class ScaraJntsEnv(AgentSCARAROS):
    def __init__(self):
        self.init_3dof_robot()
        AgentSCARAROS.__init__(self)

        if MPI.COMM_WORLD.Get_rank() == 0 and osp.exists(LOGDIR):
            shutil.rmtree(LOGDIR)
        MPI.COMM_WORLD.Barrier()
        # with logger.session(dir=LOGDIR):
        self.load()
    def init_3dof_robot(self):
            print("I am in init function for the 3DoF")
            self.choose_robot = 0
            # Topics for the robot publisher and subscriber.
            JOINT_PUBLISHER = '/scara_controller/command'
            JOINT_SUBSCRIBER = '/scara_controller/state'

            EE_POS_TGT = np.asmatrix([0.3325683, 0.0657366, 0.3746]) # center of the O
            # EE_POS_TGT = np.asmatrix([0.3305805, -0.1326121, 0.3746]) # center of the H
            # EE_POS_TGT = np.asmatrix([0.3013209, 0.1647450, 0.3746]) # S top right
            # EE_POS_TGT = np.asmatrix( [0.2877867, -0.1005370, 0.3746]) # - middle
            # EE_POS_TGT = np.asmatrix([0.3349774, 0.1570571, 0.3746]) # S center

            # env.realgoal = [0.3341184, 0.0126104, 0.3746] # R middle right
            # env.realgoal = [0.3731659, -0.0065453, 0.3746] # R down right
            # env.realgoal = [0.2250708, -0.0422738, 0.3746] # R top left
            # EE_POS_TGT = np.asmatrix([0.2250708, -0.0422738, 0.3746]) # R top left

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

            TREE_PATH = "/home/rkojcev/catkin_ws/src/scara_e1/scara_e1_description/urdf/scara_e1_3joints.urdf"

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
            # slowness = 0.2
            slowness = 1

            m_joint_order = copy.deepcopy(JOINT_ORDER)
            m_link_names = copy.deepcopy(LINK_NAMES)
            m_joint_publishers = copy.deepcopy(JOINT_PUBLISHER)
            m_joint_subscribers = copy.deepcopy(JOINT_SUBSCRIBER)

            ee_pos_tgt = EE_POS_TGT
            ee_rot_tgt = EE_ROT_TGT

            # Initialize target end effector position
            ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)
            self.realgoal = ee_tgt

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
                'goal_vel': 0.03,
            }
    def start(callback,session, workerseed, rank, comm):
        env = gym.make('GazeboModularScara3DOF-v3')
        env.init_time(slowness= 6, slowness_unit='sec', reset_jnts=False)
        env.seed(workerseed)
        np.random.seed(workerseed)
        ob_space = env.observation_space
        ac_space = env.action_space
        stochastic= True
        stochastic_subpolicy=False

        # observation in.
        savedir = " "
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0]])
        policy = Policy(name="policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)
        old_policy = Policy(name="old_policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)

        sub_policies = [SubPolicy(name="sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]
        old_sub_policies = [SubPolicy(name="old_sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]

        learner = Learner(env, policy, old_policy, sub_policies, old_sub_policies, comm, clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-5, optim_batchsize=64)
        rollout = rollouts.traj_segment_generator(policy, sub_policies, env, macro_duration,num_rollouts, replay, savedir,force_subpolicy, stochastic=False)
        #

        callback(session)
        # learner.syncSubpolicies()
        policy.reset()
        shared_goal = comm.bcast(env.realgoal, root=0)
        print("The goal to %s" % (env.realgoal))
        print("which robot? ", env.choose_robot)
        obs=env.reset()
        print("OBS: ", obs)
        t = 0

        # time.sleep(1)
        new_test = True
        while True:
            # env.init_3dof_robot()
            #print("t", t)
            if t % macro_duration == 0:
                cur_subpolicy, macro_vpred = policy.act(stochastic, obs)

            # if np.random.uniform() < 0.1:
            #         cur_subpolicy = np.random.randint(0, len(sub_policies))
            # print("cur_subpolicy", cur_subpolicy)
            ac, vpred = sub_policies[cur_subpolicy].act(stochastic_subpolicy, obs)
            # print("obs[0:3]:",obs[0:3])
            # print("ac:",ac)

            nodes = np.array( [ [obs[0], obs[1], obs[2]], [ac[0], ac[1], ac[2]]] )
            j1 = nodes[:,0]
            j2 = nodes[:,1]
            j3 = nodes[:,2]
            tick,u = interpolate.splprep([j1,j2,j3], k=1)
            j1_new,j2_new, j3_new  = interpolate.splev( np.linspace( 0, 0.01, 10 ), tick,der = 0)


            ac_new = [j1_new[9], j2_new[9], j3_new[9]]
            # file_endeffector.write(str(ac[0]) + ", " + str(ac[1]) + ", " + str(ac[2]) + ", " +  str(ac_new[0]) + ", " + str(ac_new[1]) + ", " + str(ac_new[2]) + "\n")
            obs, rew, new, info = env.step(ac_new)

            t += 1

    def callback(session):

        print("CALLBACK")
        # fname = '/home/rkojcev/baselines_networks/mlsh_params_eval/macro_dur_5_warmup_time_5/00038'
        fname = '/home/rkojcev/baselines_networks/mlsh_params_eval/macro_dur_5_warmup_time_20/saved_models/00048'
        tf.train.Saver().restore(session, fname)

        pass

    def load():
        num_timesteps=1e9
        seed = 1401
        rank = MPI.COMM_WORLD.Get_rank()
        sess = U.single_threaded_session()
        sess.__enter__()
        workerseed = seed + 1000 * MPI.COMM_WORLD.Get_rank()
        rank = MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)

        # if rank != 0:
        #     logger.set_level(logger.DISABLED)
        # logger.log("rank %i" % MPI.COMM_WORLD.Get_rank())

        world_group = MPI.COMM_WORLD.Get_group()
        mygroup = rank % 10
        theta_group = world_group.Incl([x for x in range(MPI.COMM_WORLD.size) if (x % 10 == mygroup)])
        comm = MPI.COMM_WORLD.Create(theta_group)
        comm.Barrier()
        # comm = MPI.COMM_WORLD

        #master_robotics.start(callback, args=args, workerseed=workerseed, rank=rank, comm=comm)
        start(callback,sess, workerseed=workerseed, rank=rank, comm=comm)

if __name__ == '__main__':
    ScaraJntsEnv()
