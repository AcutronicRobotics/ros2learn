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
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

from baselines.acktr.acktr_cont import learn
from baselines.agent.utility.general_utils import get_ee_points, get_position
from baselines.ppo1 import mlp_policy, pposgd_simple

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

        # joint names:
        MOTOR1_JOINT = 'motor1'
        MOTOR2_JOINT = 'motor2'
        MOTOR3_JOINT = 'motor3'
        MOTOR4_JOINT = 'motor4'

        # Set constants for links
        BASE = 'scara_e1_base_link'
        BASE_MOTOR = 'scara_e1_base_motor'

        SCARA_MOTOR1 = 'scara_e1_motor1'
        SCARA_INSIDE_MOTOR1 = 'scara_e1_motor1_inside'
        SCARA_SUPPORT_MOTOR1 = 'scara_e1_motor1_support'
        SCARA_BAR_MOTOR1 = 'scara_e1_bar1'
        SCARA_FIXBAR_MOTOR1 = 'scara_e1_fixbar1'

        SCARA_MOTOR2 = 'scara_e1_motor2'
        SCARA_INSIDE_MOTOR2 = 'scara_e1_motor2_inside'
        SCARA_SUPPORT_MOTOR2 = 'scara_e1_motor2_support'
        SCARA_BAR_MOTOR2 = 'scara_e1_bar2'
        SCARA_FIXBAR_MOTOR2 = 'scara_e1_fixbar2'

        SCARA_MOTOR3 = 'scara_e1_motor3'
        SCARA_INSIDE_MOTOR3 = 'scara_e1_motor3_inside'
        SCARA_SUPPORT_MOTOR3 = 'scara_e1_motor3_support'
        SCARA_BAR_MOTOR3 = 'scara_e1_bar3'
        SCARA_FIXBAR_MOTOR3 = 'scara_e1_fixbar3'

        SCARA_MOTOR4 = 'scara_e1_motor4'
        SCARA_INSIDE_MOTOR4 = 'scara_e1_motor4_inside'
        SCARA_SUPPORT_MOTOR4 = 'scara_e1_motor4_support'
        SCARA_BAR_MOTOR4 = 'scara_e1_bar4'
        SCARA_FIXBAR_MOTOR4= 'scara_e1_fixbar4'

        SCARA_RANGEFINDER = 'scara_e1_rangefinder'

        EE_LINK = 'ee_link'
        JOINT_ORDER = [MOTOR1_JOINT, MOTOR2_JOINT, MOTOR3_JOINT, MOTOR4_JOINT]
        LINK_NAMES = [BASE, BASE_MOTOR,
              SCARA_MOTOR1, SCARA_INSIDE_MOTOR1, SCARA_SUPPORT_MOTOR1, SCARA_BAR_MOTOR1, SCARA_FIXBAR_MOTOR1,
              SCARA_MOTOR2, SCARA_INSIDE_MOTOR2, SCARA_SUPPORT_MOTOR2, SCARA_BAR_MOTOR2, SCARA_FIXBAR_MOTOR2,
              SCARA_MOTOR3, SCARA_INSIDE_MOTOR3, SCARA_SUPPORT_MOTOR3, SCARA_BAR_MOTOR3, SCARA_FIXBAR_MOTOR3,
              SCARA_MOTOR4, SCARA_INSIDE_MOTOR4, SCARA_SUPPORT_MOTOR4,
              EE_LINK]
        # Set end effector constants
        INITIAL_JOINTS = np.array([0, 0, 0, 0])
        # where is your urdf? We load here the 4 joints.... In the agent_scara we need to generalize it for joints depending on the input urdf

        prefix_path = get_prefix_path('scara_e1_description')

        if(prefix_path==None):
            print("I can't find scara_e1_description")
            sys.exit(0)

        TREE_PATH = prefix_path + "/share/scara_e1_description/urdf/scara_e1_4joints.urdf"

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

        self.test_ppo1(env,num_timesteps=1, seed=args.seed, save_model_with_prefix=args.save_model_with_prefix, restore_model_from_file=args.restore_model_from_file)

    def _observation_callback1(self, message):
        self._observation_msg =  message

    def test_ppo1(self,env, num_timesteps, seed, save_model_with_prefix, restore_model_from_file):

        sess = U.make_session(num_cpu=1)
        sess.__enter__()
        def policy_fn(name, ob_space, ac_space):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
        # gym.logger.setLevel(logging.WARN)

        obs = env.reset()
        print("Initial obs: ", obs)
        # env.seed(seed)
        pi = policy_fn('pi', env.observation_space, env.action_space)
        tf.train.Saver().restore(sess, '/media/erle/Datos/RISTO_NN/GazeboModularScara4DOFv3Env/ppo1/04dof_ppo1_test_O_afterIter_486.model')

        env.start_steps()
        while rclpy.ok():
            action = pi.act(False, obs)[0]
            obs, reward, done, info = env.step(action)
            # if env.reward > 0.99:
            #     break;
        rclpy.shutdown()

if __name__ == '__main__':
  ScaraJntsEnv()
