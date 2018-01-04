import numpy as np
import sys

import tensorflow as tf

import argparse
import copy

sys.path.append('/home/rkojcev/devel/baselines')
from baselines.agent.scara_arm.agent_scara import AgentSCARAROS
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
from baselines.agent.utility.general_utils import get_ee_points, get_position


# from gym import utils
# from gym.envs.mujoco import mujoco_env

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
        # EE_POS_TGT = np.asmatrix([0.3325683, 0.0657366, 0.3746])
        EE_POS_TGT = np.asmatrix([0.3305805, -0.1326121, 0.4868]) # center of the H
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
        INITIAL_JOINTS = np.array([0, 0, 0])
        # where is your urdf? We load here the 4 joints.... In the agent_scara we need to generalize it for joints depending on the input urdf
        TREE_PATH = "/home/rkojcev/devel/ros_rl/environments/gym-gazebo/gym_gazebo/envs/assets/urdf/modular_scara/scara_e1_4joints.urdf"

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
        slowness = 1

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
        }
        AgentSCARAROS.__init__(self)

        # self.spec = {'timestep_limit': 5,
        # 'reward_threshold':  950.0,}
        self.max_episode_steps = 50

        env = self
        parser = argparse.ArgumentParser(description='Run Gazebo benchmark.')
        parser.add_argument('--seed', help='RNG seed', type=int, default=0)
        parser.add_argument('--save_model_with_prefix',
                            help='Specify a prefix name to save the model with after every iters. Note that this will generate multiple files (*.data, *.index, *.meta and checkpoint) with the same prefix', default='')
        parser.add_argument('--restore_model_from_file',
                            help='Specify the absolute path to the model file including the file name upto .model (without the .data-00000-of-00001 suffix). make sure the *.index and the *.meta files for the model exists in the specified location as well', default='')
        args = parser.parse_args()
        self.test_acktr(env,num_timesteps=1e6, seed=args.seed, save_model_with_prefix=args.save_model_with_prefix, restore_model_from_file=args.restore_model_from_file)

    def test_acktr(self,env, num_timesteps, seed, save_model_with_prefix, restore_model_from_file):
        # remove the seed
        # set_global_seeds(seed)
        # env.seed(seed)

        sess = U.make_session(num_cpu=1)
        sess.__enter__()
        # logger.session().__enter__()

        # # with tf.Session(config=tf.ConfigProto()) as session:
        # obs = []
        # acs = []
        # ac_dists = []
        # logps = []
        # rewards = []
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        # ob = env.reset()
        # prev_ob = np.float32(np.zeros(ob.shape))
        # state = np.concatenate([ob, prev_ob], -1)
        # obs.append(state)
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)
        tf.train.Saver().restore(sess, '/tmp/rosrl/GazeboModularScara4DOFv3Env/acktr/4dof_acktr_H_afterIter_282.model')
        while True:
            ob = env.reset()
            prev_ob = np.float32(np.zeros(ob.shape))

            obs = []
            acs = []
            ac_dists = []
            logps = []
            rewards = []

            state = np.concatenate([ob, prev_ob], -1)
            obs.append(state)
            ac, ac_dist, logp = policy.act(state)
            acs.append(ac)
            ac_dists.append(ac_dist)
            logps.append(logp)
            prev_ob = np.copy(ob)
            scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
            scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
            # ac, ac_dist, logp = policy.act(state)
            print(ac)
            ob, rew, done, _ = env.step(scaled_ac)
            # print("done is: ",done)

        #
        # learn(env, policy=policy, vf=vf,
        #     gamma=0.99, lam=0.97, timesteps_per_batch=2500,
        #     desired_kl=0.002,
        #     num_timesteps=num_timesteps, animate=False, save_model_with_prefix=save_model_with_prefix,restore_model_from_file=restore_model_from_file)


if __name__ == "__main__":
    ScaraJntsEnv()
