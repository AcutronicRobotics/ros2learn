import numpy as np
import sys

import gym
import gym_gazebo

import tensorflow as tf

import argparse
import copy
import time

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

from baselines.acktr.acktr_cont import learn
from baselines.agent.utility.general_utils import get_ee_points, get_position
from baselines.ppo1 import mlp_policy, pposgd_simple

# parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slowness', help='time for executing trajectory', type=int, default=1)
parser.add_argument('--slowness-unit', help='slowness unit',type=str, default='sec')
parser.add_argument('--target', help='target', type=int, default=1)
parser.add_argument('--penalization', help='penalized-reward',type=int, default=0)
parser.add_argument('--mod', help='penalized-mod',type=int, default=100)
args = parser.parse_args()


env = gym.make('GazeboModularScaraStaticObstacle3DOF-v1')
slowness = 2
slowness_unit= 'sec'
env.init_time(slowness= slowness, slowness_unit=slowness_unit)
env.setPenalizationMod(pen_mod=args.mod)
# initial_observation = env.reset()
# print("Initial observation: ", initial_observation)
env.addObstacle()
env.render()
seed = 0

sess = U.make_session(num_cpu=1)
sess.__enter__()
def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    hid_size=64, num_hid_layers=2)
# gym.logger.setLevel(logging.WARN)
# env.randomizeObstacle()
obs = env.reset()
print("Initial obs: ", obs)
# env.seed(seed)
# time.sleep(5)
pi = policy_fn("pi", env.observation_space, env.action_space)
tf.train.Saver().restore(sess, '/tmp/rosrl/GazeboModularScara3DOFStaticObstaclev1Env/ppo1/10000000_nsec/100/models/3dof_ppo1_test_H_afterIter_70.model') # for the H
done = False
while True:
    action = pi.act(True, obs)[0]
    obs, reward, done, info = env.step(action)
    print(action)
