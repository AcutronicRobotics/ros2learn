
import numpy as np
import sys

import gym
import gym_gazebo_2

import tensorflow as tf

import argparse
import copy
import time

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

from baselines.acktr.acktr_cont import learn
from baselines.agent.utility.general_utils import get_ee_points, get_position
from baselines.ppo1 import mlp_policy, pposgd_simple


env = gym.make('RealMARA3DoF-v0')
initial_observation = env.reset()
print("Initial observation: ", initial_observation)
# env.render()
seed = 0

sess = U.make_session(num_cpu=1)
sess.__enter__()
def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    hid_size=64, num_hid_layers=3)
# gym.logger.setLevel(logging.WARN)
obs = env.reset()
print("Initial obs: ", obs)
# env.seed(seed)
# time.sleep(5)
pi = policy_fn('pi', env.observation_space, env.action_space)

tf.train.Saver().restore(sess, '/home/erle/MARA_NN/21-aug-2018/mara_orient_ppo1_test_afterIter_350.model')

done = False
while True:
    action = pi.act(False, obs)[0]
    obs, reward, done, info = env.step(action)
    print(reward)
    if reward > 0.98:
        for i in range(10):
            env.publish_last_position(obs)
            time.sleep(0.2)
        break
