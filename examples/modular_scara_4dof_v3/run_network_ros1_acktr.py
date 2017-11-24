import numpy as np
import sys

import gym
import gym_gazebo

import tensorflow as tf

import argparse
import copy

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

# Use algorithms from baselines
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
from baselines.common import set_global_seeds


env = gym.make('GazeboModularScara4DOF-v3')
initial_observation = env.reset()
print("Initial observation: ", initial_observation)
env.render()
seed = 0

sess = U.make_session(num_cpu=1)
sess.__enter__()
# logger.session().__enter__()

# with tf.Session(config=tf.ConfigProto()) as session:
obs = []
acs = []
ac_dists = []
logps = []
rewards = []
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

loadPath = '/tmp/rosrl/' + str(env.__class__.__name__) +'_no_reset/acktr/'
tf.train.Saver().restore(sess, loadPath + '4dof_acktr_O_afterIter_397.model')
# tf.train.Saver().restore(sess, '/home/rkojcev/baselines_networks/ros1_acktr_H/saved_models/ros1_acktr_H_afterIter_263.model')
done = False
# ac, ac_dist, logp = policy.act(state)
# # print("action: ", ac)
# acs.append(ac)
# ac_dists.append(ac_dist)
# logps.append(logp)
# prev_ob = np.copy(ob)

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
    # here I need to figure out how to take non-biased action.
    # scaled_ac = env.action_space.low + (ac - 1.) * 0.5 * (env.action_space.high - env.action_space.low)
    # scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
    # scaled_ac =  ( ac + 1.5 + env.action_space.low + 0.5 *env.action_space.high)#( 0.5 * ac - (env.action_space.high + env.action_space.low)) / (env.action_space.high - env.action_space.low)
    # scaled_ac = np.clip(scaled_ac, env.action_space.low - 1, env.action_space.high +1)
    # scaled_ac = np.divide((2.0*ac -(env.action_space.low + env.action_space.high)),(env.action_space.high - env.action_space.low ))
    ob, rew, done, _ = env.step(scaled_ac)
