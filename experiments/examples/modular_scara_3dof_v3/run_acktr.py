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


env = gym.make('GazeboModularScara3DOF-v3')
env.init_time(slowness= 10, slowness_unit='sec', reset_jnts=False)
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
ob = env.reset()
prev_ob = np.float32(np.zeros(ob.shape))
state = np.concatenate([ob, prev_ob], -1)
obs.append(state)
with tf.variable_scope("vf"):
    vf = NeuralNetValueFunction(ob_dim, ac_dim)
with tf.variable_scope("pi"):
    policy = GaussianMlpPolicy(ob_dim, ac_dim)

# loadPath = '/tmp/rosrl/' + str(env.__class__.__name__) +'/acktr/'
# tf.train.Saver().restore(sess, loadPath + 'ros1_acktr_H_afterIter_263.model')
tf.train.Saver().restore(sess, '/home/rkojcev/baselines_networks/Networks_to_run/GazeboModularScara3DOFv3Env/acktr/3dof_acktr_H_afterIter_114.model')
done = False
ac, ac_dist, logp = policy.act(state)
# print("action: ", ac)
acs.append(ac)
ac_dists.append(ac_dist)
logps.append(logp)
prev_ob = np.copy(ob)

while True:
    ac, ac_dist, logp = policy.act(state)
    # here I need to figure out how to take non-biased action.
    scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
    scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
    # scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
    ob, rew, done, _ = env.step(scaled_ac)

    obs = []
    acs = []
    ac_dists = []
    logps = []
    rewards = []
    # ob_dim = env.observation_space.shape[0]
    # ac_dim = env.action_space.shape[0]
    # ob = env.reset()
    prev_ob = np.float32(np.zeros(ob.shape))
    state = np.concatenate([ob, prev_ob], -1)
    obs.append(state)
