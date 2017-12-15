import numpy as np
import sys

from mpi4py import MPI

import gym
import gym_gazebo

import tensorflow as tf

import argparse
import copy

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines.trpo_mpi import trpo_mpi



env = gym.make('GazeboModularScara4DOF-v3')
initial_observation = env.reset()
print("Initial observation: ", initial_observation)
env.render()
seed = 0

rank = MPI.COMM_WORLD.Get_rank()
if rank != 0:
    logger.set_level(logger.DISABLED)
workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

sess = U.make_session(num_cpu=1)
sess.__enter__()
def policy_fn(name, ob_space, ac_space):
    return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
        hid_size=32, num_hid_layers=2)
obs = env.reset()
print("Initial obs: ", obs)
# env.seed(seed)
pi = policy_fn('pi', env.observation_space, env.action_space)
# tf.train.Saver().restore(sess, '/home/rkojcev/devel/baselines/baselines/experiments/ros1_trpo_test_H/saved_models/ros1_trpo_test_H_afterIter_20.model')
loadPath = '/tmp/rosrl/' + str(env.__class__.__name__) +'/trpo/'
tf.train.Saver().restore(sess, loadPath + 'ros1_trpo_H_afterIter_263.model')
done = False
while True:
    action = pi.act(True, obs)[0]
    obs, reward, done, info = env.step(action)
    print(action)
