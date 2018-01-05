import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys
import numpy as np

from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple

from baselines import bench, logger
import os

env = gym.make('GazeboModularScara4DOF-v3')
logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo1/'
logger.configure(os.path.abspath(logdir))
print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))

initial_observation = env.reset()
print("Initial observation: ", initial_observation)
env.render()
seed = 0


U.make_session(num_cpu=1).__enter__()
set_global_seeds(seed)

env.seed(seed)
def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=2)

MeanEpRew = pposgd_simple.learn(env, policy_fn,
                    max_timesteps=1e6,
                    timesteps_per_actorbatch=2048,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=10, optim_stepsize=3e-4, gamma=0.99,
                    optim_batchsize=64, lam=0.95, schedule='linear', save_model_with_prefix='4dof_ppo1_test_H', outdir=logger.get_dir())
print(MeanEpRew)
