#!/usr/bin/env python3

import gym
import gym_gazebo_2
import tensorflow as tf
import copy
import sys
import numpy as np

from mpi4py import MPI

from baselines import bench, logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple_collisions
import os

# from baselines.common import tf_util as U

# from baselines.common.cmd_util import make_robotics_env, robotics_arg_parser
import time

env = gym.make('MARAOrientCollision-v0')
logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo1/'

logger.configure(os.path.abspath(logdir))
print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))


# env = Monitor(env, logger.get_dir(),  allow_early_resets=True)

rank = MPI.COMM_WORLD.Get_rank()
sess = U.single_threaded_session()
sess.__enter__()

seed = 0
workerseed = seed + 10000 * rank
set_global_seeds(seed)
env.seed(seed)


# seed = 0
# set_global_seeds(seed)

time.sleep(3)

hid_size = 128
num_hid_layers = 4
timesteps_per_actorbatch = 2048
optim_batchsize = 256

with open(logger.get_dir() + "/params.txt", 'a') as out:
   out.write( 'hid_size = ' + str(hid_size) + '\n'
               + 'num_hid_layers = ' + str(num_hid_layers) + '\n'
               + 'timesteps_per_actorbatch = ' + str(timesteps_per_actorbatch) + '\n'
               + 'optim_batchsize = ' + str(optim_batchsize) )

# initial_observation = env.reset()
# print("Initial observation: ", initial_observation)

# U.make_session(num_cpu=1).__enter__()


env.seed(seed)
def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=hid_size, num_hid_layers=num_hid_layers)

#optim_epochs=15

pposgd_simple_collisions.learn(env, policy_fn,
                    max_timesteps=1e8,
                    timesteps_per_actorbatch=timesteps_per_actorbatch,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=10, optim_stepsize=3e-4, gamma=0.99,
                    optim_batchsize=optim_batchsize, lam=0.95, schedule='linear', save_model_with_prefix='mara_orient_ppo1_test', outdir=logger.get_dir()) #

env.close()


# env.monitor.close()
