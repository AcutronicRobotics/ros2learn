#!/usr/bin/env python3

import gym
import gym_gazebo
import tensorflow as tf
import argparse
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
# parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slowness', help='time for executing trajectory', type=int, default=1000000)
parser.add_argument('--slowness-unit', help='slowness unit',type=str, default='nsec')
args = parser.parse_args()

env = gym.make('MARAOrientCollision-v0')
env.init_time(slowness= args.slowness, slowness_unit=args.slowness_unit, reset_jnts=False)
logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo1/' + str(args.slowness) + '_' + str(args.slowness_unit) + '/'

logger.configure(os.path.abspath(logdir))
print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))


# env = Monitor(env, logger.get_dir(),  allow_early_resets=True)

rank = MPI.COMM_WORLD.Get_rank()
sess = U.single_threaded_session()
sess.__enter__()

seed = 0
workerseed = seed + 10000 * rank
set_global_seeds(workerseed)
env.seed(seed)


# seed = 0
# set_global_seeds(seed)

env.goToInit()
time.sleep(3)

# initial_observation = env.reset()
# print("Initial observation: ", initial_observation)

# U.make_session(num_cpu=1).__enter__()


env.seed(seed)

hs = 128
nhl = 4
tpa = 2048
obz = 256

with open(logger.get_dir() + "/params.txt", 'a') as out:
    out.write( 'hid_size = ' + str(hs) + '\n'
                + 'num_hid_layers = ' + str(nhl) + '\n'
                + 'timesteps_per_actorbatch = ' + str(tpa) + '\n'
                + 'optim_batchsize = ' + str(obz) )

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=hs, num_hid_layers=nhl)

pposgd_simple_collisions.learn(env, policy_fn,
                    max_timesteps=1e8,
                    timesteps_per_actorbatch=tpa,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=10, optim_stepsize=3e-4, gamma=0.99,
                    optim_batchsize=obz, lam=0.95, schedule='linear', save_model_with_prefix='mara_orient_ppo1_test', outdir=logger.get_dir()) #

env.close()
# env.monitor.close()
