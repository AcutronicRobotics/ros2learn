#!/usr/bin/env python3

import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys
import numpy as np

from baselines import bench, logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple
import os


from baselines.common import tf_util as U
# parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slowness', help='time for executing trajectory', type=int, default=1)
parser.add_argument('--slowness-unit', help='slowness unit',type=str, default='sec')
args = parser.parse_args()

env = gym.make('MARANoGripper-v0')
env.init_time(slowness= args.slowness, slowness_unit=args.slowness_unit, reset_jnts=False)
logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo1/' + str(args.slowness) + '_' + str(args.slowness_unit) + '/'
# logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo1/'
logger.configure(os.path.abspath(logdir))
print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))
# env = Monitor(env, logger.get_dir(),  allow_early_resets=True)
seed = 0
env.seed(seed)
set_global_seeds(seed)
# RK: we are not using this for now but for the future left it here
# env = bench.MonitorRobotics(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True) #, allow_early_resets=True
initial_observation = env.reset()
print("Initial observation: ", initial_observation)
# env.render()



U.make_session(num_cpu=1).__enter__()


env.seed(seed)
def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=2)



pposgd_simple.learn(env, policy_fn,
                    max_timesteps=1e6,
                    timesteps_per_actorbatch=2048,
                    clip_param=0.1, entcoeff=0.0,
                    optim_epochs=20, optim_stepsize=3e-4, gamma=0.99,
                    optim_batchsize=64, lam=0.95, schedule='constant', save_model_with_prefix='mara_ppo1_test', outdir=logger.get_dir()) #

env.close()


# env.monitor.close()
