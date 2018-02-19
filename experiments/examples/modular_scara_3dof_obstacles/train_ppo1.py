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
from baselines.ppo1 import mlp_policy, pposgd_simple_obstacles
import os


from baselines.common import tf_util as U
# parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slowness', help='time for executing trajectory', type=int, default=1)
parser.add_argument('--slowness-unit', help='slowness unit',type=str, default='sec')
parser.add_argument('--target', help='target', type=int, default=1)
parser.add_argument('--penalization', help='penalized-reward',type=int, default=0)
parser.add_argument('--mod', help='penalized-mod',type=int, default=100)
args = parser.parse_args()

target=1
pen_reward=1
# slowness = 1000000
# slowness_unit= 'nsec'
env = gym.make('GazeboModularScaraStaticObstacle3DOF-v1')
#env.init_time(slowness= args.slowness, slowness_unit=args.slowness_unit)
env.init_time(slowness= 10000000, slowness_unit='nsec')
env.setPenalizationMod(pen_mod=args.mod)
# env.set_target_and_reward(target=args.target, pen_reward=args.penalization)
print("MOD", args.mod)
logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo1/1_sec_th_0_05_'+str(args.mod)+'/'
#logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo1/obstacle_test/' + str(args.slowness) + '_' + str(args.slowness_unit) + '/'
# logdir = '/tmp/rosrl/GazeboModularScaraStaticObstacle3DOF-v1/ppo/'+str(target) + '_' + str(pen_reward) + '/'
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

env.addObstacle()


U.make_session(num_cpu=1).__enter__()


env.seed(seed)
def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=64, num_hid_layers=2)



pposgd_simple_obstacles.learn(env, policy_fn,
                    max_timesteps=1e6,
                    timesteps_per_actorbatch=2048,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=10, optim_stepsize=3e-4, gamma=0.99,
                    optim_batchsize=64, lam=0.95, schedule='linear', save_model_with_prefix='3dof_ppo1_test_H', outdir=logger.get_dir()) #

env.close()


# env.monitor.close()
