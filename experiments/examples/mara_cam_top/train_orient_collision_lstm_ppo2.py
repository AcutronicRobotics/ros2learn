import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys
import numpy as np

from baselines import bench, logger

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.common import policies, models, cmd_util
# from baselines.ppo2.policies import MlpPolicy, LstmPolicy, LnLstmPolicy, LstmMlpPolicy
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from baselines.common.cmd_util import common_arg_parser, parse_unknown_args

from importlib import import_module
import multiprocessing

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import os
import os.path as osp
import time

import threading # Used for time locks to synchronize position data.

import yaml
import glob

import quaternion as quat

import csv

take_point_once = 1

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn

def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def make_env():
    env = gym.make('MARAOrientCollision-v0')
    env.init_time(slowness= args.slowness, slowness_unit=args.slowness_unit, reset_jnts=args.reset_jnts)
    logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/lstm_ppo2/' + str(args.slowness) + '_' + str(args.slowness_unit) + '/'
    #new thing in openai to also log tensorboard and choose the formats you save in the logdir
    format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
    print(format_strs)
    # format_strs = 'stdout,log,csv,tensorboard'
    logger.configure(os.path.abspath(logdir), format_strs)
    print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))
    # env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)
    # env.render()
    return env

# parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slowness', help='time for executing trajectory', type=int, default=1)
parser.add_argument('--slowness-unit', help='slowness unit',type=str, default='sec')
parser.add_argument('--reset-jnts', help='reset the enviroment',type=bool, default=True)
args = parser.parse_args()

ncpu = multiprocessing.cpu_count()
if sys.platform == 'darwin': ncpu //= 2

print("ncpu: ", ncpu)
# ncpu = 1
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu,
                        log_device_placement=False)
config.gpu_options.allow_growth = True #pylint: disable=E1101

tf.Session(config=config).__enter__()

nenvs = 64
# env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
env = DummyVecEnv([make_env])
env = VecNormalize(env)
alg='ppo2'
env_type = 'modcobot'
nsteps = 2048


common_kwargs = dict(
    seed=0,
    total_timesteps = 1e8
)

learn_kwargs = {
'a2c': {},
'ppo2': dict(nsteps=nsteps, ent_coef=0.0, nminibatches=1)
}

alg_list = learn_kwargs.keys()
rnn_list = ['lstm']

#rnn_type = {'lstm':{'nlstm':128}}

# alg = 'ppo2'

kwargs = learn_kwargs[alg]
kwargs.update(common_kwargs)

# not relevant
episode_len = 1024

learn = lambda e: get_learn_function(alg)(
env=e,
network='lstm',
value_network = 'shared',
save_interval=10,
**kwargs
)

model, _ = learn(env)
