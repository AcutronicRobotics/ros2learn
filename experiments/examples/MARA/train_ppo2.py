import gym
import gym_gazebo2
import tensorflow as tf
import copy
import sys
import numpy as np

from baselines import bench, logger

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
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
import time

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
# def make_env(rank):

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
    env = gym.make('MARA-v0')
    # env.init_time(slowness= args.slowness, slowness_unit=args.slowness_unit, reset_jnts=args.reset_jnts)
    logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo2/'
    logger.configure(os.path.abspath(logdir))
    print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))
    # env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)
    # env.render()
    return env


nenvs = 1
# env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
env = DummyVecEnv([make_env])
env = VecNormalize(env)
alg='ppo2'
env_type = 'mara'
learn = get_learn_function('ppo2')
alg_kwargs = get_learn_function_defaults('ppo2', env_type)

seed = 0
set_global_seeds(seed)
network = 'mlp'
alg_kwargs['network'] = 'mlp'
rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

save_path =  '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo2/'

model = learn(env=env,
    seed=seed,
    total_timesteps=1e6, save_interval=10, **alg_kwargs) #, outdir=logger.get_dir()

if save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)
