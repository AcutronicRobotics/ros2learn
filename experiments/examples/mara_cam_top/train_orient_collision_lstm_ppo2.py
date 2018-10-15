import gym
import gym_gazebo
import tensorflow as tf
import argparse
import sys

from baselines import bench, logger

from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from importlib import import_module
import multiprocessing

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import os
import time

# parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slowness', help='time for executing trajectory', type=int, default=1000000)
parser.add_argument('--slowness-unit', help='slowness unit',type=str, default='nsec')
parser.add_argument('--reset-jnts', help='reset the enviroment',type=bool, default=True)
args = parser.parse_args()

ncpu = multiprocessing.cpu_count()
if sys.platform == 'darwin': ncpu //= 2
# print("ncpu: ", ncpu)

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


def get_learn_function(alg, submodule=None):
    return get_alg_module(alg, submodule).learn

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
    logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo2_lstm/' + str(args.slowness) + '_' + str(args.slowness_unit) + '/'
    format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)
    print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)
    return env

# nenvs = 1
# env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
env = DummyVecEnv([make_env])
env_type = 'mara_lstm'
learn = get_learn_function('ppo2')
alg_kwargs = get_learn_function_defaults('ppo2', env_type)

set_global_seeds(alg_kwargs['seed'])
rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

with open(logger.get_dir() + "/params.txt", 'a') as out:
    out.write(  'nlstm = ' + str(alg_kwargs['nlstm']) + '\n'
                + 'layer_norm = ' + str(alg_kwargs['layer_norm']) + '\n'
                + 'nsteps = ' + str(alg_kwargs['nsteps']) + '\n'
                + 'nminibatches = ' + str(alg_kwargs['nminibatches']) )

model = learn(env=env, **alg_kwargs) #, outdir=logger.get_dir()
