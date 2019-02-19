import os
import sys
import time
from datetime import datetime
import gym
import gym_gazebo2
import tensorflow as tf
import multiprocessing

from baselines import bench, logger
from baselines.trpo_mpi import trpo_mpi, defaults
from baselines.common import set_global_seeds
from baselines.common.models import mlp
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

ncpu = multiprocessing.cpu_count()

if sys.platform == 'darwin':
    ncpu //= 2

config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu,
                        log_device_placement=False)

config.gpu_options.allow_growth = True

tf.Session(config=config).__enter__()

def make_env():
    env = gym.make(alg_kwargs['env_name'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env

# Get dictionary from baselines/trpo_mpi/defaults
alg_kwargs = defaults.mara()

# Create needed folders
timedate = datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')
logdir = '/tmp/ros_rl2/' + alg_kwargs['env_name'] + '/trpo_mpi/' + timedate

# Generate tensorboard file
format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
logger.configure(os.path.abspath(logdir), format_strs)

with open(logger.get_dir() + "/parameters.txt", 'w') as out:
    out.write(
        'num_layers = ' + str(alg_kwargs['num_layers']) + '\n'
        + 'num_hidden = ' + str(alg_kwargs['num_hidden']) + '\n'
        + 'layer_norm = ' + str(alg_kwargs['layer_norm']) + '\n'
        + 'timesteps_per_batch = ' + str(alg_kwargs['timesteps_per_batch']) + '\n'
        + 'max_kl = ' + str(alg_kwargs['max_kl']) + '\n'
        + 'cg_iters = ' + str(alg_kwargs['cg_iters']) + '\n'
        + 'cg_damping = ' + str(alg_kwargs['cg_damping']) + '\n'
        + 'total_timesteps = ' + str(alg_kwargs['total_timesteps']) + '\n'
        + 'gamma = ' + str(alg_kwargs['gamma']) + '\n'
        + 'lam = ' + str(alg_kwargs['lam']) + '\n'
        + 'seed = ' + str(alg_kwargs['seed']) + '\n'
        + 'ent_coef = ' + str(alg_kwargs['ent_coef']) + '\n'
        + 'vf_iters = ' + str(alg_kwargs['vf_iters']) + '\n'
        + 'vf_stepsize = ' + str(alg_kwargs['vf_stepsize']) + '\n'
        + 'normalize_observations = ' + str(alg_kwargs['normalize_observations']) + '\n'
        + 'env_name = ' + alg_kwargs['env_name'] + '\n'
        + 'transfer_path = ' + str(alg_kwargs['transfer_path']) )

env = DummyVecEnv([make_env])

set_global_seeds(alg_kwargs['seed'])
MPI.COMM_WORLD.Get_rank()

transfer_path = alg_kwargs['transfer_path']

# Remove unused parameters for training
alg_kwargs.pop('env_name')
alg_kwargs.pop('trained_path')
alg_kwargs.pop('transfer_path')

network = mlp(num_layers=alg_kwargs['num_layers'], num_hidden=alg_kwargs['num_hidden'], layer_norm=alg_kwargs['layer_norm'])

if transfer_path is not None:
    # Do transfer learning
    trpo_mpi.learn(env=env, network=network, load_path=transfer_path, **alg_kwargs)
else:
    trpo_mpi.learn(env=env, network=network, **alg_kwargs)
