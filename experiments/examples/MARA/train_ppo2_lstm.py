import os
import sys
import time
from datetime import datetime
import gym
import gym_gazebo2
import tensorflow as tf
import multiprocessing

from importlib import import_module
from baselines import bench, logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv, ShmemVecEnv, SubprocVecEnv, DummyVecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env, make_mujoco_env
from baselines.common.tf_util import get_session

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

config = tf.ConfigProto(allow_soft_placement=True,
                       intra_op_parallelism_threads=1,
                       inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
get_session(config=config)

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
    env = gym.make(alg_kwargs['env_name'])
    # env.set_episode_size(alg_kwargs['nsteps'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env

def make_thunk(rank, initializer=None):
    return lambda: make_env(
        env_id=env_id,
        env_type=env_type,
        mpi_rank=mpi_rank,
        subrank=rank,
        seed=seed,
        reward_scale=reward_scale,
        gamestate=gamestate,
        flatten_dict_observations=flatten_dict_observations,
        wrapper_kwargs=wrapper_kwargs,
        env_kwargs=env_kwargs,
        logger_dir=logger_dir,
        initializer=initializer
    )

#
def main():
    # configure logger, disable logging in child MPI processes (with rank > 0)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    # Get dictionary from baselines/ppo2/defaults
    env_type = 'mara_lstm'
    alg_kwargs = get_learn_function_defaults('ppo2', env_type)

    # Create needed folders
    timedate = datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')
    logdir = '/tmp/ros2learn/' + alg_kwargs['env_name'] + '/ppo2_lstm/' + timedate

    # Generate tensorboard file
    format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
    logger.configure(os.path.abspath(logdir), format_strs)

    with open(logger.get_dir() + "/parameters.txt", 'w') as out:
        out.write(
            'nlstm = ' + str(alg_kwargs['nlstm']) + '\n'
            + 'layer_norm = ' + str(alg_kwargs['layer_norm']) + '\n'
            + 'nsteps = ' + str(alg_kwargs['nsteps']) + '\n'
            + 'nminibatches = ' + str(alg_kwargs['nminibatches']) + '\n'
            + 'lam = ' + str(alg_kwargs['lam']) + '\n'
            + 'gamma = ' + str(alg_kwargs['gamma']) + '\n'
            + 'noptepochs = ' + str(alg_kwargs['noptepochs']) + '\n'
            + 'log_interval = ' + str(alg_kwargs['log_interval']) + '\n'
            + 'ent_coef = ' + str(alg_kwargs['ent_coef']) + '\n'
            + 'cliprange = ' + str(alg_kwargs['cliprange']) + '\n'
            + 'vf_coef = ' + str(alg_kwargs['vf_coef']) + '\n'
            + 'max_grad_norm = ' + str(alg_kwargs['max_grad_norm']) + '\n'
            + 'seed = ' + str(alg_kwargs['seed']) + '\n'
            + 'value_network = ' + alg_kwargs['value_network'] + '\n'
            + 'network = ' + alg_kwargs['network'] + '\n'
            + 'total_timesteps = ' + str(alg_kwargs['total_timesteps']) + '\n'
            + 'save_interval = ' + str(alg_kwargs['save_interval']) + '\n'
            + 'env_name = ' + alg_kwargs['env_name'] + '\n'
            + 'num_envs = ' + str(alg_kwargs['num_envs']) + '\n'
            + 'transfer_path = ' + str(alg_kwargs['transfer_path']) )

    # if alg_kwargs['num_envs'] > 1:
        # fns = [make_env for _ in range(alg_kwargs['num_envs'])]
    #     env = SubprocVecEnv(fns)
    # else:
    #     env = DummyVecEnv([make_env])

    # seed = alg_kwargs['seed']
    # # env = make_mujoco_env(alg_kwargs['env_name'],  seed, reward_scale=1.0)
    # fns = [make_env for _ in range(alg_kwargs['num_envs'])]

    env_id = env_type
    mpi_rank  = 0
    start_index = 0
    seed = 0
    reward_scale = 1.0
    gamestate=None
    flatten_dict_observations=True
    wrapper_kwargs=None
    env_kwargs=None
    logger_dir=None
    initializer=None

    # num_env = alg_kwargs['num_envs']
    # if num_env > 1:
    #     fns = [make_env for _ in range(alg_kwargs['num_envs'])]
    #     env = SubprocVecEnv(fns)

    env = make_vec_env(alg_kwargs['env_name'], env_type, alg_kwargs['num_envs'] or 1, seed, reward_scale=1.0)

    # if env_type == 'mujoco':
    #     env = VecNormalize(env, use_tf=True)

    learn = get_learn_function('ppo2')
    transfer_path = alg_kwargs['transfer_path']

    # Remove unused parameters for training
    alg_kwargs.pop('env_name')
    alg_kwargs.pop('num_envs')
    alg_kwargs.pop('transfer_path')
    alg_kwargs.pop('trained_path')

    if transfer_path is not None:
        # Do transfer learning
        learn(env=env,load_path=transfer_path, **alg_kwargs)
    else:
        learn(env=env, **alg_kwargs)

if __name__ == '__main__':
    main()
