import os
import gym
import sys
import time
from datetime import datetime
import gym_gazebo2

from baselines import bench, logger
from baselines.acktr import acktr, defaults
from baselines.common.models import mlp
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def make_env():
    env = gym.make(alg_kwargs['env_name'])
    env.set_episode_size(alg_kwargs['nsteps'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env

# Get dictionary from baselines/acktr/defaults
alg_kwargs = defaults.mara_mlp()

# Create needed folders
timedate = datetime.now().strftime('%Y-%m-%d_%Hh%Mmin')
logdir = '/tmp/ros2learn/' + alg_kwargs['env_name'] + '/acktr/' + timedate

# Generate tensorboard file
format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
logger.configure(os.path.abspath(logdir), format_strs)

with open(logger.get_dir() + "/parameters.txt", 'w') as out:
    out.write(
        'num_layers = ' + str(alg_kwargs['num_layers']) + '\n'
        + 'num_hidden = ' + str(alg_kwargs['num_hidden']) + '\n'
        + 'layer_norm = ' + str(alg_kwargs['layer_norm']) + '\n'
        + 'nsteps = ' + str(alg_kwargs['nsteps']) + '\n'
        + 'nprocs = ' + str(alg_kwargs['nprocs']) + '\n'
        + 'gamma = ' + str(alg_kwargs['gamma']) + '\n'
        + 'lam = ' + str(alg_kwargs['lam']) + '\n'
        + 'ent_coef = ' + str(alg_kwargs['ent_coef']) + '\n'
        + 'vf_coef = ' + str(alg_kwargs['vf_coef']) + '\n'
        + 'vf_fisher_coef = ' + str(alg_kwargs['vf_fisher_coef']) + '\n'
        + 'lr = ' + str(alg_kwargs['lr']) + '\n'
        + 'max_grad_norm = ' + str(alg_kwargs['max_grad_norm']) + '\n'
        + 'kfac_clip = ' + str(alg_kwargs['kfac_clip']) + '\n'
        + 'is_async = ' + str(alg_kwargs['is_async']) + '\n'
        + 'seed = ' + str(alg_kwargs['seed']) + '\n'
        + 'total_timesteps = ' + str(alg_kwargs['total_timesteps']) + '\n'
        # + 'network = ' + alg_kwargs['network'] + '\n'
        + 'value_network = ' + alg_kwargs['value_network'] + '\n'
        + 'lrschedule = ' + alg_kwargs['lrschedule'] + '\n'
        + 'log_interval = ' + str(alg_kwargs['log_interval']) + '\n'
        + 'save_interval = ' + str(alg_kwargs['save_interval']) + '\n'
        + 'env_name = ' + alg_kwargs['env_name'] + '\n'
        + 'transfer_path = ' + str(alg_kwargs['transfer_path']) )

env = DummyVecEnv([make_env])
transfer_path = alg_kwargs['transfer_path']

# Remove unused parameters for training
alg_kwargs.pop('env_name')
alg_kwargs.pop('trained_path')
alg_kwargs.pop('transfer_path')

network = mlp(num_layers=alg_kwargs['num_layers'], num_hidden=alg_kwargs['num_hidden'], layer_norm=alg_kwargs['layer_norm'])

if transfer_path is not None:
    # Do transfer learning
    _ = acktr.learn(env=env, network=network, load_path=transfer_path, **alg_kwargs)
else:
    _ = acktr.learn(env=env, network=network, **alg_kwargs)

env.dummy().gg2().close()
os.kill(os.getpid(), 9)
