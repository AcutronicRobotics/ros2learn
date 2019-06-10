"""
Script for testing that trpo experiment is working
properly by taking hyperparameters from defaults
"""

import os
import sys
import time
import gym
import gym_gazebo2
import tensorflow as tf

from baselines import bench, logger
from baselines.trpo_mpi import trpo_mpi, defaults
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.input import observation_placeholder
from baselines.common.models import mlp
from baselines.common.policies import build_policy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def make_env():
    env = gym.make(env_name)
    env.set_episode_size(alg_kwargs['timesteps_per_batch'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env

# Get dictionary from baselines/trpo_mpi/defaults
alg_kwargs = defaults.mara_mlp()
env_name = alg_kwargs['env_name']
alg_kwargs['total_timesteps'] = alg_kwargs['timesteps_per_batch']

# Generate tensorboard file
format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
logger.configure(os.path.abspath('/tmp/trpo'), format_strs)

env = DummyVecEnv([make_env])

# Remove unused parameters for training
alg_kwargs.pop('env_name')
alg_kwargs.pop('trained_path')
alg_kwargs.pop('transfer_path')

network = mlp(num_layers=alg_kwargs['num_layers'], num_hidden=alg_kwargs['num_hidden'], layer_norm=alg_kwargs['layer_norm'])

_ = trpo_mpi.learn(env=env, network=network, **alg_kwargs)

savedir = "/tmp/trpo/checkpoints/00000"
exists = os.path.isfile(savedir)
if not exists:
    raise AssertionError("Trained NN is missing")

tf.get_default_session().close()
tf.reset_default_graph()

env.dummy().gg2().close()

sess = U.get_session( config=tf.ConfigProto(
    allow_soft_placement = True,
    inter_op_parallelism_threads = 1,
    intra_op_parallelism_threads = 1) )

U.initialize()

env = DummyVecEnv([make_env])

set_global_seeds(alg_kwargs['seed'])

policy = build_policy(env, network, value_network='copy', **alg_kwargs)
obs_space = observation_placeholder(env.observation_space)
pi = policy(observ_placeholder=obs_space)
pi.load_var(savedir)

obs = env.reset()
assert obs is not None
assert env.dummy().gg2().obs_dim == len(obs[0])

actions = pi.step_deterministic(obs)[0]
assert len(actions[0]) == 6

obs, rew, done, info = env.step_runtime(actions)
assert (obs, rew, done) is not None

sess.close()
env.dummy().gg2().close()
os.kill(os.getpid(), 9)
