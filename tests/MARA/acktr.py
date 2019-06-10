"""
Script for testing that acktr experiment is working
properly by taking hyperparameters from defaults
"""

import os
import gym
import sys
import time
import gym_gazebo2
import tensorflow as tf

from baselines import bench, logger
from baselines.acktr import acktr, defaults
from baselines.common.models import mlp
from baselines.common.policies import build_policy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

config = tf.ConfigProto(allow_soft_placement=True,
                        inter_op_parallelism_threads = 1,
                        intra_op_parallelism_threads = 1)

def make_env():
    env = gym.make(env_name)
    env.set_episode_size(alg_kwargs['nsteps'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env

# Get dictionary from baselines/acktr/defaults
alg_kwargs = defaults.mara_mlp()
env_name = alg_kwargs['env_name']
alg_kwargs['total_timesteps'] = alg_kwargs['nsteps']

# Generate tensorboard file
format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
logger.configure(os.path.abspath('/tmp/acktr'), format_strs)

env = DummyVecEnv([make_env])

# Remove unused parameters for training
alg_kwargs.pop('env_name')
alg_kwargs.pop('trained_path')
alg_kwargs.pop('transfer_path')

network = mlp(num_layers=alg_kwargs['num_layers'], num_hidden=alg_kwargs['num_hidden'], layer_norm=alg_kwargs['layer_norm'])

with tf.Session(config=config) as train_sess:
    _ = acktr.learn(env=env, network=network, **alg_kwargs)

tf.reset_default_graph()

savedir = "/tmp/acktr/checkpoints/00001"
exists = os.path.isfile(savedir)
if not exists:
    raise AssertionError("Trained NN is missing")

env.dummy().gg2().close()

env = DummyVecEnv([make_env])
policy = build_policy(env, network, **alg_kwargs)

with tf.Session(config=config) as run_sess:
    make_model = lambda : acktr.Model(policy, env.observation_space, env.action_space, env.num_envs,
                            alg_kwargs['total_timesteps'], alg_kwargs['nprocs'], alg_kwargs['nsteps'], alg_kwargs['ent_coef'],
                            alg_kwargs['vf_coef'], alg_kwargs['vf_fisher_coef'], alg_kwargs['lr'], alg_kwargs['max_grad_norm'],
                            alg_kwargs['kfac_clip'], alg_kwargs['lrschedule'], alg_kwargs['is_async'])
model = make_model()
model.load(savedir)

obs = env.reset()
assert obs is not None
assert env.dummy().gg2().obs_dim == len(obs[0])

actions = model.step_deterministic(obs)[0]
assert len(actions[0]) == 6

obs, rew, done, _  = env.step_runtime(actions)
assert (obs, rew, done) is not None

env.dummy().gg2().close()
os.kill(os.getpid(), 9)
