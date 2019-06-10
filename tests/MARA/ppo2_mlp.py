"""
Script for testing that ppo2_mlp experiment is working
properly by taking hyperparameters from defaults
"""

import os
import sys
import time
import gym
import gym_gazebo2
import tensorflow as tf
import multiprocessing

from importlib import import_module
from baselines import bench, logger
from baselines.ppo2 import ppo2
from baselines.ppo2 import model as ppo
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.policies import build_policy

ncpu = multiprocessing.cpu_count()

if sys.platform == 'darwin':
    ncpu //= 2

config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu,
                        log_device_placement=False)

config.gpu_options.allow_growth = True

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

def constfn(val):
    def f(_):
        return val
    return f

def make_env():
    env = gym.make(env_name)
    env.set_episode_size(alg_kwargs['nsteps'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env

# Get dictionary from baselines/ppo2/defaults
alg_kwargs = get_learn_function_defaults('ppo2', 'mara_mlp')
env_name = alg_kwargs['env_name']
alg_kwargs['total_timesteps'] = alg_kwargs['nsteps']

# Generate tensorboard file
format_strs = os.getenv('MARA_LOG_FORMAT', 'stdout,log,csv,tensorboard').split(',')
logger.configure(os.path.abspath('/tmp/ppo2_mlp'), format_strs)

env = DummyVecEnv([make_env])

# Remove unused parameters for training
alg_kwargs.pop('env_name')
alg_kwargs.pop('trained_path')
alg_kwargs.pop('transfer_path')

learn = get_learn_function('ppo2')

with tf.Session(config=config) as train_sess:
    _ = learn(env=env, **alg_kwargs)

tf.reset_default_graph()

savedir = "/tmp/ppo2_mlp/checkpoints/00001"
exists = os.path.isfile(savedir)
if not exists:
    raise AssertionError("Trained NN is missing")

env.dummy().gg2().close()

env = DummyVecEnv([make_env])

set_global_seeds(alg_kwargs['seed'])

if isinstance(alg_kwargs['lr'], float):
    alg_kwargs['lr'] = constfn(alg_kwargs['lr'])
else:
    assert callable(alg_kwargs['lr'])
if isinstance(alg_kwargs['cliprange'], float):
    alg_kwargs['cliprange'] = constfn(alg_kwargs['cliprange'])
else:
    assert callable(alg_kwargs['cliprange'])

nn ={ 'num_layers': alg_kwargs['num_layers'], 'num_hidden': alg_kwargs['num_hidden'] }
policy = build_policy(env, alg_kwargs['network'], **nn)
nenvs = env.num_envs
ob_space = env.observation_space
ac_space = env.action_space
nbatch = nenvs * alg_kwargs['nsteps']
nbatch_train = nbatch // alg_kwargs['nminibatches']

with tf.Session(config=config) as run_sess:

    make_model = lambda : ppo.Model(policy=policy, ob_space=ob_space,
                                    ac_space=ac_space, nbatch_act=nenvs,
                                    nbatch_train=nbatch_train,
                                    nsteps=alg_kwargs['nsteps'], ent_coef=alg_kwargs['ent_coef'], vf_coef=alg_kwargs['vf_coef'],
                                    max_grad_norm=alg_kwargs['max_grad_norm'])

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
