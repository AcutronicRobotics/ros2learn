import gym
import gym_gazebo
import tensorflow as tf
import argparse
import sys
import numpy as np

from baselines import bench, logger

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2, model
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from baselines.common import set_global_seeds
from baselines.common.policies import build_policy

from importlib import import_module
import multiprocessing

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import os
import time

import write_csv as csv_file

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

def initialize_placeholders(nlstm=256,**kwargs):
    return np.zeros((num_env or 1, 2*nlstm)), np.zeros((1))

def constfn(val):
    def f(_):
        return val
    return f

def make_env():
    env = gym.make('MARAOrientCollision-v0')
    env.init_time(slowness= args.slowness, slowness_unit=args.slowness_unit, reset_jnts=args.reset_jnts)
    logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo2_lstm/' + str(args.slowness) + '_' + str(args.slowness_unit) + '/'
    logger.configure(os.path.abspath(logdir))
    print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))
    # env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)
    # env.render()
    return env

# parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slowness', help='time for executing trajectory', type=int, default=1)
parser.add_argument('--slowness-unit', help='slowness unit',type=str, default='sec')
parser.add_argument('--reset-jnts', help='reset the enviroment',type=bool, default=False)
args = parser.parse_args()

ncpu = multiprocessing.cpu_count()
if sys.platform == 'darwin': ncpu //= 2

config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu,
                        log_device_placement=False)
config.gpu_options.allow_growth = True #pylint: disable=E1101

tf.Session(config=config).__enter__()
# def make_env(rank):

# env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
env = DummyVecEnv([make_env])
env = VecNormalize(env, ob=True, ret=False)
alg='ppo2'
env_type = 'mara_lstm'
defaults = get_learn_function_defaults('ppo2', env_type)
# tricks to run it faster and if it has been trained for many enviroment instance to launch only one.
defaults['nsteps'] = 1
defaults['nminibatches'] = 1


alg_kwargs ={
'nlstm': defaults['nlstm'],
'layer_norm': defaults['layer_norm']
}

set_global_seeds(defaults['seed'])
rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

if isinstance(defaults['lr'], float):
    defaults['lr'] = constfn(defaults['lr'])
else:
    assert callable(defaults['lr'])
if isinstance(defaults['cliprange'], float):
    defaults['cliprange'] = constfn(defaults['cliprange'])
else:
    assert callable(defaults['cliprange'])

# defaults['nsteps'] = 1
# defaults['nminibatches'] = 1

policy = build_policy(env, defaults['network'], **alg_kwargs)

nenvs = env.num_envs
ob_space = env.observation_space
ac_space = env.action_space
nbatch = nenvs * defaults['nsteps']
nbatch_train = nbatch // defaults['nminibatches']
global num_env
num_env = 1


# dones = [False for _ in range(nenvs)]

# load_path='/tmp/rosrl/GazeboMARATopOrientCollisionv0Env/ppo2_lstm/1000000_nsec/checkpoints/01830'
# load_path = '/tmp/rosrl/GazeboMARATopOrientCollisionv0Env/ppo2_lstm/1000000_nsec/checkpoints/00360'

# load_path = '/media/rkojcev/Data_Networks/ppo2_lstm/21112018/openai-2018-11-22-13-25-21-339062/checkpoints/00360'
load_path = '/media/rkojcev/Data_Networks/ppo2_lstm/23112018/test/1000000_nsec/checkpoints/00260'

make_model = lambda : model.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                nsteps=defaults['nsteps'], ent_coef=defaults['ent_coef'], vf_coef=defaults['vf_coef'],
                max_grad_norm=defaults['max_grad_norm'])

model = make_model()
if load_path is not None:
    print("Loading model from: ", load_path)
    model.load(load_path)

obs = env.reset()

csv_obs_path = "csv/ppo2_det_obs.csv"
csv_acs_path = "csv/ppo2_det_acs.csv"
csv_rew_path = "csv/ppo2_det_rew.csv"
# csv_obs_path = "csv/ppo2_sto_obs.csv"
# csv_acs_path = "csv/ppo2_sto_acs.csv"
# csv_rew_path = "csv/ppo2_sto_rew.csv"

if not os.path.exists("csv"):
    os.makedirs("csv")
else:
    if os.path.exists(csv_obs_path):
        os.remove(csv_obs_path)
    if os.path.exists(csv_acs_path):
        os.remove(csv_acs_path)
    if os.path.exists(csv_rew_path):
        os.remove(csv_rew_path)

state, dones = initialize_placeholders(**alg_kwargs)

while True:
    actions, _, state, _ = model.step_deterministic(obs,S=state, M=dones)
    obs, reward, done, _  = env.step_runtime(actions)

    # actions, _, state, _ = model.step(obs,S=state, M=dones)
    # obs, _, done, _ = env.step(actions)

    # obs, reward, done, _  = env.step_wait_runtime(actions)

    # csv_file.write_obs(obs[0], csv_obs_path)
    # csv_file.write_acs(actions[0], csv_acs_path)
    # csv_file.write_rew(reward, csv_rew_path)
    #
    # if reward > 0.97:
    #     for i in range(10):
    #         env.step(obs[:6])
    #         time.sleep(0.2)
    #     break
