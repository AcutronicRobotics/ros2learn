import os
import sys
import time
import gym
import gym_gazebo2
import numpy as np
import multiprocessing
import tensorflow as tf
import write_csv as csv_file
import numpy as np

from importlib import import_module
from baselines import bench, logger
from baselines.ppo2 import model as ppo2
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

tf.Session(config=config).__enter__()

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module

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
    env = gym.make(defaults['env_name'])
    env.set_episode_size(defaults['nsteps'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env

def initialize_placeholders(nlstm=256,**kwargs):
    return np.zeros((defaults['num_envs'] or 1, 2*nlstm)), np.zeros((1))

# Get dictionary from baselines/ppo2/defaults
defaults = get_learn_function_defaults('ppo2', 'mara_lstm')
# Tricks to run it faster and if it has been trained with many environment instances to launch only one
defaults['nsteps'] = 1
defaults['nminibatches'] = 1
defaults['num_envs'] = 1

# Create needed folders
try:
    logdir = defaults['trained_path'].split('checkpoints')[0] + 'results' + defaults['trained_path'].split('checkpoints')[1]
except:
    logdir = '/tmp/ros2learn/' + defaults['env_name'] + '/ppo2_lstm_results/'
finally:
    logger.configure( os.path.abspath(logdir) )
    csvdir = logdir + "/csv/"

csv_files = [csvdir + "det_obs.csv", csvdir + "det_acs.csv", csvdir + "det_rew.csv" ]
if not os.path.exists(csvdir):
    os.makedirs(csvdir)
else:
    for f in csv_files:
        if os.path.isfile(f):
            os.remove(f)

env = DummyVecEnv([make_env])

set_global_seeds(defaults['seed'])

if isinstance(defaults['lr'], float):
    defaults['lr'] = constfn(defaults['lr'])
else:
    assert callable(defaults['lr'])
if isinstance(defaults['cliprange'], float):
    defaults['cliprange'] = constfn(defaults['cliprange'])
else:
    assert callable(defaults['cliprange'])

alg_kwargs ={ 'nlstm': defaults['nlstm'], 'layer_norm': defaults['layer_norm'] }
policy = build_policy(env, defaults['network'], **alg_kwargs)

nenvs = env.num_envs
ob_space = env.observation_space
ac_space = env.action_space
nbatch = nenvs * defaults['nsteps']
nbatch_train = nbatch // defaults['nminibatches']

make_model = lambda : ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                                nbatch_train=nbatch_train,
                                nsteps=defaults['nsteps'], ent_coef=defaults['ent_coef'], vf_coef=defaults['vf_coef'],
                                max_grad_norm=defaults['max_grad_norm'])

model = make_model()

if defaults['trained_path'] is not None:
    model.load(defaults['trained_path'])

obs = env.reset()
state, dones = initialize_placeholders(**alg_kwargs)
loop = True
while loop:
    actions, _, state, _ = model.step_deterministic(obs,S=state, M=dones)
    obs, reward, done, _  = env.step_runtime(actions)

    print("Reward: ", reward)
    print("ee_translation[x, y, z]: ", obs[0][6:9])
    print("ee_orientation[w, x, y, z]: ", obs[0][9:13])

    # csv_file.write_obs(obs[0], csv_files[0], defaults['env_name'])
    # csv_file.write_acs(actions[0], csv_files[1])
    # csv_file.write_rew(reward, csv_files[2])

    if np.allclose(obs[0][6:9], np.asarray([0., 0., 0.]), atol=0.005 ): # lock if less than 5mm error in each axis
        env.step_runtime(obs[0][:6])
        loop = False

env.dummy().gg2().close()
os.kill(os.getpid(), 9)
