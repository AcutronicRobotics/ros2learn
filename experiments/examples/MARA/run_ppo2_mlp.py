import os
import sys
import time
import gym
import gym_gazebo2
import multiprocessing
import tensorflow as tf
import write_csv as csv_file

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
    env.set_reward_params({"alpha": defaults["_alpha"],
                            "beta": defaults["_beta"],
                            "gamma": defaults["_gamma"],
                            "delta": defaults["_delta"],
                            "eta": defaults["_eta"],
                            "done": defaults["done"]})
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env

# Get dictionary from baselines/ppo2/defaults
defaults = get_learn_function_defaults('ppo2', 'mara_mlp')

# Create needed folders
try:
    logdir = defaults['trained_path'].split('checkpoints')[0] + 'results' + defaults['trained_path'].split('checkpoints')[1]
except:
    logdir = '/tmp/ros_rl2/' + defaults['env_name'] + '/ppo2_mlp_results/'
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

alg_kwargs ={ 'num_layers': defaults['num_layers'], 'num_hidden': defaults['num_hidden'] }
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
while True:
    actions = model.step_deterministic(obs)[0]
    obs, reward, done, _  = env.step_runtime(actions)

    print("Reward: ", reward)
    print("ee_translation[x, y, z]: ", obs[0][6:9])
    print("ee_orientation[w, x, y, z]: ", obs[0][9:13])

    csv_file.write_obs(obs[0], csv_files[0], defaults['env_name'])
    csv_file.write_acs(actions[0], csv_files[1])
    csv_file.write_rew(reward, csv_files[2])

    # if reward > 0.99:
    #     for i in range(10):
    #         env.step(obs[:6])
    #         time.sleep(0.2)
    #     break
