import os
import sys
import gym
import gym_gazebo2
import numpy as np
import tensorflow as tf
import write_csv as csv_file

from baselines import bench, logger
from baselines.acktr import acktr
from baselines.acktr import defaults
from baselines.common import tf_util as U
from baselines.common.models import mlp
from baselines.common.policies import build_policy
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

U.get_session( config=tf.ConfigProto(
    allow_soft_placement = True,
    inter_op_parallelism_threads = 1,
    intra_op_parallelism_threads = 1) )

U.initialize()

# Get dictionary from baselines/acktr/defaults
defaults = defaults.mara_mlp()

# Create needed folders
try:
    logdir = defaults['trained_path'].split('checkpoints')[0] + 'results' + defaults['trained_path'].split('checkpoints')[1]
except:
    logdir = '/tmp/ros2learn/' + defaults['env_name'] + '/acktr_results/'
finally:
    logger.configure( os.path.abspath(logdir) )
    csvdir = logdir + "csv/"

csv_files = [csvdir + "det_obs.csv", csvdir + "det_acs.csv", csvdir + "det_rew.csv"]
if not os.path.exists(csvdir):
    os.makedirs(csvdir)
else:
    for f in csv_files:
        if os.path.isfile(f):
            os.remove(f)

def make_env():
    env = gym.make(defaults['env_name'])
    env.set_episode_size(defaults['nsteps'])
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)

    return env

env = DummyVecEnv([make_env])

network = mlp(num_layers=defaults['num_layers'], num_hidden=defaults['num_hidden'], layer_norm=defaults['layer_norm'])
policy = build_policy(env, network, **defaults)

make_model = lambda : acktr.Model(policy, env.observation_space, env.action_space, env.num_envs,
                            defaults['total_timesteps'], defaults['nprocs'], defaults['nsteps'], defaults['ent_coef'],
                            defaults['vf_coef'], defaults['vf_fisher_coef'], defaults['lr'], defaults['max_grad_norm'],
                            defaults['kfac_clip'], defaults['lrschedule'], defaults['is_async'])
model = make_model()

if defaults['trained_path'] is not None:
    model.load(defaults['trained_path'])

obs = env.reset()
loop = True
while loop:
    actions = model.step_deterministic(obs)[0]
    obs, reward, done, _  = env.step_runtime(actions)

    print("Reward: ", reward)
    print("ee_translation[x, y, z]: ", obs[0][6:9])
    print("ee_orientation[w, x, y, z]: ", obs[0][9:13])

    csv_file.write_obs(obs[0], csv_files[0], defaults['env_name'])
    csv_file.write_acs(actions[0], csv_files[1])
    csv_file.write_rew(reward, csv_files[2])

    if np.allclose(obs[0][6:9], np.asarray([0., 0., 0.]), atol=0.005 ): # lock if less than 5mm error in each axis
        env.step_runtime(obs[0][:6])
        loop = False

env.dummy().gg2().close()
os.kill(os.getpid(), 9)
