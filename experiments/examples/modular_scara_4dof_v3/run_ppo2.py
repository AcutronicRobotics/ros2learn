import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys
import numpy as np

from baselines import bench, logger

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import joblib

import multiprocessing

import os
import time

ncpu = multiprocessing.cpu_count()
if sys.platform == 'darwin': ncpu //= 2

print("ncpu: ", ncpu)
# ncpu = 1
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu,
                        log_device_placement=False)
config.gpu_options.allow_growth = True #pylint: disable=E1101

sess = tf.Session(config=config)

sess.__enter__()


def make_env():
    env = gym.make('GazeboModularScara4DOF-v3')
    env.init_time(slowness= 2, slowness_unit='sec', reset_jnts=False)
    # logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo2/' + str(args.slowness) + '_' + str(args.slowness_unit) + '/'
    # logger.configure(os.path.abspath(logdir))
    # print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))
    # env = bench.MonitorRobotics(env, logger.get_dir(), allow_early_resets=True)
    env.render()
    return env


env = DummyVecEnv([make_env])
env = VecNormalize(env)

nenvs = env.num_envs
ob_space = env.observation_space
ac_space = env.action_space
nsteps = 1 # default
nbatch = nenvs * nsteps
nminibatches=4

nbatch_train = nbatch // nminibatches
vf_coef=0.5
max_grad_norm=0.5
ent_coef=0.0

gamma=0.99
lam=0.95

dones = [False for _ in range(nenvs)]

policy = MlpPolicy
make_model = lambda : ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                max_grad_norm=max_grad_norm)

model = make_model()

model.load('/home/rkojcev/baselines_networks/paper/data/GazeboModularScara3DOFv3Env_diff_times/ppo2/100000000_nsec/checkpoints/00250')

runner = ppo2.Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)


# def policy_fn(sess, ob_space, ac_space, nbatch, nsteps):
#     return MlpPolicy(sess=sess, ob_space=ob_space, ac_space=ac_space,
#     nbatch=nbatch, nsteps=nsteps)
# model = policy_fn(sess, env.observation_space, env.action_space, nbatch, nsteps)
# tf.train.Saver().restore(sess, '/tmp/rosrl/GazeboModularScara3DOFv3Env/ppo2/1000000_nsec/checkpoints/00480') # for the H
# #
obs = np.zeros((nenvs,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
# obs = np.zeros((nenvs,) + env.observation_space.shape, dtype=model.X.dtype.name)
obs[:] = env.reset()
print("Initial obs: ", obs)
obs_extended = np.tile(obs, (nsteps,1))

# print(obs_extended.shape)
#
done = False
while True:
# for _ in range(nsteps):
    # runner.run()
    # obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
    # action = pi.step(True, obs)[0]
    # obs_extended = np.tile(obs, (nsteps,1))
    actions, values, states, neglogpacs = model.step(obs)
    # # action = pi.step(obs)
    #
    # print("action is: ", actions[0])
    obs, reward, done, info = env.step(actions)

# time.sleep(10)


    # obs = []
    # # obs = np.zeros((nenvs,) + env.observation_space.shape, dtype=pi.X.dtype.name)
    # # obs[:] = env.reset()
    # # print(action)
