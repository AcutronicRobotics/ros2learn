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

import os


def make_env():
    env = gym.make('GazeboModularScara3DOF-v3')
    # env.init_time(slowness= args.slowness, slowness_unit=args.slowness_unit)
    # logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo2/' + str(args.slowness) + '_' + str(args.slowness_unit) + '/'
    # logger.configure(os.path.abspath(logdir))
    # print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))
    env = bench.MonitorRobotics(env, logger.get_dir(), allow_early_resets=True)
    env.render()
    return env


env = DummyVecEnv([make_env])
env = VecNormalize(env)

# env = gym.make('GazeboModularScara3DOF-v3')
# initial_observation = env.reset()
# print("Initial observation: ", initial_observation)
# env.render()
seed = 0

ncpu = 1
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)

sess = tf.Session(config=config)
sess.__enter__()
# env.render()
seed = 0
set_global_seeds(seed)



# def policy_fn(name, ob_space, ac_space):
#     return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
#     hid_size=64, num_hid_layers=2)
# gym.logger.setLevel(logging.WARN)
policy = MlpPolicy
nsteps=2048
nminibatches=32
ent_coef=0.0
vf_coef=0.5
max_grad_norm=0.5
nenvs = env.num_envs
ob_space = env.observation_space
ac_space = env.action_space
nbatch = nenvs * nsteps
nbatch_train = nbatch // nminibatches

make_model = lambda : ppo2.Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm)
model = make_model()
model.load('/home/rkojcev/baselines_networks/paper/data/GazeboModularScara3DOFv3Env_12222017/ppo2/checkpoints/00480')
