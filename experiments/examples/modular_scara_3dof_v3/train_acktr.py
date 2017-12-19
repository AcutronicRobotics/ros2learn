import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys

# Use algorithms from baselines
from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
from baselines.common import set_global_seeds

from baselines import bench, logger
import os

env = gym.make('GazeboModularScara3DOF-v3')
logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/acktr/monitor/'
logger.configure(os.path.abspath(logdir))
print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))
env = bench.MonitorRobotics(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True) #, allow_early_resets=True

# set_global_seeds(seed)
# env.seed(seed)

initial_observation = env.reset()
print("Initial observation: ", initial_observation)
env.render()

seed=0
set_global_seeds(seed)
env.seed(seed)

with tf.Session(config=tf.ConfigProto()) as session:
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    with tf.variable_scope("vf"):
        vf = NeuralNetValueFunction(ob_dim, ac_dim)
    with tf.variable_scope("pi"):
        policy = GaussianMlpPolicy(ob_dim, ac_dim)

    learn(env,
        policy=policy, vf=vf,
        gamma=0.99,
        lam=0.97,
        timesteps_per_batch=2500,
        desired_kl=0.002,
        num_timesteps=1e6,
        animate=False,
        save_model_with_prefix='3dof_acktr_O',
        restore_model_from_file='')
