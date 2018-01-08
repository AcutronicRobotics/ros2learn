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

# parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slowness', help='time for executing trajectory', type=int, default=1)
parser.add_argument('--slowness-unit', help='slowness unit',type=str, default='sec')
args = parser.parse_args()

env = gym.make('GazeboModularScara4DOF-v3')
env.init_time(slowness= args.slowness, slowness_unit=args.slowness_unit)
logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/acktr/' + str(args.slowness) + '_' + str(args.slowness_unit) + '/'
# logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/acktr/'
logger.configure(os.path.abspath(logdir))
print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))

init_obs = env.goToInit()
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
        desired_kl=0.001,
        num_timesteps=4e6,
        animate=False,
        save_model_with_prefix='4dof_acktr_H',
        restore_model_from_file='', outdir=logger.get_dir())
