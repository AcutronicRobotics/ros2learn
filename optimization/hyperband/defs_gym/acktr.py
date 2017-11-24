"function (and parameter space) definitions for hyperband"
"binary classification with Keras (multilayer perceptron)"

from common_defs import *

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

import os
import time

space = {
    'gamma': hp.uniform( 'gamma', 0.1, 0.99 ),
    'lamda': hp.uniform( 'lam', 0.1, 0.99 ),
    'timesteps_per_batch': hp.quniform( 'timesteps_per_batch', 25, 25000, 1),
    'desired_kl': hp.uniform( 'desired_kl', 0.001, 0.01 ),
    'num_timesteps': hp.quniform( 'num_timesteps', 1, 2, 1 ),
    'max_pathlength': hp.quniform( 'max_pathlength', 50, 200,1 ),
    'stepsize': hp.quniform( 'stepsize', 0.01, 0.1, 0.01 )
}

def get_params():
    params = sample( space )
    return handle_integers(params)

def print_params( params ):
    pprint({ k: v for k, v in params.items() if not k.startswith( 'layer_' )})
    print
def init_enviroment():
    print("init acktr env")
    # global env
    # env = gym.make('GazeboModularScara4DOF-v3')
    # time.sleep(5)
    # init_obs = env.goToInit()
    # initial_observation = env.reset()
    # print("Initial observation: ", initial_observation)
    # env.render()
    #
    # seed=0
    # set_global_seeds(seed)
    # env.seed(seed)
    # ob_dim = env.observation_space.shape[0]
    # ac_dim = env.action_space.shape[0]

def try_params( n_iterations, params ):
    # print("itter: ", itter)
    print("iterations:", n_iterations)
    print_params( params )
    # os.system("killall -9 roslaunch roscore gzclient gzserver")
    env = gym.make('GazeboModularScara4DOF-v3')
    # time.sleep(5)
    # init_obs = env.goToInit()
    initial_observation = env.reset()
    # print("Initial observation: ", initial_observation)
    env.render()

    seed=0
    set_global_seeds(seed)
    env.seed(seed)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    with tf.Session(config=tf.ConfigProto()) as session:
        with tf.variable_scope("vf"):
            vf = NeuralNetValueFunction(ob_dim, ac_dim)
        with tf.variable_scope("pi"):
            policy = GaussianMlpPolicy(ob_dim, ac_dim)
        mean_reward = learn(env,
              policy=policy, vf=vf,
              gamma=params['gamma'],
              lam=params['lamda'],
              timesteps_per_batch=params['timesteps_per_batch'],
              desired_kl=params['desired_kl'],
              num_timesteps=params['num_timesteps'],
              animate=False,
              save_model_with_prefix='4dof_acktr_O_' + str(n_iterations),
              restore_model_from_file='')
    session.close()
    tf.reset_default_graph()

    if mean_reward > 0.0:
        mean_reward = mean_reward * (-1)
    else:
        mean_reward = abs(mean_reward)

    print("mean_reward: ", mean_reward)



    return { 'loss':mean_reward, 'loss':mean_reward}
