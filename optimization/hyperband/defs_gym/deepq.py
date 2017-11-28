"function (and parameter space) definitions for hyperband"
"binary classification with Keras (multilayer perceptron)"

from common_defs import *

import argparse
import time
import os
import logging
import gym
from gym import spaces
import gym_gazebo

import tensorflow as tf

import numpy as np
import pandas
from baselines import deepq
from  baselines.deepq import models
from  baselines.deepq import build_graph_robotics
from  baselines.deepq import replay_buffer
from  baselines.deepq.simple_robotics import learn, load

import os
import time

from hyperopt.pyll.base import scope

space = {
    'actor_lr': hp.uniform( 'actor_lr', 1e-05, 0.1 ),
    'critic_lr': hp.uniform( 'critic_lr', 1e-05, 0.1),
    'gamma': hp.uniform('gamma', 0.1, 1.0),
    'nb_epoch_cycles': scope.int(hp.uniform( 'nb_epoch_cycles', 1, 2)),
    'gamma': hp.uniform( 'gamma', 0.1, 0.99),
    'nb_train_steps': scope.int(hp.uniform( 'nb_train_steps', 20, 100)),
    'nb_eval_steps': scope.int(hp.uniform( 'nb_eval_steps', 50, 200)),
    'nb_rollout_steps': scope.int(hp.uniform( 'nb_rollout_steps', 50, 200)),
    'nb_epochs': scope.int(hp.uniform( 'nb_epochs', 200, 700))
}

def get_params():
    params = sample( space )
    return handle_integers(params)

def print_params( params ):
    pprint({ k: v for k, v in params.items() if not k.startswith( 'layer_' )})
    print
def init_enviroment():
    print("init env")
    # global env
    # global itter
    #
    # itter = 0
    # policy_to_run = None
    # print("init ppo1 env")
    # # policy_to_run  = policy_fn
    # env = gym.make('GazeboModularScara4DOF-v3')
    # global env
    # time.sleep(5)
    # initial_observation = env.reset()
    # print("Initial observation: ", initial_observation)
    # env.render()
    # seed = 0
    # set_global_seeds(seed)
    # env.seed(seed)


# def get_scope_variable(scope_name, var, shape=None):
#     with tf.variable_scope(scope_name) as scope:
#         try:
#             v = tf.get_variable(var, shape)
#         except ValueError:
#             scope.reuse_variables()
#             v = tf.get_variable(var)
#     return v
# def policy_fn(name, ob_space, ac_space):
#     return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,hid_size=64, num_hid_layers=2)

def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def try_params( n_iterations, params ):

    env = gym.make("GazeboModularScara3DOF-v2")
    tf.reset_default_graph()

    #Discrete actions
    goal_average_steps = 2
    max_number_of_steps = 20
    last_time_steps = np.ndarray(0)
    n_bins = 10
    epsilon_decay_rate = 0.99 ########
    it = 1 ######

    # tf.reset_default_graph()

    # Number of states is huge so in order to simplify the situation
    # typically, we discretize the space to: n_bins ** number_of_features
    joint1_bins = pandas.cut([-np.pi/2, np.pi/2], bins=n_bins, retbins=True)[1][1:-1]
    joint2_bins = pandas.cut([-np.pi/2, np.pi/2], bins=n_bins, retbins=True)[1][1:-1]
    joint3_bins = pandas.cut([-np.pi/2, np.pi/2], bins=n_bins, retbins=True)[1][1:-1]
    action_bins = pandas.cut([-np.pi/2, np.pi/2], bins=n_bins, retbins=True)[1][1:-1]

    difference_bins = abs(joint1_bins[0] - joint1_bins[1])
    action_bins = [(difference_bins, 0.0, 0.0), (-difference_bins, 0.0, 0.0),
            (0.0, difference_bins, 0.0), (0.0, -difference_bins, 0.0),
            (0.0, 0.0, difference_bins), (0.0, 0.0, -difference_bins),
            (0.0, 0.0, 0.0)]
    discrete_action_space = spaces.Discrete(7)
    model = models.mlp([64])
    # print("learning rate", learning_rate)
    # print("gam", gam)
    # print("max_timesteps", max_t)
    # print("buffer size", buff_size)
    # print("learning starts", lr_start)


    act, mean_rew = learn(
        env,
        q_func=model,
        lr=1e-05,
        gamma=0.8,
        max_timesteps=10,
        buffer_size=1000,
        checkpoint_freq = 100,
        learning_starts = 100,
        target_network_update_freq = 100,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback)

    # session.close()

    # env.close()


    # tf.reset_default_graph()
    #print("Saving model to cartpole_model.pkl")
    # act.save("scara_model_" + str(job_id) + ".pkl")

    print("MEAN REWARD", mean_rew)


    #1 - mean of simulation because Spearmin
    return { 'loss':mean_rew, 'loss':mean_rew}
