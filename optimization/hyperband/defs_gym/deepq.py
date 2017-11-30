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
    'lr': hp.uniform( 'lr', 5e-05, 1e-02),
    'gamma': hp.uniform('gamma', 0.2, 0.8),
    'max_timesteps': scope.int(hp.uniform( 'max_timesteps', 500, 1000)),
    'buffer_size': scope.int(hp.uniform( 'buffer_size', 5000, 10000)),
    'learning_starts': scope.int(hp.uniform( 'learning_starts', 100, 1000))
}

def get_params():
    params = sample( space )
    return handle_integers(params)

def print_params( params ):
    pprint({ k: v for k, v in params.items() if not k.startswith( 'layer_' )})
    print
def init_enviroment():
    print("init env")


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
    with tf.Session(config=tf.ConfigProto()) as session:
        model = models.mlp([64])
        act, mean_rew = learn(
            env,
            q_func=model,
            lr=params['lr'],
            gamma=params['gamma'],
            max_timesteps=params['max_timesteps'],
            buffer_size=params['buffer_size'],
            checkpoint_freq = 100,
            learning_starts = params['learning_starts'],
            target_network_update_freq = 100,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            print_freq=10,
            callback=callback)
        print("MEAN REWARD", mean_rew)
        act.save("scara_model_" + str(n_iterations) + ".pkl")

    session.close()
    tf.reset_default_graph()


    #1 - mean of simulation because Spearmin
    return { 'loss':mean_rew, 'loss':mean_rew}
