"function (and parameter space) definitions for hyperband"
"binary classification with Keras (multilayer perceptron)"

from common_defs import *

from hyperopt.pyll.base import scope

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

import os
import time

space = {
    'nsteps': hp.choice( 'nsteps', ( 1024, 2048, 4096)),
    'nminibatches': hp.choice( 'nminibatches', ( 8, 16, 32, 64, 128)),
    'lam': hp.uniform( 'lam', 0.1, 0.99),
    'gamma': hp.uniform( 'gamma', 0.1, 0.99),
    'noptepochs': scope.int(hp.uniform('optim_epochs', 10, 100)),
    'lr': hp.uniform( 'lr', 3e-2, 3e-4 ),
    'cliprange': hp.uniform( 'cliprange', 0.2, 0.4),
    'total_timesteps': scope.int(hp.quniform( 'total_timesteps', 1, 2, 1 ))
}


def make_env():
    env = gym.make('GazeboModularScara4DOF-v3')
    env.render()
    print(logger.get_dir())
    env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
    return env

def get_params():
    params = sample( space )
    return handle_integers(params)

def print_params( params ):
    pprint({ k: v for k, v in params.items() if not k.startswith( 'layer_' )})
    print
def init_enviroment():
    global env
    global policy_to_run
    global itter

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    seed = 0
    set_global_seeds(seed)

    # env = gym.make('GazeboModularScara4DOF-v3')
    # time.sleep(5)
    # initial_observation = env.reset()
    # print("Initial observation: ", initial_observation)
    # env.render()

def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v
def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,hid_size=64, num_hid_layers=2)

def try_params( n_iterations, params ):
    global policy_to_run
    print("iterations:", n_iterations)
    print_params( params )
    global itter

    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)

    with tf.Session(config=config) as session:
        policy = MlpPolicy
        mean_reward = ppo2.learn (policy=policy, env=env, nsteps=params['nsteps'], nminibatches=params['nminibatches'],
                    lam=params['lam'], gamma=params['gamma'], noptepochs=params['noptepochs'], log_interval=1,
                    ent_coef=0.0,
                    lr=params['lr'],
                    cliprange=params['cliprange'],
                    total_timesteps=params['total_timesteps'], save_interval=1)
        # g1 = tf.get_default_graph()
        # print("after leanrn: ",g1.get_operations())
        # policy_to_run  = None
        assert tf.get_default_session() is session
        assert session.graph is tf.get_default_graph()
    session.close()
    tf.reset_default_graph()
    # itter+=1
        # assert tf.get_default_graph() is session.graph()
        # session.close()
        # tf.reset_default_graph()
    # # assert tf.get_default_session() is session
    # print("Varibale scope is: ", tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi'))



         # tf.Session.reset(target, ["experiment0"])
         # tf.Session.reset(target='', containers=None, config=None)

         # print("Graph is still the same: ", tf.get_default_graph())
    # print(tf.get_default_graph())
    # tf.reset_default_graph()
    # policy_fn = None


    print("mean_reward: ", mean_reward)

    if mean_reward > 0.0:
        mean_reward = mean_reward * (-1)
    else:
        mean_reward = abs(mean_reward)

    print("mean_reward: ", mean_reward)

    # policy_to_run = None

    return { 'loss':mean_reward, 'loss':mean_reward}
    # return { 'loss':0.01, 'loss':0.001}
