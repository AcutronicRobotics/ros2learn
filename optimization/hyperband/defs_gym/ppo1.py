"function (and parameter space) definitions for hyperband"
"binary classification with Keras (multilayer perceptron)"

from common_defs import *

import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys
import numpy as np

from hyperopt.pyll.base import scope

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple

import os
import time

space = {
    'max_timesteps': hp.quniform( 'max_timesteps', 1, 2, 1 ),
    'timesteps_per_actorbatch': hp.choice( 'timesteps_per_actorbatch', ( 1024, 2048, 4096)),
    'optim_epochs': scope.int(hp.uniform('optim_epochs', 10, 100)),
    'optim_stepsize': hp.uniform( 'optim_stepsize', 3e-2, 3e-6),
    'gamma': hp.uniform( 'gamma', 0.1, 0.99),
    'optim_batchsize': hp.choice( 'timesteps_per_batch', ( 32, 64, 128)),
    'lam': hp.uniform( 'lam', 0.1, 0.99),
}

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

    itter = 0
    policy_to_run = None
    print("init ppo1 env")
    # policy_to_run  = policy_fn
    env = gym.make('GazeboModularScara4DOF-v3')
    global env
    time.sleep(5)
    initial_observation = env.reset()
    print("Initial observation: ", initial_observation)
    env.render()
    seed = 0
    # session = U.make_session(num_cpu=1)
    # session.__enter__()
    set_global_seeds(seed)
    env.seed(seed)

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
    # os.system("killall -9 roslaunch roscore gzclient gzserver")
    # env = gym.make('GazeboModularScara4DOF-v3')
    # time.sleep(5)
    # initial_observation = env.reset()
    # print("Initial observation: ", initial_observation)
    # env.render()
    # seed = 0
    # # print("itter: ", itter)
    # print("iterations:", n_iterations)
    # print_params( params )
    # # U.reset()
    #
    # # session = U.make_session(num_cpu=1)
    # # session.__enter__()
    # set_global_seeds(seed)
    #
    # env.seed(seed)
    # tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)


    # with session.as_default():
    # with tf.Graph().as_default() as g:
    # g = tf.Graph()
    # with g.as_default():

    #     def policy_fn(name, ob_space, ac_space):
    #         return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,hid_size=64, num_hid_layers=2)
    # with tf.Session(config=tf.ConfigProto()) as session:
         # policy_to_run  = policy_fn
    # if policy_to_run is None:
    #     print("policy_to_run is None")
    #     policy_to_run = policy_fn
    g1 = tf.get_default_graph()
    print("Before session: ",g1.get_operations())
    with U.make_session(num_cpu=1) as session:
        print("In session: ",g1.get_operations())
    # with tf.Session( graph = g ) as session:
        print("I am in session")
        ob_space = env.observation_space
        ac_space = env.action_space
        def policy_fn(name, ob_space, ac_space):
            return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,hid_size=64, num_hid_layers=2)
        if itter is 0:
            policy_to_run = policy_fn
        # with tf.variable_scope("pi"):
        #     pi = policy_fn#("pi", ob_space, ac_space)

    #     # session.__enter__()
    #     # session.as_default()
    #     # policy_to_run  = policy_fn
        mean_reward = pposgd_simple.learn(env,
                                policy_to_run,
                                max_timesteps=params['max_timesteps'],
                                timesteps_per_actorbatch=params['timesteps_per_actorbatch'],
                                clip_param=0.2, entcoeff=0.0,
                                optim_epochs=params['optim_epochs'], optim_stepsize=params['optim_stepsize'], gamma=params['gamma'],
                                optim_batchsize=params['optim_batchsize'], lam=params['lam'], schedule='linear', save_model_with_prefix='4dof_ppo1_test_H' + str(n_iterations))
        g1 = tf.get_default_graph()
        print("after leanrn: ",g1.get_operations())
        policy_to_run  = None
        assert tf.get_default_session() is session
        assert session.graph is tf.get_default_graph()
    session.close()
    tf.reset_default_graph()
    itter+=1
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


    # print("mean_reward: ", mean_reward)
    #
    # if mean_reward > 0.0:
    #     mean_reward = mean_reward * (-1)
    # else:
    #     mean_reward = abs(mean_reward)
    #
    # print("mean_reward: ", mean_reward)

    # policy_to_run = None

    # return { 'loss':mean_reward, 'loss':mean_reward}
    return { 'loss':0.01, 'loss':0.001}
