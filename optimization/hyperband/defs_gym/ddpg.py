"function (and parameter space) definitions for hyperband"
"binary classification with Keras (multilayer perceptron)"

from common_defs import *

import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
import gym_gazebo
import tensorflow as tf
from mpi4py import MPI


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

def try_params( n_iterations, params ):

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    #Default parameters
    env_id = "GazeboModularScara3DOF-v2"
    noise_type='adaptive-param_0.2'
    layer_norm = True
    seed = 0
    render_eval = False
    render = False
    normalize_returns=False
    normalize_observations=True
    critic_l2_reg=1e-2
    batch_size=64
    popart=False
    reward_scale=1.
    clip_norm=None
    num_timesteps=None
    evaluation = True
    nb_epochs = 100

    # Create envs.
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    gym.logger.setLevel(logging.WARN)

    eval_env = gym.make(env_id)
    # Parse noise_type
    action_noise = None
    param_noise = None

    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)


    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    optim_metric = training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory,
        actor_lr = params['actor_lr'], critic_lr = params['critic_lr'], gamma = params['gamma'],
        nb_epoch_cycles = params['nb_epoch_cycles'], nb_train_steps = params['nb_train_steps'], nb_eval_steps = params['nb_eval_steps'], nb_rollout_steps = params['nb_rollout_steps'],
        nb_epochs= params['nb_epochs'], render_eval=render_eval, reward_scale=reward_scale, render=render,
        normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        critic_l2_reg=critic_l2_reg, batch_size = batch_size, popart=popart,
        clip_norm=clip_norm)

    env.close()
    # if eval_env is not None:
    #     eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

    # policy_to_run = None

    # return { 'loss':mean_reward, 'loss':mean_reward}
    return { 'loss':optim_metric, 'loss':optim_metric}
