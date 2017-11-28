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

def train_setup(job_id, act_lr, cr_lr, no_epochs, gam, no_cycles, no_train_steps, no_eval_steps, no_rollout_steps):
#train_setup(job_id, params['actor_lr'], params['critic_lr'], params['no_epochs'], params['gamma'], params['no_cycles'], params['no_train_steps'], params['no_eval_steps'], params['no_rollout_steps'])
    # Configure things.
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
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, actor_lr = float(act_lr), critic_lr = float(cr_lr), gamma = float(gam), nb_epoch_cycles = int(no_cycles), nb_train_steps = int(no_train_steps), nb_eval_steps = int(no_eval_steps), nb_rollout_steps = int(no_rollout_steps),
        nb_epochs= int(nb_epochs), render_eval=render_eval, reward_scale=reward_scale, render=render, normalize_returns=normalize_returns, normalize_observations=normalize_observations, critic_l2_reg=critic_l2_reg, batch_size = batch_size, popart=popart,
        clip_norm=clip_norm)
    env.close()
    # if eval_env is not None:
    #     eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

    logger.info(' Got optimization_metric %d', optim_metric)
    return optim_metric # Hyperparameter optimization purposes



def main(job_id, params):

    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    return train_setup(job_id, params['actor_lr'], params['critic_lr'], params['no_epochs'], params['gamma'], params['no_cycles'], params['no_train_steps'], params['no_eval_steps'], params['no_rollout_steps'])
