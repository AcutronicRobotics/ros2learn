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
import baselines.common.tf_util as U

import gym
import gym_gazebo
import tensorflow as tf
from mpi4py import MPI

def train_setup(job_id, ac_lr, cr_lr, g, rew_sc):
#train_setup(job_id, params['actor_lr'], params['critic_lr'], params['no_epochs'], params['gamma'], params['no_cycles'], params['no_train_steps'], params['no_eval_steps'], params['no_rollout_steps'])
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    #Default parameters
    env_id = "GazeboModularScara3DOF-v2"
    noise_type='ou_0.2'
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
    nb_epochs = 150

    # Create envs.
    env = gym.make(env_id)
    env.reset()
    #env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
    #gym.logger.setLevel(logging.WARN)

    #eval_env = gym.make(env_id)
    eval_env = None

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

    # with tf.variable_scope("ddpg_" + str(job_id)):
    # with U.single_threaded_session() as session:
    with tf.Session(config=tf.ConfigProto()) as session:

        # Configure components.
        memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
        critic = Critic(layer_norm=layer_norm)
        actor = Actor(nb_actions, layer_norm=layer_norm)

        #no_rollout_i = list(map(int, no_rollout_steps))
        ack_lr = list(map(float, ac_lr))
        gam = list(map(float, g))
        reward_scale = list(map(float, rew_sc))

        # Seed everything to make things reproducible.
        seed = seed + 1000000 * rank
        logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
        # tf.reset_default_graph()
        set_global_seeds(seed)
        env.seed(seed)
        if eval_env is not None:
            eval_env.seed(seed)

            # tf.variable_scope("graph" + str(itteration)):
            # graph = tf.Graph()
            # graph.as_default()
        # Disable logging for rank != 0 to avoid noise.
        if rank == 0:
            start_time = time.time()
        optim_metric = training.train(env=env,
                                          eval_env=eval_env,
                                          session=session,
                                          param_noise=param_noise,
                                          action_noise=action_noise,
                                          actor=actor,
                                          critic=critic,
                                          memory=memory,
                                          actor_lr = ack_lr[0],
                                          critic_lr = cr_lr[0],
                                          gamma =gam[0],
                                          nb_epoch_cycles = 10,
                                          nb_train_steps = 5,
                                          #nb_rollout_steps = 100,
                                          nb_rollout_steps = 200,
                                          nb_epochs= 100,
                                          render_eval=render_eval,
                                          reward_scale=1,
                                          render=render,
                                          normalize_returns=normalize_returns,
                                          normalize_observations=normalize_observations,
                                          critic_l2_reg=critic_l2_reg,
                                          batch_size = batch_size,
                                          popart=popart,
                                          clip_norm=clip_norm,
                                          job_id=str(job_id))
            # env.close()
            # if eval_env is not None:
            #     eval_env.close()
        if rank == 0:
            logger.info('total runtime: {}s'.format(time.time() - start_time))

    logger.info(' Got optimization_metric ', optim_metric)
    return optim_metric # Hyperparameter optimization purposes
    # tf.Session.reset(target, ["ddpg_" + str(job_id)])

def main(job_id, params):

    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    # Run actual script.
    #return train_setup(job_id, params['critic_lr'], params['gamma'])
    return train_setup(job_id, params['actor_lr'], params['critic_lr'], params['gamma'], params['reward_scale'])
