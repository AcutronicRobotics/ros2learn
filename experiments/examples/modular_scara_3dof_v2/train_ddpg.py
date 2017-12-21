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


#Default parameters
#env_id = "HalfCheetah-v1"
env_id = "GazeboModularScara3DOF-v2"
noise_type='ou_0.2'
#noise_type='ou_1'
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

logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ddpg/monitor/'
logger.configure(os.path.abspath(logdir))
print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))
env = bench.MonitorRobotics(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True) #, allow_early_resets=True
gym.logger.setLevel(logging.WARN)

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



    # Seed everything to make things reproducible.
    seed = 0

    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

        # tf.variable_scope("graph" + str(itteration)):
        # graph = tf.Graph()
        # graph.as_default()
    # Disable logging for rank != 0 to avoid noise.



    optim_metric = training.train(env=env,
                                      eval_env=eval_env,
                                      session=session,
                                      param_noise=param_noise,
                                      action_noise=action_noise,
                                      actor=actor,
                                      critic=critic,
                                      memory=memory,
                                      actor_lr = 1e-04,
                                      critic_lr = 1e-03,
                                      gamma =0.99,
                                      nb_epoch_cycles = 20,
                                      nb_train_steps = 50,
                                      #nb_rollout_steps = 100,
                                      nb_rollout_steps = 100,
                                      nb_epochs= 500,
                                      render_eval=render_eval,
                                      reward_scale=1,
                                      render=render,
                                      normalize_returns=normalize_returns,
                                      normalize_observations=normalize_observations,
                                      critic_l2_reg=critic_l2_reg,
                                      batch_size = batch_size,
                                      popart=popart,
                                      clip_norm=clip_norm,
                                      job_id=str(0))
        # env.close()
        # if eval_env is not None:
        #     eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

logger.info(' Got optimization_metric ', optim_metric)
