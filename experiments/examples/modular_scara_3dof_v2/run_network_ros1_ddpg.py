import numpy as np
import sys

import gym
import gym_gazebo

import tensorflow as tf

import argparse
import copy
import time

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from collections import deque
from baselines.acktr.acktr_cont import learn
from baselines.agent.utility.general_utils import get_ee_points, get_position

from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory

env = gym.make('GazeboModularScara3DOF-v2')

tf.reset_default_graph()
env.render()
seed = 0

sess = U.make_session(num_cpu=1)
sess.__enter__()


max_action = env.action_space.high
nb_epochs= 200
nb_epoch_cycles = 10
nb_eval_steps=100
layer_norm = True
nb_actions = env.action_space.shape[-1]


# Configure components.
memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
critic = Critic(layer_norm=layer_norm)
actor = Actor(nb_actions, layer_norm=layer_norm)


agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape)

#loadPath = '/tmp/rosrl/GazeboModularScara3DOFv2Env/ddpg/'
#tf.train.Saver().restore(sess, '/tmp/rosrl/GazeboModularScara3DOFv2Env/ddpg/0epoch_90episode_90.model')

graph = tf.get_default_graph()#################
print(tf.global_variables)

agent.initialize(sess)

logger.info('Using agent with the following configuration:')
logger.info(str(agent.__dict__.items()))


step = 0
episode = 0
eval_episode_rewards_history = deque(maxlen=100)
episode_rewards_history = deque(maxlen=100)

# session.graph.finalize()
agent.reset()
obs = env.reset()

done = False
episode_reward = 0.
episode_step = 0
episodes = 0
t = 0

sim_r = 0
sim_t = 0
done_quant = 0
epoch = 0
start_time = time.time()

epoch_episode_rewards = []
epoch_episode_steps = []
epoch_episode_eval_rewards = []
epoch_episode_eval_steps = []
epoch_start_time = time.time()
epoch_actions = []
epoch_qs = []
epoch_episodes = 0
for epoch in range(nb_epochs):
    logger.info('epoch %i:',epoch)
    for cycle in range(nb_epoch_cycles):
        episode_rewards = []
        #qs = []
        if env is not None:
            episode_reward = 0.
            for t_rollout in range(nb_eval_steps):
                action, q = agent.pi(obs, apply_noise=False, compute_Q= False)
                print("action", action)
                print("max_action * action", max_action * action)
                obs,r, done,info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                env.render()
                episode_reward += r

                #qs.append(q)
                if done:
                    obs = env.reset()
                    episode_rewards.append(episode_reward)
                    episode_rewards_history.append(episode_reward)
                    episode_reward = 0.
                    print("\n\n ****************************** ENVIRONMENT SOLVED ****************\n \b")
                    break
