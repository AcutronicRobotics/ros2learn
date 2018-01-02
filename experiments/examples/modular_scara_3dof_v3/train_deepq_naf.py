import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.deepqnaf
from baselines.deepqnaf import experiment
import baselines.deepqnaf.naf as naf
from baselines.deepqnaf.experiment import recursive_experiment

import baselines.common.tf_util as U

import gym
import gym_gazebo
import tensorflow as tf
from mpi4py import MPI


#Default parameters
graph = True
render = True
env_id = "GazeboModularScara3DOF-v3"
noise_type='ou_0.2'
repeats = 1
episodes = 200
max_episode_steps = 1000
train_steps = 5
learning_rate = 0.01
batch_normalize = True
gamma = 0.99
tau = 0.99
epsilon = 0.1
hidden_size = [16, 32, 200]
hidden_n = 2
hidden_activation = tf.nn.relu
batch_size = 128
memory_capacity = 10000
load_path = None
output_path = '/tmp/rosrl/' + str(env_id.__class__.__name__) +'/deepq_naf/'
covariance = "original"
solve_threshold = None
v = 0

# Create envs.
env = gym.make(env_id)
env.reset()

logdir = '/tmp/rosrl/' + str(env_id.__class__.__name__) +'/deepq_naf/monitor/'
logger.configure(os.path.abspath(logdir))
print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))
env = bench.MonitorRobotics(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True) #, allow_early_resets=True
gym.logger.setLevel(logging.WARN)

#construct experiment inputs
keys=['v', 'graph','render','environment','repeats','episodes','max_episode_steps','train_steps','batch_normalize', 'learning_rate','gamma','tau','epsilon','hidden_size','hidden_n','hidden_activation','batch_size', 'memory_capacity', 'load_path', 'covariance', 'solve_threshold']
vals=[v, graph, render, env_id, repeats, episodes, max_episode_steps, train_steps, batch_normalize, learning_rate, gamma, tau, epsilon, hidden_size, hidden_n,hidden_activation, batch_size, memory_capacity, load_path, covariance, solve_threshold]

#run experiments
rewards = recursive_experiment(keys, vals, [])

#get legend
l_keys = []
l_vals = []
for i,v in enumerate(vals):
  if type(v) == list and len(v) > 1:
    l_keys += [keys[i]]
    l_vals += [v]
legend = recursive_legend(l_keys, l_vals, [])

#save results
if output_path is not None:
  result_file = open(output_path, 'w')
  result_file.write('\n')
  result_file.write(str(legend) + '\n')
  result_file.write(str(rewards))
  result_file.close()

#plot
if args.graph: #TODO separate plots for different envs - subplots or completely separate?
  for r in np.mean(rewards, -2): #mean of repeats
    plt.plot(r,'o')
  plt.legend(legend)
  plt.show()
