import gym
import gym_gazebo
import logging
import numpy as np
import tensorflow as tf
import os
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from baselines.deepqnaf.naf import NAF
from baselines.deepqnaf.network import Network
from baselines.deepqnaf.statistic import Statistic
from baselines.deepqnaf.exploration import OUExploration, BrownianExploration, LinearDecayExploration
from baselines.deepqnaf.learn import learn
from baselines.deepqnaf.utils import get_model_dir, preprocess_conf

#Default parameters:

# env = 'GazeboModularScara3DOF-v3'
# #network
# hidden_dims = [100, 100]
# batch_norm = True
# clip_action= False
# use_seperate_networks = False
# hidden_w= 'uniform_big' # [uniform_small, uniform_big, he]')
# hidden_fn = 'tanh' # [none, tanh, relu]
# action_w =  'uniform_big' # [uniform_small, uniform_big, he]')
# action_fn = 'tanh' # [none, tanh, relu]')
# w_reg = None #[none, l1, l2]')
# w_reg_scale = 0.001 #scale of regularization')
# #exploration
# noise_scale = 0.3
# noise ='ou' # [ou, linear_decay, brownian]')
# tau = 0.001
# discount = 0.99
# learning_rate = 1e-3
# batch_size = 100
# max_steps = 200
# update_repeat = 5
# max_episodes = 1000

# logger = logging.getLogger()
# logger.propagate = False
# logger.setLevel(conf.log_level)
#
# logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/deepq_naf/monitor/'
# logger.configure(os.path.abspath(logdir))
# env = bench.MonitorRobotics(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True) #, allow_early_resets=True
# gym.logger.setLevel(logging.WARN)
flags = tf.app.flags
# environment
flags.DEFINE_string('env_name', 'GazeboModularScara4DOF-v3', 'name of environment')

# network
flags.DEFINE_string('hidden_dims', '[64, 64]', 'dimension of hidden layers')
flags.DEFINE_boolean('use_batch_norm', False, 'use batch normalization or not')
flags.DEFINE_boolean('clip_action', False, 'whether to clip an action with given bound')
flags.DEFINE_boolean('use_seperate_networks', False, 'use seperate networks for mu, V and A')
flags.DEFINE_string('hidden_w', 'uniform_big', 'weight initialization of hidden layers [uniform_small, uniform_big, he]')
flags.DEFINE_string('hidden_fn', 'tanh', 'activation function of hidden layer [none, tanh, relu]')
flags.DEFINE_string('action_w', 'uniform_big', 'weight initilization of action layer [uniform_small, uniform_big, he]')
flags.DEFINE_string('action_fn', 'tanh', 'activation function of action layer [none, tanh, relu]')
flags.DEFINE_string('w_reg', 'none', 'weight regularization [none, l1, l2]')
flags.DEFINE_float('w_reg_scale', 0.001, 'scale of regularization')

# exploration
flags.DEFINE_float('noise_scale', 0.2, 'scale of noise')
flags.DEFINE_string('noise', 'ou', 'type of noise exploration [ou, linear_decay, brownian]')

# training
flags.DEFINE_float('tau', 0.001, 'tau of soft target update')
flags.DEFINE_float('discount', 0.99, 'discount factor of Q-learning')
flags.DEFINE_float('learning_rate', 1e-3, 'value of learning rate')
flags.DEFINE_integer('batch_size', 100, 'The size of batch for minibatch training')
flags.DEFINE_integer('max_steps', 200, 'maximum # of steps for each episode')
flags.DEFINE_integer('update_repeat', 5, 'maximum # of q-learning updates for each step')
flags.DEFINE_integer('max_episodes', 1000, 'maximum # of episodes to train')

# Debug
flags.DEFINE_boolean('is_train', True, 'training or testing')
flags.DEFINE_integer('random_seed', 123, 'random seed')
flags.DEFINE_boolean('monitor', False, 'monitor the training or not')
flags.DEFINE_boolean('display', False, 'display the game screen or not')
flags.DEFINE_string('log_level', 'INFO', 'log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]')

conf = flags.FLAGS
#    ['is_train', 'random_seed', 'monitor', 'display', 'log_level'])
preprocess_conf(conf)
env = 'GazeboModularScara4DOF-v3'
# learn (env,
#             conf.noise,
#             conf.noise_scale,
#             conf.hidden_dims,
#             conf.use_batch_norm,
#             conf.use_seperate_networks,
#             conf.hidden_w,
#             conf.action_w,
#             conf.hidden_fn,
#             conf.action_fn,
#             conf.w_reg,
#             conf.clip_action,
#             conf.tau,
#             conf.discount,
#             conf.learning_rate,
#             conf.batch_size,
#             conf.max_steps,
#             conf.update_repeat,
#             conf.max_episodes)


# set random seed
tf.set_random_seed(123)
np.random.seed(123)


with tf.Session() as sess:
    # environment
    env = gym.make(env)
    env._seed(123)
    learn (env,
            sess)

if __name__ == '__main__':
  tf.app.run()
