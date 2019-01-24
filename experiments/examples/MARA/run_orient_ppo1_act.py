import numpy as np
import sys

import gym
import gym_gazebo_2

import tensorflow as tf

import argparse
import copy
import time

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

from baselines.acktr.acktr_cont import learn
from baselines.agent.utility.general_utils import get_ee_points, get_position
from baselines.ppo1 import mlp_policy, pposgd_simple_collisions

env = gym.make('MARAOrientCollision-v0')
initial_observation = env.reset()
# env.init_time(slowness=5, slowness_unit='sec', reset_jnts=True)
# env.render()
seed = 0

sess = U.make_session(num_cpu=1)
sess.__enter__()
def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    hid_size=128, num_hid_layers=4)
# gym.logger.setLevel(logging.WARN)
obs = env.reset()
# env.seed(seed)
# time.sleep(5)
pi = policy_fn('pi', env.observation_space, env.action_space)

# loadPath = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo1/'
# tf.train.Saver().restore(sess, loadPath + 'ros1_ppo1_H_afterIter_263.model')

# tf.train.Saver().restore(sess, '/home/yue/experiments/orient_collisions/alex_point2/1000000_nsec_dist<0.01_orient<0.005_sameRot_punish2/models/mara_orient_ppo1_test_afterIter_980.model')
# tf.train.Saver().restore(sess, '/home/yue/experiments/orient_collisions/alex_point2/1000000_nsec_128_3/models/mara_orient_ppo1_test_afterIter_820.model')
# tf.train.Saver().restore(sess, '/home/yue/experiments/orient_collisions/left_corner/1000000_nsec_128_4_2048/models/mara_orient_ppo1_test_afterIter_970.model') #930 970
# tf.train.Saver().restore(sess, '/home/yue/experiments/orient_collisions/left_corner/axis_angle/1000000_nsec_default/models/mara_orient_ppo1_test_afterIter_560.model')
tf.train.Saver().restore(sess, '/media/yue/801cfad1-b3e4-4e07-9420-cc0dd0e83458/orient_collisions/axilex2/1000000_nsec/models/mara_orient_ppo1_test_afterIter_3130.model')
done = False
collided = False
while True:
    action = pi.act(False, obs)[0]
    if not collided:
        obs, reward, done, collided, info = env.step(action, action)
    else:
        obs, reward, done, collided, info = env.step(action, action)
        print("COLLISION :(")
    # print(action)
