import numpy as np
import sys

import gym
import gym_gazebo_2

import tensorflow as tf

import copy
import time

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

from baselines.acktr.acktr_cont import learn
from baselines.agent.utility.general_utils import get_ee_points, get_position
from baselines.ppo1 import mlp_policy, pposgd_simple_collisions
import os
import write_csv as csv_file

env = gym.make('MARAOrientCollision-v0')
initial_observation = env.reset()
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

# tf.train.Saver().restore(sess, '/tmp/rosrl/GazeboMARATopOrientCollisionv0Env/ppo1/1000000_nsec/models/mara_orient_ppo1_test_afterIter_860.model') # 850, 1310
tf.train.Saver().restore(sess, '/home/yue/experiments/ppo1_orient_collisions/alex_point2/1000000_nsec_dist<0.01_orient<0.005_sameRot_punish2/models/mara_orient_ppo1_test_afterIter_980.model')

# loadPath = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo1/'
# tf.train.Saver().restore(sess, loadPath + 'ros1_ppo1_H_afterIter_263.model')
# tf.train.Saver().restore(sess, '/home/rkojcev/baselines_networks/ros1_ppo1_test_O/saved_models/ros1_ppo1_test_O_afterIter_421.model') # for the O
done = False

csv_obs_path = "csv/ppo_sto_obs.csv"
csv_acs_path = "csv/ppo_sto_acs.csv"
# csv_obs_path = "csv/ppo_det_obs.csv"
# csv_acs_path = "csv/ppo_det_acs.csv"

if os.path.exists(csv_obs_path):
    os.remove(csv_obs_path)
if os.path.exists(csv_acs_path):
    os.remove(csv_acs_path)

while True:
    # action = pi.act(False, obs)[0]
    action = pi.act(True, obs)[0]
    # obs, reward, done, info, collision = env.step(action, action)

    csv_file.write_obs(obs, csv_obs_path)
    csv_file.write_acs(action, csv_acs_path)

    obs, reward, done, info = env.step(action)

    # print(action)
