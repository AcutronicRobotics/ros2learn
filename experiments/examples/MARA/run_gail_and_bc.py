from baselines.common import dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
import os
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from baselines.gail.statistics import stats
import gym
import gym_gazebo_2
from baselines.common import set_global_seeds, tf_util as U
from baselines.gail import mlp_policy
from baselines.gail.adversary import TransitionClassifier

env = gym.make('MARATop3DOF-v0')
# env.init_time(slowness=1, slowness_unit='sec', reset_jnts=True)
env.init_time(slowness=1000000, slowness_unit='nsec', reset_jnts=True)
# initial_observation = env.reset()
seed = 0

sess = U.make_session(num_cpu=1)
sess.__enter__()

def policy_fn(name, ob_space, ac_space, reuse=False):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                reuse=reuse, hid_size=100, num_hid_layers=2)

obs = env.reset()

ob_space = env.observation_space
ac_space = env.action_space
pretrained_weight=None
pi = policy_fn("pi", ob_space, ac_space, reuse=(pretrained_weight != None))

path = "/home/yue/experiments/checkpoint/mara/bc/new/10_100_repeated2"
tf.train.Saver().restore(sess,path)
done = False
time.sleep(2)

while True:
    action = pi.act(False, obs)[0]
    # obs, reward, done, info = env.step(action)
    obs, reward, done, ee_points_diff, info = env.step(action)
    # print(ee_points_diff)
    print( np.linalg.norm(ee_points_diff) )
