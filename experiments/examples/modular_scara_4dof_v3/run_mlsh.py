import argparse
import tensorflow as tf

# python main.py --task MovementBandits-v0 --num_subs 2 --macro_duration 10 --num_rollouts 1000 --warmup_time 60 --train_time 1 --replay True test

from mpi4py import MPI
from rl_algs.common import set_global_seeds, tf_util as U
import os.path as osp
import gym, logging
import numpy as np
from collections import deque
from gym import spaces
import mlsh_code.misc_util
import sys
import shutil
import subprocess
import mlsh_code.master_robotics as master_robotics
from mlsh_code.policy_network import Policy
from mlsh_code.subpolicy_network import SubPolicy
import gym_gazebo
import time

env = gym.make('GazeboModularScara4DOF-v3')
initial_observation = env.reset()
num_subs = 2
ob_space = env.observation_space
ac_space = env.action_space



env.render()
seed = 0
tf.reset_default_graph()
sess = U.make_session(num_cpu=1)

sess.__enter__()

basePath = '/tmp/mlsh/' + str(env.__class__.__name__) +'/test/'
path ="/home/erle/networks/savedir/checkpoints/00070.meta"
imported_meta = tf.train.import_meta_graph(path)
imported_meta.restore(sess, '/home/erle/networks/savedir/checkpoints/00070')
#imported_meta.restore(sess, tf.train.latest_checkpoint('/home/erle/networks/savedir/checkpoints/'))
graph = tf.get_default_graph()#################
print(tf.global_variables)
ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0]])
print("ob", ob)
cur_ep_ret = 0
cur_ep_len = 0
pi = Policy(name="policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)
print("Policy", pi)
sub_policies = [SubPolicy(name="sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]
print("Subpolicies", sub_policies)

init_op=tf.global_variables_initializer()
sess.run(init_op)
obs = env.goToInit()
#obs = env.reset()

init_obs = obs

summary_writer = tf.summary.FileWriter(basePath, graph=tf.get_default_graph())

done = False
while True:
    #print("obs.shape", obs.shape)
    #obs = obs.reshape(1,10)
    #feed_dict = {ob:obs}
    #pol=sess.run(pi,feed_dict=feed_dict)
    #obs=sess.run([pi],feed_dict=feed_dict)


    #cur_subpolicy, macro_vpred = pi.act(True, obs)
    #print("Current subpolicy", cur_subpolicy)

    ac = sub_policies[0].act(True, obs)[0]
    #ac, vpred = sub_policies[cur_subpolicy].act(True, obs)
    print("obs", obs)
    print("init_obs",init_obs)
    print("action subpolicy 0", ac)



    #TODO check if subpolicies should be given at every iteration
    # if t % macrolen == 0:
    #     #print("t , macrolen")
    #     cur_subpolicy, macro_vpred = pi.act(False, ob)
    #
    #     ac, vpred = sub_policies[cur_subpolicy].act(False, ob)

    obs, rew, new, info = env.step(ac)
    print("New", new)
    print("Reward", rew)
    print("Next observation", obs)
    cur_ep_ret += rew
    cur_ep_len += 1
    summary = tf.Summary(value=[tf.Summary.Value(tag="EpRew", simple_value = cur_ep_ret)])
    summary_writer.add_summary(summary, cur_ep_len)
    #time.sleep(3)


    if new:
        ob = env.reset()
        print("Environment SOLVED")
        break
