import argparse
import tensorflow as tf
#parser.add_argument('savename', type=str)
#parser.add_argument('--task', type=str)
#parser.add_argument('--num_subs', type=int)
#parser.add_argument('--macro_duration', type=int)
#parser.add_argument('--num_rollouts', type=int)
#parser.add_argument('--warmup_time', type=int)
#parser.add_argument('--force_subpolicy', type=int)
#arser.add_argument('--replay', type=str)
#parser.add_argument('--train_time', type=int)
#parser.add_argument('-s', action='store_true')
#parser.add_argument('--continue_iter', type=str)
#args = parser.parse_args()

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
import gym_gazebo

import mlsh_code.rollouts as rollouts
from mlsh_code.policy_network import Policy
from mlsh_code.subpolicy_network import SubPolicy
from mlsh_code.observation_network import Features
from mlsh_code.learner import Learner
import rl_algs.common.tf_util as U
#
# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

# #replay = str2bool(replay)
# #replay = str2bool(replay)
# savename = 'ScaraTest'
# continue_iter = None
# RELPATH = osp.join(savename)
# LOGDIR = osp.join('/root/results' if sys.platform.startswith('linux') else '/tmp', RELPATH)
# def callback(it):
#     if it >= 1:
#         fname = osp.join("/Users/kevin/data/tinkerbell/gce/"+args.savename+"/checkpoints/", format(it*5, '05d'))
#         U.load_state(fname)
#     else:
#         fname = osp.join("/Users/kevin/data/tinkerbell/gce/"+args.savename+"/checkpoints/", "00005")
#         subvars = []
#         for i in range(args.num_subs):
#             subvars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="sub_policy_%i" % i)
#         U.load_state(fname, subvars)

def callback(it):
    if MPI.COMM_WORLD.Get_rank()==0:
        if it % 5 == 0 and it > 3 and not replay:
            fname = osp.join("savedir/", 'checkpoints', '%.5i'%it)
            # logger.log('Saving model to %s'%fname)
            U.save_state(fname)
    if it == 0 and continue_iter is not None:
        #fname = osp.join(""+args.savename+"/checkpoints/", str(args.continue_iter))
        # osp.join("/home/rkojcev/devel/mlsh_aux/savedir/checkpoints/0070.meta")
        # fname = "/home/rkojcev/devel/gps/experiments/ur_tf_mdgps_example/tf_train/policy_model.ckpt"
        U.load_state(fname)

        # fname = osp.join(""+args.savename+"/checkpoints/", args.continue_iter)
        # subvars = []
        # for i in range(args.num_subs-1):
        #     subvars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="sub_policy_%i" % (i+1))
        # print([v.name for v in subvars])
        # U.load_state(fname, subvars)
        pass

if __name__ == "__main__":
    env = gym.make('GazeboModularScara4DOF-v3')
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    filename = "/home/rkojcev/devel/mlsh_aux/savedir/checkpoints/00075"
    saver = tf.train.import_meta_graph('/home/rkojcev/devel/mlsh_aux/savedir/checkpoints/00075.meta')
    saver.restore(sess, filename)
    graph = tf.get_default_graph()
    print(tf.global_variables())
    num_subs = 2
    ob_space = env.observation_space
    ac_space = env.action_space

    ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0]])

    print("ob: ", ob)

    pi = Policy(name="policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)

    print("policy: ", pi)

    while True:
        feed_dict = {ob: observation}
        obs = sess.run(out, feed_dict=feed_dict)

        action = pi.act(True, ob)[0]
        obs, reward, done, info = env.step(action)
        print(action)
    # # fname = osp.join(""+savename+"/checkpoints/", str(continue_iter))
    # fname = osp.join("/home/rkojcev/devel/mlsh_aux/savedir/checkpoints/0070.meta")
    # U.load_state(fname)
    # replay = False
