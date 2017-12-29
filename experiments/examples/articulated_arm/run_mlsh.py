import argparse
import tensorflow as tf

from mpi4py import MPI
from rl_algs.common import set_global_seeds, tf_util as U
import os.path as osp
import gym, logging
import numpy as np
from collections import deque
from gym import spaces
import rl_algs.common.misc_util
import sys
import shutil
import subprocess
import gym_gazebo
import time

import time
import gym_gazebo

import mlsh_code.rollouts_robotics as rollouts
from mlsh_code.policy_network import Policy
from mlsh_code.subpolicy_network import SubPolicy
from mlsh_code.observation_network import Features
from mlsh_code.learner import Learner
import rl_algs.common.tf_util as U
import pickle

# here we define the parameters necessary to launch
savename = 'ScaraTest'
replay=False
macro_duration = 10
num_subs = 2
num_rollouts = 2500
warmup_time = 30 #1 # 30
train_time = 200 #2 # 200
force_subpolicy=None
store=True



# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

# replay = False
# args.replay = str2bool(args.replay)

RELPATH = osp.join(savename)
LOGDIR = osp.join('/root/results' if sys.platform.startswith('linux') else '/tmp', RELPATH)

def start(callback, workerseed, rank, comm):
    env = gym.make('GazeboModularArticulatedArm4DOF-v1')
    env.seed(workerseed)
    np.random.seed(workerseed)
    ob_space = env.observation_space
    ac_space = env.action_space
    stochastic=False

    # num_subs = args.num_subs
    # macro_duration = args.macro_duration
    # num_rollouts = args.num_rollouts
    # warmup_time = args.warmup_time
    # train_time = args.train_time

    # num_batches = 15

    # observation in.
    ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0]])
    policy = Policy(name="policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)
    old_policy = Policy(name="old_policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)

    sub_policies = [SubPolicy(name="sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]
    old_sub_policies = [SubPolicy(name="old_sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]

    learner = Learner(env, policy, old_policy, sub_policies, old_sub_policies, comm, clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-5, optim_batchsize=64)
    rollout = rollouts.traj_segment_generator(policy, sub_policies, env, macro_duration, num_rollouts, replay, force_subpolicy, stochastic=stochastic)
    #
    callback(0)
    learner.syncSubpolicies()
    policy.reset()
    learner.syncMasterPolicies()
    env.randomizeCorrect()
    shared_goal = comm.bcast(env.realgoal, root=0)
    print("The goal to %s" % (env.realgoal))
    obs=env.reset()
    print("OBS")
    t = 0

    time.sleep(10)
    while True:
        #print("t", t)
        if t % macro_duration == 0:
            cur_subpolicy, macro_vpred = policy.act(stochastic, obs)

        ac, vpred = sub_policies[1].act(stochastic, obs)

        obs, rew, new, info = env.step(ac)

        if new:
            print("ENVIRONMENT SOLVED")
            time.sleep(20)
        t += 1


def callback(it):
    if MPI.COMM_WORLD.Get_rank()==0:
        if it % 5 == 0 and it > 3: # and not replay:
            fname = osp.join("savedir/", 'checkpoints', '%.5i'%it)
            U.save_state(fname)
    if it == 0:
        print("CALLBACK")
        fname = '/home/rkojcev/baselines_networks/mlsh/saved_models/00040'
        subvars = []
        for i in range(num_subs-1):
            subvars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="sub_policy_%i" % (i+1))
        print([v.name for v in subvars])
        U.load_state(fname, subvars)
        time.sleep(5)
        pass

def load():
    num_timesteps=1e9
    seed = 1401
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 1000 * MPI.COMM_WORLD.Get_rank()
    rank = MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    # if rank != 0:
    #     logger.set_level(logger.DISABLED)
    # logger.log("rank %i" % MPI.COMM_WORLD.Get_rank())

    world_group = MPI.COMM_WORLD.Get_group()
    mygroup = rank % 10
    theta_group = world_group.Incl([x for x in range(MPI.COMM_WORLD.size) if (x % 10 == mygroup)])
    comm = MPI.COMM_WORLD.Create(theta_group)
    comm.Barrier()
    # comm = MPI.COMM_WORLD

    #master_robotics.start(callback, args=args, workerseed=workerseed, rank=rank, comm=comm)
    start(callback, workerseed=workerseed, rank=rank, comm=comm)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--optimize', type=bool)
    # args = parser.parse_args()
    #
    # env = 'GazeboModularScara4DOF-v3'

    # if 'optimize' == True:
    #     main(job_id, env, savename, replay, params['macro_duration'], params['num_subs'], params['num_rollouts'], params['warmup_time'],  params['train_time'], force_subpolicy, store)
    # else:
    #     #Parameters set by user
    #     job_id = None


    if MPI.COMM_WORLD.Get_rank() == 0 and osp.exists(LOGDIR):
        shutil.rmtree(LOGDIR)
    MPI.COMM_WORLD.Barrier()
    # with logger.session(dir=LOGDIR):
    load()

if __name__ == '__main__':
    main()
