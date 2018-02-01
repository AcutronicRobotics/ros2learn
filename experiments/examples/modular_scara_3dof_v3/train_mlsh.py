import argparse
import tensorflow as tf


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
import mlsh_code.master_robotics_mult as master_robotics
# import mlsh_code.master as master
import gym_gazebo
from baselines import bench, logger
import os

from baselines import logger

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#replay = str2bool(replay)
#replay = str2bool(replay)
savename = 'ScaraTest'
continue_iter = None
RELPATH = osp.join(savename)
LOGDIR = osp.join('/root/results' if sys.platform.startswith('linux') else '/tmp', RELPATH)
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
        # RK change back to 5
        if it % 2 == 0 and it > 1 and not replay:
            basePath = '/tmp/rosrl/mlsh/'
            if not os.path.exists(basePath):
                os.makedirs(basePath)
            # print("calling the save network from here: ")
            modelF= basePath + "/saved_models/" + str('%.5i'%it) # + ".model"
            U.save_state(modelF)
            logger.log("Saved model to file :{}".format(modelF))
            # fname = osp.join("savedir/", 'checkpoints', '%.5i'%it)
            # # logger.log('Saving model to %s'%fname)
            # U.save_state(fname)
    if it == 0 and continue_iter is not None:
        #fname = osp.join(""+args.savename+"/checkpoints/", str(args.continue_iter))
        fname = osp.join(""+savename+"/checkpoints/", str(continue_iter))
        U.load_state(fname)

        # fname = osp.join(""+args.savename+"/checkpoints/", args.continue_iter)
        # subvars = []
        # for i in range(args.num_subs-1):
        #     subvars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="sub_policy_%i" % (i+1))
        # print([v.name for v in subvars])
        # U.load_state(fname, subvars)
        pass

def train(env, savename, replay, macro_duration, num_subs,  num_rollouts, warmup_time, train_time, force_subpolicy, store):
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
    master_robotics.start(callback, env, savename, replay, macro_duration, num_subs,  num_rollouts, warmup_time, train_time, force_subpolicy, store, workerseed=workerseed, rank=rank, comm=comm)

#def main(job_id, env, savename, replay, macro_duration, num_subs,  num_rollouts, warmup_time, train_time, force_subpolicy, store):
def main(env, savename, replay, macro_duration, num_subs, num_rollouts, warmup_time, train_time, force_subpolicy, store):
    if MPI.COMM_WORLD.Get_rank() == 0 and osp.exists(LOGDIR):
        shutil.rmtree(LOGDIR)
    MPI.COMM_WORLD.Barrier()
    # with logger.session(dir=LOGDIR):
    train(env, savename, replay, macro_duration, num_subs,  num_rollouts, warmup_time, train_time, force_subpolicy, store)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--optimize', type=bool)
    # args = parser.parse_args()
    #
    env = 'GazeboModularScara3DOF-v3'

    # if 'optimize' == True:
    #     main(job_id, env, savename, replay, params['macro_duration'], params['num_subs'], params['num_rollouts'], params['warmup_time'],  params['train_time'], force_subpolicy, store)
    # else:
    #     #Parameters set by user
    #     job_id = None
    savename = 'ScaraTest'
    replay=False
    macro_duration = 1
    num_subs = 2
    num_rollouts = 2500
    warmup_time = 20 #1 # 30
    train_time = 200 #2 # 200
    force_subpolicy=None
    store=True

    main(env, savename, replay, macro_duration, num_subs, num_rollouts, warmup_time, train_time, force_subpolicy, store)
