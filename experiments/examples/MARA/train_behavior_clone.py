'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''

import argparse
import tempfile
import os.path as osp
import gym
import logging
from tqdm import tqdm

import tensorflow as tf

from baselines.gail import mlp_policy
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam
from baselines.gail.dataset.h_ros_dset import H_ros_Dset
from train_gail import runner
import numpy as np

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--env_id', help='environment ID', default='MARA-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='/home/yue/experiments/expert_data/mara/collisions_model/100_46_bc_down.npz')###############
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log/bc')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=5)###############################################################################
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e5)
    return parser.parse_args()


def learn(env, policy_func, dataset, optim_batch_size=128, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=3e-4,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):

    val_per_iter = int(max_iters/10)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    loss = tf.reduce_mean(tf.square(ac-pi.ac))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    adam.sync()
    logger.log("Pretraining with Behavior Cloning...")
    summary_writer=tf.summary.FileWriter('/home/yue/experiments/tensorboard/mara/collisions_model/bc_down/5_46',graph=tf.get_default_graph())##########

    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        train_loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            val_loss, _ = lossandgrad(ob_expert, ac_expert, True)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss, val_loss))

            val_loss_sum = tf.Summary(value=[tf.Summary.Value(tag="val_loss", simple_value = np.mean(val_loss))])
            summary_writer.add_summary(val_loss_sum, iter_so_far)

        train_loss_sum = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value = np.mean(train_loss))])
        summary_writer.add_summary(train_loss_sum, iter_so_far)

        ob_expert_sum = tf.Summary(value=[tf.Summary.Value(tag="ob_expert", simple_value = np.mean(ob_expert))])
        summary_writer.add_summary(ob_expert_sum, iter_so_far)

        ac_expert_sum = tf.Summary(value=[tf.Summary.Value(tag="ac_expert", simple_value = np.mean(ac_expert))])
        summary_writer.add_summary(ac_expert_sum, iter_so_far)

        # if iter_so_far % 10000 == 0 and iter_so_far != 0:
        #     savedir_fname = "/home/yue/baselines/baselines/gail/checkpoint/bc_10_46-" + str(iter_so_far)
        #     U.save_state(savedir_fname, var_list=pi.get_variables())

    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = "/home/yue/experiments/checkpoint/mara/bc/down/5_46"#######################################################
    U.save_state(savedir_fname, var_list=pi.get_variables())
    return savedir_fname


def get_task_name(args):
    task_name = 'BC'
    task_name += '.{}'.format(args.env_id.split("-")[0])
    task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    dataset = H_ros_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
    savedir_fname = learn(env,
                          policy_fn,
                          dataset,
                          max_iters=args.BC_max_iter,
                          ckpt_dir=args.checkpoint_dir,
                          log_dir=args.log_dir,
                          task_name=task_name,
                          verbose=True)
    # savedir_fname = "/home/yue/baselines/baselines/gail/checkpoint/bc_10_100_repeated2"
    # avg_len, avg_ret = runner(env,
    #                           policy_fn,
    #                           savedir_fname,
    #                           timesteps_per_batch=1024,
    #                           number_trajs=10,
    #                           stochastic_policy=args.stochastic_policy,
    #                           save=args.save_sample,
    #                           reuse=True)


if __name__ == '__main__':
    args = argsparser()
    main(args)
