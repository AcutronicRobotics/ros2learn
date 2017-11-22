import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys
import numpy as np

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple

# from baselines.agent.utility.general_utils import get_ee_points, get_position


def train_setup(job_id, max_t, t_per_actorbatch, optim_ep, optim_st, gam, optim_batchs, l):
    env = gym.make('GazeboModularScara3DOF-v3')
    initial_observation = env.reset()
    print("Initial observation: ", initial_observation)
    env.render()
    seed = 0


    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)

    env.seed(seed)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    optim_metric = pposgd_simple.learn(env, policy_fn,
                        max_timesteps=max_t,
                        timesteps_per_actorbatch=t_per_actorbatch,
                        clip_param=0.2, entcoeff=0.0,
                        optim_epochs=optim_ep, optim_stepsize=optim_st, gamma=gam,
                        optim_batchsize=optim_batchs, lam=l, schedule='linear', save_model_with_prefix='ros1_ppo1_test_O')
    return optim_metric

def main(job_id, params):
    return train_setup(job_id, params['max_timesteps'], params['timesteps_per_actorbatch'], params['optim_epochs'], params['optim_stepsize'],  params['gamma'], params['optim_batchsize'], params['lam'])
