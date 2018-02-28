#!/usr/bin/env python3

import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys
import numpy as np
import time
from baselines import bench, logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple_obstacles
import os


from baselines.common import tf_util as U



env = gym.make('Box3DOF-v1')
env.addObstacle()
