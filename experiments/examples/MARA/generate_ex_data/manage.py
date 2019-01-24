
import rospy
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import os
import numpy as np
import time
import subprocess
from scipy.interpolate import interpolate
import random

import sys
import gym
import gym_gazebo_2
import tensorflow as tf
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.gail import mlp_policy

save_path = "/home/yue/experiments/expert_data/mara/collisions_model"

def print_last10_rollouts(ed, traj):
    for roll in range( len(ed.f.acs)-11, len(ed.f.acs)-1 ):
        print(ed.f.acs[traj][roll])

def publish_expert_data(ed, robot):
    action_msg = JointTrajectory()
    joints = ['motor1', 'motor2', 'motor3','motor4', 'motor5', 'motor6']
    action_msg.joint_names = joints

    target = JointTrajectoryPoint()

    # roscore = subprocess.Popen('roscore')
    # time.sleep(1)  # wait
    pub = rospy.Publisher('/mara_controller/command', JointTrajectory, queue_size=100)
    rospy.init_node('ex_data', anonymous=True)

    while not rospy.is_shutdown():
        for traj in range( 0, len(ed.f.obs) ):
            print("obs_traj: ", traj)
            for roll in range( 0, len(ed.f.obs[traj]) ):
                # roll = 99
                ex1 = np.array(ed.f.obs[traj][roll][0])
                ex2 = np.array(ed.f.obs[traj][roll][1])
                ex3 = np.array(ed.f.obs[traj][roll][2])
                if robot == 'scara':
                    target.positions = [ex1, ex2, ex3]
                    target.time_from_start.secs = 1
                    action_msg.points = [target]
                    time.sleep(2.0)  # wait
                    pub.publish(action_msg)
                    print("obs", roll)

                elif robot == 'mara':
                    ex4 = np.array(ed.f.obs[traj][roll][3])
                    ex5 = np.array(ed.f.obs[traj][roll][4])
                    ex6 = np.array(ed.f.obs[traj][roll][5])
                    target.positions = [ex1, ex2, ex3, ex4, ex5, ex6]
                    target.time_from_start.secs = 1
                    action_msg.points = [target]
                    # time.sleep(2.0)  # wait
                    pub.publish(action_msg)
                    print("obs", roll)

        for traj in range( 0, len(ed.f.acs) ):
            print("acs_traj: ", traj)
            for roll in range( 0, len(ed.f.acs[traj]) ):
                # roll = 99
                ex1 = np.array(ed.f.acs[traj][roll][0])
                ex2 = np.array(ed.f.acs[traj][roll][1])
                ex3 = np.array(ed.f.acs[traj][roll][2])
                if robot == 'scara':
                    target.positions = [ex1, ex2, ex3]
                    target.time_from_start.secs = 1
                    action_msg.points = [target]
                    time.sleep(2.0)  # wait
                    pub.publish(action_msg)
                    print("acs", roll)

                elif robot == 'mara':
                    ex4 = np.array(ed.f.acs[traj][roll][3])
                    ex5 = np.array(ed.f.acs[traj][roll][4])
                    ex6 = np.array(ed.f.acs[traj][roll][5])
                    target.positions = [ex1, ex2, ex3, ex4, ex5, ex6]
                    target.time_from_start.secs = 1
                    action_msg.points = [target]
                    # time.sleep(2.0)  # wait
                    pub.publish(action_msg)
                    print("acs", roll)

def add_rollout(ed):
    new_obs_roll = []
    new_obs_traj = []
    new_roll = []
    new_joint = []

    for traj in range( 0, len(ed.f.obs) ):
        for roll in range( 0, len(ed.f.obs[traj])-1 ):
            if roll == (len(ed.f.obs[traj])-2):
                for joint in range( 0, len(ed.f.obs[traj][roll]) ):
                    x = ed.f.obs[traj][roll][joint]
                    y = ed.f.obs[traj][roll + 1][joint]
                    new = np.linspace(x, y, num=3)
                    new_joint.append(new[1])
                new_roll.append(new_joint)
                new_joint = []

    count = 0;

    for traj in range( 0, len(ed.f.obs) ):
        for roll in range( 0, len(ed.f.obs[traj]) ):
            if roll == (len(ed.f.obs[traj])-1):
                new_obs_roll.append(new_roll[count])
                count += 1
                # print(count)
            new_obs_roll.append(ed.f.obs[traj][roll])
        new_obs_traj.append(new_obs_roll)
        new_obs_roll = []

    new_acs_roll = []
    new_acs_traj = []
    new_roll = []
    new_joint = []

    for traj in range( 0, len(ed.f.acs) ):
        for roll in range( 0, len(ed.f.acs[traj])-1 ):
            if roll == (len(ed.f.obs[traj])-2):
                for joint in range( 0, len(ed.f.acs[traj][roll]) ):
                    x = ed.f.acs[traj][roll][joint]
                    y = ed.f.acs[traj][roll + 1][joint]
                    new = np.linspace(x, y, num=3)
                    new_joint.append(new[1])
                new_roll.append(new_joint)
                new_joint = []

    count = 0;

    for traj in range( 0, len(ed.f.acs) ):
        for roll in range( 0, len(ed.f.acs[traj]) ):
            if roll == (len(ed.f.acs[traj])-1):
                new_acs_roll.append(new_roll[count])
                count += 1
            new_acs_roll.append(ed.f.acs[traj][roll])
        new_acs_traj.append(new_acs_roll)
        new_acs_roll = []

    np.savez( os.path.join( save_path, 'expert_data'), obs=(new_obs_traj), acs=(new_acs_traj) )


def increase_rollouts(ed, times, num_new_roll):
    new_obs_roll = []
    new_obs_traj = []
    new_roll = []
    new_joint = []
    # print(ed.f.acs)
    for traj in range( 0, len(ed.f.obs) ):
        for roll in range( 0, len(ed.f.obs[traj])-1, 1 ):
            for joint in range( 0, len(ed.f.obs[traj][roll]) ):
                x = ed.f.obs[traj][roll][joint]
                y = ed.f.obs[traj][roll + 1][joint]
                new = np.linspace(x, y, num=num_new_roll)
                for n in range (1, len(new) - 1): #en cada n vienen dos
                    new_joint.append(new[n])
            for num in range(0, num_new_roll - 2):
                new_roll.append(new_joint[num::(num_new_roll - 2)])
            new_joint = []
        print(traj)

    count = 0;
    for traj in range( 0, len(ed.f.obs) ):
        for roll in range( 0, len(ed.f.obs[traj]) ):
            if( (roll != 0 ) and ( roll + (1-1) ) % 1 == 0 ):
                for num in range(0, num_new_roll - 2):
                    new_obs_roll.append(new_roll[count + num])
                count += (num_new_roll - 2)
                # print(count)
            new_obs_roll.append(ed.f.obs[traj][roll])
        new_obs_traj.append(new_obs_roll)
        new_obs_roll = []

    new_acs_roll = []
    new_acs_traj = []
    new_roll = []
    new_joint = []

    for traj in range( 0, len(ed.f.acs) ):
        for roll in range( 0, len(ed.f.acs[traj])-1, 1 ):
            for joint in range( 0, len(ed.f.acs[traj][roll]) ):
                x = ed.f.acs[traj][roll][joint]
                y = ed.f.acs[traj][roll + 1][joint]
                new = np.linspace(x, y, num=num_new_roll)
                for n in range (1, len(new) - 1):
                    new_joint.append(new[n])
            for num in range(0, num_new_roll - 2):
                new_roll.append(new_joint[num::(num_new_roll - 2)])
            new_joint = []
        print(traj)

    count = 0;
    for traj in range( 0, len(ed.f.acs) ):
        for roll in range( 0, len(ed.f.acs[traj]) ):
            if( (roll != 0 ) and ( roll + (1-1) ) % 1 == 0 ):
                for num in range(0, num_new_roll - 2):
                    new_acs_roll.append(new_roll[count + num])
                count += (num_new_roll - 2)
            new_acs_roll.append(ed.f.acs[traj][roll])
        new_acs_traj.append(new_acs_roll)
        new_acs_roll = []

    np.savez( os.path.join( save_path, 'expert_data'), obs=(new_obs_traj), acs=(new_acs_traj) )

def match_last(ed, robot):
    cp = []
    cp_last = []
    for traj in range ( 0, len(ed.f.acs) ):
        for roll in ed.f.acs[traj][9]:
            cp.append(roll)
        cp_last.append(cp)
        cp = []

    count = 0
    new_obs_traj = []
    new_obs_roll = []
    for traj in range( 0, len(ed.f.obs) ):
        for roll in range( 0, len(ed.f.obs[traj]) ):
            if roll == len(ed.f.obs[traj]) - 1:
                old_joint = ed.f.obs[traj][roll]
                new_joint = np.concatenate( (cp_last[count], old_joint[6:]), axis=0 )
                count += 1
                new_obs_roll.append(new_joint)
            else:
                new_obs_roll.append(ed.f.obs[traj][roll])
        new_obs_traj.append(new_obs_roll)
        new_obs_roll = []

    actions = ed.f.acs

    np.savez( os.path.join( save_path, 'expert_data'), obs=(new_obs_traj), acs=(actions) )

def print_shape(expertdata):
    # print(expertdata.f.obs)
    # print(expertdata.f.acs)
    print(expertdata.f.obs.shape)
    print(expertdata.f.acs.shape)

def print_expert_data(ed):
    for obs in ed.f.obs:
        print(obs)
    # for acs in ed.f.acs:
        print(acs)

def delete_rollouts(ed, poss):
    new_obs = np.delete(ed.f.obs, poss, 1)
    new_acs = np.delete(ed.f.acs, poss, 1)

    np.savez( os.path.join( save_path, 'expert_data'), obs=(new_obs), acs=(new_acs) )

def add_trajs(ed, mu):
    new_obs_roll = []
    new_acs_roll = []
    new_obs_traj = ed.f.obs.tolist()
    new_acs_traj = ed.f.acs.tolist()
    sigma = random.uniform(0.0001, 0.0003)
    for m in range( 0, len(mu) ): #num of new traj
        noise_obs = np.random.normal(mu[m], sigma, ed.f.obs.shape)
        noise_acs = np.random.normal(mu[m], sigma, ed.f.acs.shape)
        for traj in range( 0, len(ed.f.obs) ):
            for roll in range( 0, len(ed.f.obs[traj]) ):
                if roll == 0 or roll == ( len(ed.f.obs[traj]) - 1 ):
                    new_obs_roll.append(ed.f.obs[traj][roll])
                    new_acs_roll.append(ed.f.acs[traj][roll])
                else:
                    new_obs_joint = ed.f.obs[traj][roll] + noise_obs[traj][roll]
                    new_obs_roll.append(new_obs_joint)
                    new_acs_joint = ed.f.acs[traj][roll] + noise_acs[traj][roll]
                    new_acs_roll.append(new_acs_joint)
            new_obs_traj.append(new_obs_roll)
            new_obs_roll = []
            new_acs_traj.append(new_acs_roll)
            new_acs_roll = []

    np.savez( os.path.join( save_path, 'expert_data'), obs=(new_obs_traj), acs=(new_acs_traj) )

ed = np.load( save_path + '/1_10_bc_down.npz')
# match_last(ed, 'mara')
# add_trajs(ed, [0] * 99) # * __ num of new traj
# increase_rollouts(ed, 1, 40) #every _ rollout, num_new_roll = _ - 2 | 112--> 1000
# 18 --> 154 | 34 --> 298 | 56 --> 496 | 12 -->100 | 6 --> 46
publish_expert_data(ed, 'mara')
# add_rollout(ed)
# delete_rollouts(ed, [10, 32, 54]) #positions of rollouts to be removed
ed = np.load(save_path + '/expert_data.npz')
print_shape(ed)
# print_expert_data(ed)
# print_last10_rollouts(ed, 0) #num traj
