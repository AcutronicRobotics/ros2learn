#!/usr/bin/env python3

import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys
import numpy as np

from mpi4py import MPI

from baselines import bench, logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds, tf_util as U
from baselines.ppo1 import mlp_policy, pposgd_simple
import os

from sensor_msgs.msg import Image as ImageMsg
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Pose

import threading # Used for time locks to synchronize position data.

import cv2
from darkflow.net.build import TFNet
import time

from darkflow.utils.utils import *

import rospy

import yaml
import glob

import quaternion as quat

import csv

# RK: Long Version
# dumps = list()
# # points_3d = list()
# cur_dir = os.getcwd()
# #models info for now is hardcoded to a particular folder:
# models_file = '/home/rkojcev/devel/darkflow/models_info/'
# os.chdir(models_file)
# annotations = sorted(os.listdir('.'))
# for i, file in enumerate(annotations):
#     print(i, file)
#     if not os.path.isdir(file):
#         print("annotations: ", file)
#         models_file_path = file
#         model_file = open(file)
#         yaml_model=yaml.load(model_file)
#         models_info = yaml_model
#         annotations.remove(file)
#
# print("models_info: ", models_info)

# Short version of loading models file
model_file = open('/home/rkojcev/devel/darkflow/models_info/models_info.yml')
yaml_model=yaml.load(model_file)
models_info = yaml_model
print("models_info: ", models_info)

take_point_once = 1

def _observation_image_callback(msg):
    """
    Code for processing the results of the vision CNN
    """
    # print("subscribed to image topic")
    # print("Received an image!")
    result_max = 0.
    result_max_iter = 0

    global take_point_once

    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        result = tfnet.return_predict(cv2_img)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if not result:
            print("No predicted points")
        else:
            for i in range(len(result)):
                # print("nr:",i, "confidence: ", result[i]['confidence'],"label: ", result[i]['label'], "center(x,y): ", int(result[i]['point_9']['x']),int(result[i]['point_9']['y']) )
                if result[i]['confidence'] > result_max:
                    result_max = result[i]['confidence']
                    result_max_iter = i
                    # print(result[i]['label'])

            # print("Label is: ",result[result_max_iter]['label'])
            # print(result_max_iter)

            if(result[result_max_iter]['label'] is "1"):
                label_human_readable = "coffe cup"

            if(result[result_max_iter]['label'] is "2"):
                label_human_readable = "juice box"


            if(result[result_max_iter]['label'] is "3"):
                label_human_readable = "rubik cube"

            corner_3D_1 = [models_info[int(result[result_max_iter]['label'])]['min_x'], models_info[int(result[result_max_iter]['label'])]['min_y'], models_info[int(result[result_max_iter]['label'])]['min_z']] # bottom left back
            corner_3D_2 = [models_info[int(result[result_max_iter]['label'])]['min_x'], models_info[int(result[result_max_iter]['label'])]['min_y'] + models_info[int(result[result_max_iter]['label'])]['size_y'], models_info[int(result[result_max_iter]['label'])]['min_z']] # bottom right back
            corner_3D_3 = [models_info[int(result[result_max_iter]['label'])]['min_x'], models_info[int(result[result_max_iter]['label'])]['min_y'],  models_info[int(result[result_max_iter]['label'])]['min_z'] + models_info[int(result[result_max_iter]['label'])]['size_z']] # up left back
            corner_3D_4 = [models_info[int(result[result_max_iter]['label'])]['min_x'], models_info[int(result[result_max_iter]['label'])]['min_y'] + models_info[int(result[result_max_iter]['label'])]['size_y'], models_info[int(result[result_max_iter]['label'])]['min_z']
                          + models_info[int(result[result_max_iter]['label'])]['size_z']] #up right back


            # the front rectangle of the box

            corner_3D_5 = [models_info[int(result[result_max_iter]['label'])]['min_x'] + models_info[int(result[result_max_iter]['label'])]['size_x'], models_info[int(result[result_max_iter]['label'])]['min_y'],  models_info[int(result[result_max_iter]['label'])]['min_z']] #bottom left front
            corner_3D_6 = [models_info[int(result[result_max_iter]['label'])]['min_x'] + models_info[int(result[result_max_iter]['label'])]['size_x'], models_info[int(result[result_max_iter]['label'])]['min_y'] + models_info[int(result[result_max_iter]['label'])]['size_y'], models_info[int(result[result_max_iter]['label'])]['min_z']] # bottom right front
            corner_3D_7 = [models_info[int(result[result_max_iter]['label'])]['min_x'] + models_info[int(result[result_max_iter]['label'])]['size_x'], models_info[int(result[result_max_iter]['label'])]['min_y'],  models_info[int(result[result_max_iter]['label'])]['min_z'] + models_info[int(result[result_max_iter]['label'])]['size_z']] # up left front
            corner_3D_8 = [models_info[int(result[result_max_iter]['label'])]['min_x'] + models_info[int(result[result_max_iter]['label'])]['size_x'], models_info[int(result[result_max_iter]['label'])]['min_y'] + models_info[int(result[result_max_iter]['label'])]['size_y'], models_info[int(result[result_max_iter]['label'])]['min_z']
                           + models_info[int(result[result_max_iter]['label'])]['size_z']] #up right front

            # here we calculate the center of the box
            corner_3D_9 = [models_info[int(result[result_max_iter]['label'])]['min_x'] + models_info[int(result[result_max_iter]['label'])]['size_x']*0.5,
                           models_info[int(result[result_max_iter]['label'])]['min_y'] + models_info[int(result[result_max_iter]['label'])]['size_y']*0.5,
                           models_info[int(result[result_max_iter]['label'])]['min_z'] + models_info[int(result[result_max_iter]['label'])]['size_z']*0.5]
            # here we assemble all 3D points
            corners3D = np.asarray([corner_3D_1, corner_3D_2, corner_3D_3, corner_3D_4, corner_3D_5, corner_3D_6, corner_3D_7, corner_3D_8, corner_3D_9], dtype=np.float)


            cv2.circle(cv2_img,(int(result[result_max_iter]['point_1']['x']),int(result[result_max_iter]['point_1']['y'])), 4, (255,255,0), -1)
            # cv2.putText(rgb_image,'1',(int(result[result_max_iter]['point_1']['x']),int(result[result_max_iter]['point_1']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

            cv2.circle(cv2_img,(int(result[result_max_iter]['point_2']['x']),int(result[result_max_iter]['point_2']['y'])), 4, (255,255,0), -1)
            # cv2.putText(rgb_image,'2',(int(result[result_max_iter]['point_2']['x']),int(result[result_max_iter]['point_2']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

            cv2.circle(cv2_img,(int(result[result_max_iter]['point_3']['x']),int(result[result_max_iter]['point_3']['y'])), 4, (255,255,0), -1)
            # cv2.putText(rgb_image,'3',(int(result[result_max_iter]['point_3']['x']),int(result[result_max_iter]['point_3']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

            cv2.circle(cv2_img,(int(result[result_max_iter]['point_4']['x']),int(result[result_max_iter]['point_4']['y'])), 4, (255,255,0), -1)
            # cv2.putText(rgb_image,'4',(int(result[result_max_iter]['point_4']['x']),int(result[result_max_iter]['point_4']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

            cv2.circle(cv2_img,(int(result[result_max_iter]['point_5']['x']),int(result[result_max_iter]['point_5']['y'])), 4, (255,255,0), -1)
            # cv2.putText(rgb_image,'5',(int(result[result_max_iter]['point_5']['x']),int(result[result_max_iter]['point_5']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

            cv2.circle(cv2_img,(int(result[result_max_iter]['point_6']['x']),int(result[result_max_iter]['point_6']['y'])), 4, (255,255,0), -1)
            # cv2.putText(rgb_image,'6',(int(result[result_max_iter]['point_6']['x']),int(result[result_max_iter]['point_6']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

            cv2.circle(cv2_img,(int(result[result_max_iter]['point_7']['x']),int(result[result_max_iter]['point_7']['y'])), 4, (255,255,0), -1)
            # cv2.putText(rgb_image,'7',(int(result[result_max_iter]['point_7']['x']),int(result[result_max_iter]['point_7']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

            cv2.circle(cv2_img,(int(result[result_max_iter]['point_8']['x']),int(result[result_max_iter]['point_8']['y'])), 4, (255,255,0), -1)
            # cv2.putText(rgb_image,'8',(int(result[result_max_iter]['point_8']['x']),int(result[result_max_iter]['point_8']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)
            cv2.circle(cv2_img,(int(result[result_max_iter]['point_9']['x']),int(result[result_max_iter]['point_9']['y'])), 4, (0,0,255), -1)
            # cv2.putText(imgcv,'center',(int(result[0]['point_9']['x']),int(result[0]['point_9']['y'])), font, 0.8,(255,255,255),2,cv2.LINE_AA)
            # cv2.imshow("depth camera", depth_map)
            # cv2.putText(rgb_image,result[result_max_iter]['label'],(int(result[result_max_iter]['point_1']['x']-30),int(result[result_max_iter]['point_1']['y']-30)), font, 1.0,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(cv2_img,label_human_readable,(int(result[result_max_iter]['point_1']['x']-30),int(result[result_max_iter]['point_1']['y']-30)), font, 0.8,(255,255,255),2,cv2.LINE_AA)

            corner_2D_1_pred = (int(result[result_max_iter]['point_1']['x']),int(result[result_max_iter]['point_1']['y']))
            corner_2D_2_pred = (int(result[result_max_iter]['point_2']['x']),int(result[result_max_iter]['point_2']['y']))
            corner_2D_3_pred = (int(result[result_max_iter]['point_3']['x']),int(result[result_max_iter]['point_3']['y']))
            corner_2D_4_pred = (int(result[result_max_iter]['point_4']['x']),int(result[result_max_iter]['point_4']['y']))
            corner_2D_5_pred = (int(result[result_max_iter]['point_5']['x']),int(result[result_max_iter]['point_5']['y']))
            corner_2D_6_pred = (int(result[result_max_iter]['point_6']['x']),int(result[result_max_iter]['point_6']['y']))
            corner_2D_7_pred = (int(result[result_max_iter]['point_7']['x']),int(result[result_max_iter]['point_7']['y']))
            corner_2D_8_pred = (int(result[result_max_iter]['point_8']['x']),int(result[result_max_iter]['point_8']['y']))
            corner_2D_9_pred = (int(result[result_max_iter]['point_9']['x']),int(result[result_max_iter]['point_9']['y']))

            corners_2D_pred = np.asarray([corner_2D_1_pred, corner_2D_2_pred,
                               corner_2D_3_pred, corner_2D_4_pred,
                               corner_2D_5_pred, corner_2D_6_pred,
                               corner_2D_7_pred, corner_2D_8_pred,
                                       corner_2D_9_pred], dtype=np.float)

            cv2.line(cv2_img,corner_2D_1_pred, corner_2D_2_pred, (0,255,0))
            cv2.line(cv2_img,corner_2D_2_pred, corner_2D_6_pred, (0,255,0))
            cv2.line(cv2_img,corner_2D_5_pred, corner_2D_6_pred, (0,255,0))
            cv2.line(cv2_img,corner_2D_2_pred, corner_2D_4_pred, (0,255,0))
            cv2.line(cv2_img,corner_2D_6_pred, corner_2D_8_pred, (0,255,0))
            cv2.line(cv2_img,corner_2D_5_pred, corner_2D_7_pred, (0,255,0))
            cv2.line(cv2_img,corner_2D_1_pred, corner_2D_3_pred, (0,255,0))
            cv2.line(cv2_img,corner_2D_3_pred, corner_2D_4_pred, (0,255,0))
            cv2.line(cv2_img,corner_2D_4_pred, corner_2D_8_pred, (0,255,0))
            cv2.line(cv2_img,corner_2D_7_pred, corner_2D_8_pred, (0,255,0))
            cv2.line(cv2_img,corner_2D_3_pred, corner_2D_7_pred, (0,255,0))
            cv2.line(cv2_img,corner_2D_1_pred, corner_2D_5_pred, (0,255,0))

            objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')
            K = np.array(internal_calibration, dtype='float32')

            R_pred, t_pred = pnp(corners3D,  corners_2D_pred, K)

            Rt_pred = np.concatenate((R_pred, t_pred), axis=1)

            # note to RK: here you need to multiply the camera transformation with the Rt_pred and take that parameters in the final calculation

            # print("Rt_pred: ", Rt_pred)

            #now here publish the detected target position from the vision system. And calculate camera to world so we get the final point to the world:
            # is it good idea for this to be detected on the first time we load something? or streem continiously.
            #If we stream continiously when the robot covers the cube we cant detect anything and if the target is updated at that time
            #uncomment if we want to use like servoing every time, just wont work if the robot is in front of the object!!!
            cam_pose_x = -0.496768318 #+ 0.0488#-0.5087683179567231 # random.uniform(-0.25, -0.6)#-0.5087683179567231#0.0 #random.uniform(-0.25, -0.6)#-0.5087683179567231#random.uniform(-0.3, -0.6)#random.uniform(-0.25, -0.6) # -0.5087683179567231#
            cam_pose_y = -0.013376 #+ 0.0488#-0.040128#-0.013376#random.uniform(0.0, -0.2)
            cam_pose_z = 1.2808068867058566+0.0488 #1.4808068867058566

            # print("t_pred[2]: ",t_pred)

            if t_pred[2] > 0.0:
                t_pred[2] = -t_pred[2]

            pose_target = Pose()
            pose_target.position.x = -t_pred[0]/3.0 + cam_pose_x
            pose_target.position.y = -t_pred[1]/3.0 - cam_pose_y
            pose_target.position.z = t_pred[2]/3.0 + cam_pose_z

            q_rubik = quat.from_rotation_matrix(R_pred)
            # print("q_rubik: ", q_rubik.x, q_rubik.y, q_r
            pose_target.orientation.x = q_rubik.x#0.0#q_rubik[0]
            pose_target.orientation.y = q_rubik.y#0.0#q_rubik[1]
            pose_target.orientation.z = q_rubik.z#0.0#q_rubik[2]
            pose_target.orientation.w = q_rubik.w#0.0#q_rubik[3]
            # uncomment this if we want to do like servoing
            _pub_target.publish(pose_target)

            if take_point_once is 1:
                f_tgt = open(logger.get_dir() + '/target.csv', 'w')#
                TFields = ['EE_POS_TGT','EE_ROT_TGT']
                writer_tgt = csv.DictWriter(f_tgt, fieldnames=TFields)
                writer_tgt.writeheader()
                writer_tgt.writerow({'EE_POS_TGT': str(np.asarray([pose_target.position.x, pose_target.position.y, pose_target.position.z])),'EE_ROT_TGT': str(R_pred)})
                take_point_once = 0

        cv2.imshow("Image window", cv2_img)
        cv2.waitKey(3)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slowness', help='time for executing trajectory', type=int, default=1)
parser.add_argument('--slowness-unit', help='slowness unit',type=str, default='sec')
args = parser.parse_args()

env = gym.make('MARAVisionOrientCollision-v0')
env.init_time(slowness= args.slowness, slowness_unit=args.slowness_unit, reset_jnts=False)
logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo1/' + str(args.slowness) + '_' + str(args.slowness_unit) + '/'

logger.configure(os.path.abspath(logdir))
print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))


# rate = rospy.Rate(0.3) # 10hz


options = {"pbLoad": "/home/rkojcev/devel/darkflow/built_graph/yolo-new.pb", "metaLoad": "/home/rkojcev/devel/darkflow/built_graph/yolo-new.meta", "threshold": 0.02, "gpu": 1.00}
tfnet = TFNet(options)

bridge = CvBridge()
TARGET_PUBLISHER = '/mara/target'
# Read intrinsic camera parameters
internal_calibration = get_camera_intrinsic()
_sub_image = rospy.Subscriber("/mara/rgb/image_raw", ImageMsg, _observation_image_callback)
_pub_target = rospy.Publisher(TARGET_PUBLISHER, Pose)

# env = Monitor(env, logger.get_dir(),  allow_early_resets=True)

rank = MPI.COMM_WORLD.Get_rank()
sess = U.single_threaded_session()
sess.__enter__()

# # remove seeds for now
seed = 0
workerseed = seed + 10000 * rank
set_global_seeds(seed)
env.seed(seed)


# seed = 0
# set_global_seeds(seed)

env.goToInit()
time.sleep(3)

# initial_observation = env.reset()
# print("Initial observation: ", initial_observation)

# U.make_session(num_cpu=1).__enter__()


env.seed(seed)

def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
        hid_size=128, num_hid_layers=4)

pposgd_simple.learn(env, policy_fn,
                    max_timesteps=1e8,
                    timesteps_per_actorbatch=2048,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=10, optim_stepsize=3e-4, gamma=0.99,
                    optim_batchsize=128, lam=0.95, schedule='linear', save_model_with_prefix='mara_orient_ppo1_test', outdir=logger.get_dir()) #

# def policy_fn(name, ob_space, ac_space):
#     return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
#         hid_size=256, num_hid_layers=3)

# pposgd_simple.learn(env, policy_fn,
#                     max_timesteps=1e8,
#                     timesteps_per_actorbatch=2048,
#                     clip_param=0.2, entcoeff=0.0,
#                     optim_epochs=10, optim_stepsize=3e-4, gamma=0.99,
#                     optim_batchsize=256, lam=0.95, schedule='linear', save_model_with_prefix='mara_orient_ppo1_test', outdir=logger.get_dir()) #

env.close()


# env.monitor.close()
