import gym
import gym_gazebo
import tensorflow as tf
import argparse
import copy
import sys
import numpy as np

from baselines import bench, logger

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
# from baselines.ppo2.policies import MlpPolicy, LstmPolicy, LnLstmPolicy, LstmMlpPolicy
import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from baselines.common.cmd_util import common_arg_parser, parse_unknown_args

from importlib import import_module
import multiprocessing

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import os
import time

from sensor_msgs.msg import Image as ImageMsg
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Pose

import threading # Used for time locks to synchronize position data.

import cv2
from darkflow.net.build import TFNet

from darkflow.utils.utils import *

import rospy

import yaml
import glob

import quaternion as quat

def _observation_image_callback(msg):
    """
    Code for processing the results of the vision CNN
    """
    # print("subscribed to image topic")
    # print("Received an image!")
    result_max = 0.
    result_max_iter = 0

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

            # print("Rt_pred: ", Rt_pred)

            #now here publish the detected target position from the vision system. And calculate camera to world so we get the final point to the world:
            # is it good idea for this to be detected on the first time we load something? or streem continiously.
            #If we stream continiously when the robot covers the cube we cant detect anything and if the target is updated at that time
            #uncomment if we want to use like servoing every time, just wont work if the robot is in front of the object!!!
            cam_pose_x = -0.5087683179567231 # random.uniform(-0.25, -0.6)#-0.5087683179567231#0.0 #random.uniform(-0.25, -0.6)#-0.5087683179567231#random.uniform(-0.3, -0.6)#random.uniform(-0.25, -0.6) # -0.5087683179567231#
            cam_pose_y = -0.013376#random.uniform(0.0, -0.2)
            cam_pose_z = 1.3808068867058566 #1.4808068867058566

            pose_target = Pose()
            pose_target.position.x = -t_pred[0]/3.0 + cam_pose_x
            pose_target.position.y = -t_pred[1]/3.0 - cam_pose_y
            pose_target.position.z =  t_pred[2]/3.0 + cam_pose_z

            q_rubik = quat.from_rotation_matrix(R_pred)
            # print("q_rubik: ", q_rubik.x, q_rubik.y, q_r
            pose_target.orientation.x = q_rubik.x#0.0#q_rubik[0]
            pose_target.orientation.y = q_rubik.y#0.0#q_rubik[1]
            pose_target.orientation.z = q_rubik.z#0.0#q_rubik[2]
            pose_target.orientation.w = q_rubik.w#0.0#q_rubik[3]
            # uncomment this if we want to do like servoing
            _pub_target.publish(pose_target)

        cv2.imshow("Image window", cv2_img)
        cv2.waitKey(3)


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn

def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs

def make_env():
    env = gym.make('MARAVisionOrient-v0')
    env.init_time(slowness= args.slowness, slowness_unit=args.slowness_unit, reset_jnts=args.reset_jnts)
    logdir = '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo2/' + str(args.slowness) + '_' + str(args.slowness_unit) + '/'
    logger.configure(os.path.abspath(logdir))
    print("logger.get_dir(): ", logger.get_dir() and os.path.join(logger.get_dir()))
    # env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True)
    # env.render()
    return env

# parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slowness', help='time for executing trajectory', type=int, default=1)
parser.add_argument('--slowness-unit', help='slowness unit',type=str, default='sec')
parser.add_argument('--reset-jnts', help='reset the enviroment',type=bool, default=True)
args = parser.parse_args()

# arg_parser = common_arg_parser()
# args, unknown_args = arg_parser.parse_known_args()
# extra_args = {k: parse(v) for k,v in parse_unknown_args(unknown_args).items()}

"""
ROS and Vision stuff goes here
"""

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


options = {"pbLoad": "/home/rkojcev/devel/darkflow/built_graph/yolo-new.pb", "metaLoad": "/home/rkojcev/devel/darkflow/built_graph/yolo-new.meta", "threshold": 0.02, "gpu": 1.00}
tfnet = TFNet(options)

bridge = CvBridge()
TARGET_PUBLISHER = '/mara/target'
# Read intrinsic camera parameters
internal_calibration = get_camera_intrinsic()
_sub_image = rospy.Subscriber("/mara/rgb/image_raw", ImageMsg, _observation_image_callback)
_pub_target = rospy.Publisher(TARGET_PUBLISHER, Pose)


ncpu = multiprocessing.cpu_count()
if sys.platform == 'darwin': ncpu //= 2

print("ncpu: ", ncpu)
# ncpu = 1
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu,
                        log_device_placement=False)
config.gpu_options.allow_growth = True #pylint: disable=E1101

tf.Session(config=config).__enter__()

nenvs = 1
# env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
env = DummyVecEnv([make_env])
env = VecNormalize(env)
alg='ppo2'
env_type = 'mujoco'
learn = get_learn_function('ppo2')
alg_kwargs = get_learn_function_defaults('ppo2', env_type)
# alg_kwargs.update(extra_args)

seed = 0
set_global_seeds(seed)
network = 'mlp'
alg_kwargs['network'] = 'mlp'
rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

save_path =  '/tmp/rosrl/' + str(env.__class__.__name__) +'/ppo2/'

model, _ = learn(env=env,
    seed=seed,
    total_timesteps=1e8, save_interval=10, **alg_kwargs) #, outdir=logger.get_dir()

if save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)
