import numpy as np
import sys

import gym
import gym_gazebo

import tensorflow as tf

import argparse
import copy
import time

from baselines import logger
from baselines.common import set_global_seeds, tf_util as U

from baselines.acktr.acktr_cont import learn
from baselines.agent.utility.general_utils import get_ee_points, get_position
from baselines.ppo1 import mlp_policy, pposgd_simple

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
# import xml.etree.ElementTree as ET
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
        # Save your OpenCV2 image as a jpeg
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
        #     # cv2.putText(imgcv,'center',(int(result[0]['point_9']['x']),int(result[0]['point_9']['y'])), font, 0.8,(255,255,255),2,cv2.LINE_AA)
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
            cam_pose_y = 0.013376#random.uniform(0.0, -0.2)
            cam_pose_z = 1.4808068867058566

            pose_target = Pose()
            pose_target.position.x = -t_pred[0]/3.0 + cam_pose_x
            pose_target.position.y = -t_pred[1]/3.0 - cam_pose_y
            pose_target.position.z = -t_pred[2]/3.0 + cam_pose_z

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



env = gym.make('MARAVisionOrient-v0')
initial_observation = env.reset()
print("Initial observation: ", initial_observation)
# env.render()
seed = 0

sess = U.make_session(num_cpu=1)
sess.__enter__()
def policy_fn(name, ob_space, ac_space):
    return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    hid_size=128, num_hid_layers=3)
# gym.logger.setLevel(logging.WARN)
obs = env.reset()
print("Initial obs: ", obs)
# env.seed(seed)
# time.sleep(5)
pi = policy_fn('pi', env.observation_space, env.action_space)
tf.train.Saver().restore(sess, '/tmp/rosrl/GazeboMARATopOrientVisionv0Env/ppo1/1000000_nsec/models/mara_orient_ppo1_test_afterIter_1303.model') # for the H


dumps = list()
# points_3d = list()
cur_dir = os.getcwd()
#models info for now is hardcoded to a particular folder:
models_file = '/home/rkojcev/devel/darkflow/models_info/'
os.chdir(models_file)
annotations = sorted(os.listdir('.'))
for i, file in enumerate(annotations):
    print(i, file)
    if not os.path.isdir(file):
        print("annotations: ", file)
        models_file_path = file
        model_file = open(file)
        yaml_model=yaml.load(model_file)
        models_info = yaml_model
        annotations.remove(file)

print("models_info: ", models_info)


options = {"pbLoad": "/home/rkojcev/devel/darkflow/built_graph/yolo-new.pb", "metaLoad": "/home/rkojcev/devel/darkflow/built_graph/yolo-new.meta", "threshold": 0.02, "gpu": 1.00}
tfnet = TFNet(options)

bridge = CvBridge()
TARGET_PUBLISHER = '/mara/target'
# Read intrinsic camera parameters
internal_calibration = get_camera_intrinsic()
_sub_image = rospy.Subscriber("/mara/rgb/image_raw", ImageMsg, _observation_image_callback)
_pub_target = rospy.Publisher(TARGET_PUBLISHER, Pose)

done = False
while True:
    action = pi.act(False, obs)[0]
    obs, reward, done, info = env.step(action)
    # print(action)
