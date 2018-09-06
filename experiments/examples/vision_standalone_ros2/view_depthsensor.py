#!/usr/bin/python3
from darkflow.utils.utils import *
from darkflow.net.build import TFNet

import rclpy

from hrim_sensor_camera_msgs.msg import Image
from geometry_msgs.msg import Pose
from rclpy.qos import qos_profile_default, qos_profile_sensor_data

import cv2
import numpy as np
import os
import struct
from rclpy.executors import MultiThreadedExecutor
import time
import yaml
import quaternion as quat

from transforms3d.euler import euler2mat, mat2euler


def depth_callback(msg):
    global img_depth
    global depth_bool
    data_bytes = np.array(msg.data, dtype=np.uint8)
    img_depth = data_bytes.view(dtype=np.float32)
    img_depth = img_depth.reshape(msg.height, msg.width, 1)
    depth_bool = True




def color_callback(msg):
    global img
    global img_bool
    img = np.asarray(msg.data, dtype=np.uint8)
    img = img.reshape(msg.height, msg.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_bool = True

    result_max = 0.
    result_max_iter = 0

    result = tfnet.return_predict(img)

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


        cv2.circle(img,(int(result[result_max_iter]['point_1']['x']),int(result[result_max_iter]['point_1']['y'])), 4, (255,255,0), -1)
        # cv2.putText(rgb_image,'1',(int(result[result_max_iter]['point_1']['x']),int(result[result_max_iter]['point_1']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

        cv2.circle(img,(int(result[result_max_iter]['point_2']['x']),int(result[result_max_iter]['point_2']['y'])), 4, (255,255,0), -1)
        # cv2.putText(rgb_image,'2',(int(result[result_max_iter]['point_2']['x']),int(result[result_max_iter]['point_2']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

        cv2.circle(img,(int(result[result_max_iter]['point_3']['x']),int(result[result_max_iter]['point_3']['y'])), 4, (255,255,0), -1)
        # cv2.putText(rgb_image,'3',(int(result[result_max_iter]['point_3']['x']),int(result[result_max_iter]['point_3']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

        cv2.circle(img,(int(result[result_max_iter]['point_4']['x']),int(result[result_max_iter]['point_4']['y'])), 4, (255,255,0), -1)
        # cv2.putText(rgb_image,'4',(int(result[result_max_iter]['point_4']['x']),int(result[result_max_iter]['point_4']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

        cv2.circle(img,(int(result[result_max_iter]['point_5']['x']),int(result[result_max_iter]['point_5']['y'])), 4, (255,255,0), -1)
        # cv2.putText(rgb_image,'5',(int(result[result_max_iter]['point_5']['x']),int(result[result_max_iter]['point_5']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

        cv2.circle(img,(int(result[result_max_iter]['point_6']['x']),int(result[result_max_iter]['point_6']['y'])), 4, (255,255,0), -1)
        # cv2.putText(rgb_image,'6',(int(result[result_max_iter]['point_6']['x']),int(result[result_max_iter]['point_6']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

        cv2.circle(img,(int(result[result_max_iter]['point_7']['x']),int(result[result_max_iter]['point_7']['y'])), 4, (255,255,0), -1)
        # cv2.putText(rgb_image,'7',(int(result[result_max_iter]['point_7']['x']),int(result[result_max_iter]['point_7']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)

        cv2.circle(img,(int(result[result_max_iter]['point_8']['x']),int(result[result_max_iter]['point_8']['y'])), 4, (255,255,0), -1)
        # cv2.putText(rgb_image,'8',(int(result[result_max_iter]['point_8']['x']),int(result[result_max_iter]['point_8']['y'])), font, 0.8,(0,255,0),2,cv2.LINE_AA)
        cv2.circle(img,(int(result[result_max_iter]['point_9']['x']),int(result[result_max_iter]['point_9']['y'])), 4, (0,0,255), -1)
        # cv2.putText(imgcv,'center',(int(result[0]['point_9']['x']),int(result[0]['point_9']['y'])), font, 0.8,(255,255,255),2,cv2.LINE_AA)
        # cv2.imshow("depth camera", depth_map)
        # cv2.putText(rgb_image,result[result_max_iter]['label'],(int(result[result_max_iter]['point_1']['x']-30),int(result[result_max_iter]['point_1']['y']-30)), font, 1.0,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img,label_human_readable,(int(result[result_max_iter]['point_1']['x']-30),int(result[result_max_iter]['point_1']['y']-30)), font, 0.8,(255,255,255),2,cv2.LINE_AA)

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

        cv2.line(img,corner_2D_1_pred, corner_2D_2_pred, (0,255,0))
        cv2.line(img,corner_2D_2_pred, corner_2D_6_pred, (0,255,0))
        cv2.line(img,corner_2D_5_pred, corner_2D_6_pred, (0,255,0))
        cv2.line(img,corner_2D_2_pred, corner_2D_4_pred, (0,255,0))
        cv2.line(img,corner_2D_6_pred, corner_2D_8_pred, (0,255,0))
        cv2.line(img,corner_2D_5_pred, corner_2D_7_pred, (0,255,0))
        cv2.line(img,corner_2D_1_pred, corner_2D_3_pred, (0,255,0))
        cv2.line(img,corner_2D_3_pred, corner_2D_4_pred, (0,255,0))
        cv2.line(img,corner_2D_4_pred, corner_2D_8_pred, (0,255,0))
        cv2.line(img,corner_2D_7_pred, corner_2D_8_pred, (0,255,0))
        cv2.line(img,corner_2D_3_pred, corner_2D_7_pred, (0,255,0))
        cv2.line(img,corner_2D_1_pred, corner_2D_5_pred, (0,255,0))

        objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32')
        K = np.array(internal_calibration, dtype='float32')

        R_pred, t_pred = pnp(corners3D,  corners_2D_pred, K)

        # print("t_pred before div: \n", t_pred)

        # print("R_pred: \n", R_pred)
        # print("t_pred after div: \n", t_pred)



        Rt_pred = np.concatenate((R_pred, t_pred), axis=1)

        R_transform = Rt_pred * Rt_cam

        # print("R_transform: \n",R_transform)

        # pose_target = Pose()
        # pose_target.position.x = float(t_pred[0] + cam_pose_x)
        # pose_target.position.y = float(t_pred[1] - cam_pose_y)
        # pose_target.position.z = float(t_pred[2] + cam_pose_z)
        t_pred[2] = t_pred[2]/3

        if t_pred[2] > 0.0:
            t_pred[2] = -t_pred[2]

        pose_target = Pose()
        pose_target.position.x = float(-t_pred[0])
        pose_target.position.y = float(t_pred[1])
        pose_target.position.z = float(t_pred[2] + cam_pose_z)

        q_rubik = quat.from_rotation_matrix(R_pred)
        # print("q_rubik: ", q_rubik.x, q_rubik.y, q_r
        pose_target.orientation.x = q_rubik.x#0.0#q_rubik[0]
        pose_target.orientation.y = q_rubik.y#0.0#q_rubik[1]
        pose_target.orientation.z = q_rubik.z#0.0#q_rubik[2]
        pose_target.orientation.w = q_rubik.w#0.0#q_rubik[3]
        # uncomment this if we want to do like servoing

        # print("Rt_pred: \n", Rt_pred)
        print("Pose in world: ", pose_target)
        publisher_pose.publish(pose_target)

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('hros_sensing_depthsensor_image_visualizer')

    subscription_depth = node.create_subscription(Image, '/hros_sensing_depthsensor_00FA35000222/depth', depth_callback,  qos_profile=qos_profile_sensor_data)
    subscription_color = node.create_subscription(Image, '/hros_sensing_depthsensor_00FA35000222/rgb/image_raw', color_callback,  qos_profile=qos_profile_sensor_data)


    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)

    global depth_bool
    global img_bool
    depth_bool = False
    img_bool = False

    global img_depth
    global img
    global tfnet
    global models_info
    global internal_calibration
    global cam_pose_x
    global cam_pose_y
    global cam_pose_z
    global publisher_pose

    global Rt_cam

    internal_calibration = get_camera_intrinsic()

    publisher_pose = node.create_publisher(Pose, '/mara/target')

    cam_pose_x = -0.461956 #+ 0.0488#-0.5087683179567231 # random.uniform(-0.25, -0.6)#-0.5087683179567231#0.0 #random.uniform(-0.25, -0.6)#-0.5087683179567231#random.uniform(-0.3, -0.6)#random.uniform(-0.25, -0.6) # -0.5087683179567231#
    cam_pose_y = 0.0095#-0.040128#-0.013376#random.uniform(0.0, -0.2)
    cam_pose_z = 1.33626#+0.0488 #1.4808068867058566

    cam_orientation_x = -0.707099
    cam_orientation_y =  0.707101
    cam_orientation_z = -0.00269089
    cam_orientation_w =  0.00325398


    t_cam = np.asarray([[cam_pose_x], [cam_pose_y], [cam_pose_z]])

    cam_x_angle = 0.0
    cam_y_angle = np.pi / 2
    cam_z_angle = np.pi
    R_cam = euler2mat(cam_x_angle, cam_y_angle, cam_z_angle, 'sxyz')

    # print("t_cam: \n", t_cam)
    # print("R_cam: \n", R_cam)

    Rt_cam = np.concatenate((R_cam, t_cam), axis=1)

    # print("Rt_cam: ", Rt_cam)


    dumps = list()

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

    while rclpy.ok():
       executor.spin_once()

       if(depth_bool):
         cv2.imshow("Image window depth", img_depth)
       if(img_bool):
         cv2.imshow("Image window color", img)
       cv2.waitKey(20)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
