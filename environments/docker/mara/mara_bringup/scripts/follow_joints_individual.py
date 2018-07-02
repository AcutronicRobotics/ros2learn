#! /usr/bin/env python

import rospy

import actionlib

import math
import copy

import control_msgs.msg
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class maraFollowJoint(object):
    # create messages that are used to publish feedback/result
    _feedback = control_msgs.msg.FollowJointTrajectoryFeedback()
    _result = control_msgs.msg.FollowJointTrajectoryResult()

    def jointStateCallback(self, msg):
        self.joints_state_msg = msg

    def __init__(self, name):
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, control_msgs.msg.FollowJointTrajectoryAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()
        self.state_subscriber = rospy.Subscriber("/mara_controller/state", JointTrajectoryControllerState, self.jointStateCallback)
        self.pub_commands = rospy.Publisher('/mara_controller/command', JointTrajectory, queue_size=1)

    def execute_cb(self, goal):
        pass
        # helper variables
        r = rospy.Rate(50)
        success = True

        print goal
        number_of_joints = 6
        number_of_points_to_follow = len(goal.trajectory.points)
        goal_points = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        for j in range(number_of_points_to_follow):
            for k in range(number_of_joints):
                goal_points[k].append(goal.trajectory.points[j].positions[k])

        all_finished = [0]*number_of_joints
        index_joints = [0]*number_of_joints
        joint_names =  copy.deepcopy(self.joints_state_msg.joint_names)

        while True:
            r.sleep()

            if(sum(all_finished) == 6 ):
                break;

            points_robot = list(copy.deepcopy(self.joints_state_msg.actual.positions))
            for i in range(6):
                points_robot[i] = points_robot[i]*3.14/180
            msg = JointTrajectory()
            msg.joint_names = joint_names;
            point = JointTrajectoryPoint();
            point.positions  = [0.0]*6
            point.velocities = [0.0]*6

            # print("number_of_points_to_follow ", number_of_points_to_follow)
            for i in range(number_of_joints):
                p_robot = float("{0:.5f}".format(points_robot[i]))
                p_goal = float("{0:.5f}".format(goal_points[i][index_joints[i]]))
                d = abs(p_robot - p_goal)
                if d < 5*3.14/180:
                    index_joints[i] = index_joints[i] + 1
                    if index_joints[i] >= (number_of_points_to_follow - 1):
                        index_joints[i] = (number_of_points_to_follow - 1)
                all_finished[i] = index_joints[i] == (number_of_points_to_follow - 1)
                point.positions[i] = p_goal
                # print str(i) + " " + str(index_joints[i]) + " | " + str(p_robot*180/3.1416)+ " " + str(p_goal*180/3.1416) + " " +  str(d*180/3.1416)
            # print("-----------------")

            msg.points.append(point)
            self.pub_commands.publish(msg)

        if success:
            # self._result.sequence = self._feedback.sequence
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)

if __name__ == '__main__':
    rospy.init_node('mara_follow_joint_trajectory')
    server = maraFollowJoint('follow_joint_trajectory')
    rospy.spin()
