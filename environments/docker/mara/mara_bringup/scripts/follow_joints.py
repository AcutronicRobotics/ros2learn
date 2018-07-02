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
        r = rospy.Rate(100)
        success = True

        print goal
        number_of_points_to_follow = len(goal.trajectory.points)
        goal_points = copy.deepcopy(goal.trajectory.points)
        joint_names =  copy.deepcopy(self.joints_state_msg.joint_names)

        for p in range(1, number_of_points_to_follow):
            while True:
                r.sleep()
                point_reached = True
                points_robot = list(copy.deepcopy(self.joints_state_msg.actual.positions))
                for i in range(6):
                    points_robot[i] = points_robot[i]*3.14/180
                for i in range(6):
                    p_robot = float("{0:.5f}".format(points_robot[i]))
                    p_goal = float("{0:.5f}".format(goal_points[p].positions[i]))
                    d = abs(p_robot - p_goal)
                    if p == (number_of_points_to_follow-1):
                        dist_objetivo = 3.14/180
                    else:
                        dist_objetivo = 3*3.14/180
                    if d > dist_objetivo:
                        point_reached = False
                if point_reached:
                    break;
                else:
                    msg = JointTrajectory()
                    msg.joint_names = joint_names;
                    point = JointTrajectoryPoint();
                    point.positions  = [0.0]*6
                    point.velocities = [0.0]*6
                    point.positions[0] = float("{0:.3f}".format(goal_points[p].positions[0]))
                    point.positions[1] = float("{0:.3f}".format(goal_points[p].positions[1]))
                    point.positions[2] = float("{0:.3f}".format(goal_points[p].positions[2]))
                    point.positions[3] = float("{0:.3f}".format(goal_points[p].positions[3]))
                    point.positions[4] = float("{0:.3f}".format(goal_points[p].positions[4]))
                    point.positions[5] = float("{0:.3f}".format(goal_points[p].positions[5]))
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
