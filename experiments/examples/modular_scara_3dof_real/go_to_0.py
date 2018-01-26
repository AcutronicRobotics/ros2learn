import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing scara joint angles.
from control_msgs.msg import JointTrajectoryControllerState

import time

class Talker(Node):


    def _observation_callback(self, message):
        self._observation_msg = message

    def __init__(self):
        super().__init__('talker')

        JOINT_PUBLISHER = '/scara_controller/commandMotor'
        JOINT_SUBSCRIBER = '/scara_controller/stateMotor'
        self._pub = self.create_publisher(JointTrajectory,
                                               JOINT_PUBLISHER,
                                               qos_profile=qos_profile_sensor_data)
        self._sub = self.create_subscription(JointTrajectoryControllerState,
                                                  JOINT_SUBSCRIBER,
                                                  self._observation_callback,
                                                  qos_profile=qos_profile_sensor_data)
        self.i = 0
        timer_period = 0.02
        self.tmr = self.create_timer(timer_period, self.timer_callback)

        first = True;

        self.last_time = time.time()
        self._observation_msg = None

        self.goal_vel_value = 0.1
        self.goal_vel1 = self.goal_vel_value
        self.goal_vel2 = self.goal_vel_value
        self.goal_vel3 = self.goal_vel_value
        self.goal_vel4 = self.goal_vel_value

        self.first = False


    def timer_callback(self):

        dt = time.time() - self.last_time

        if(self._observation_msg==None):
            return;

        if not self.first:
            self.goal_cmd1 = self._observation_msg.actual.positions[0]
            self.goal_cmd2 = self._observation_msg.actual.positions[1]
            self.goal_cmd3 = self._observation_msg.actual.positions[2]
            # self.goal_cmd4 = self._observation_msg.actual.positions[3]
            self.first = True
            return

        # Set constants for joints
        MOTOR1_JOINT = 'motor1'
        MOTOR2_JOINT = 'motor2'
        MOTOR3_JOINT = 'motor3'
        # MOTOR4_JOINT = 'motor4'
        JOINT_ORDER = [MOTOR1_JOINT, MOTOR2_JOINT, MOTOR3_JOINT]

        if(self._observation_msg.actual.positions[0] > 0):
            self.goal_vel1 = -self.goal_vel_value
        else:
            self.goal_vel1 = self.goal_vel_value
        self.goal_cmd1 += dt*self.goal_vel1

        if(self._observation_msg.actual.positions[1] > 0):
            self.goal_vel2 = -self.goal_vel_value
        else:
            self.goal_vel2 = self.goal_vel_value
        self.goal_cmd2 += dt*self.goal_vel2

        if(self._observation_msg.actual.positions[2] > 0):
            self.goal_vel3 = -self.goal_vel_value
        else:
            self.goal_vel3 = self.goal_vel_value
        self.goal_cmd3 += dt*self.goal_vel3

        # if(self._observation_msg.actual.positions[3] > 0):
        #     self.goal_vel4 = -self.goal_vel_value
        # else:
        #     self.goal_vel4 = self.goal_vel_value
        #
        # self.goal_cmd4 += (dt)*self.goal_vel4

        # Set up a trajectory message to publish.
        action_msg = JointTrajectory()
        action_msg.joint_names = JOINT_ORDER

        # Create a point to tell the robot to move to.
        target = JointTrajectoryPoint()
        target.positions  = [self.goal_cmd1,
                             self.goal_cmd2,
                             self.goal_cmd3]

        target.velocities = [self.goal_vel_value]*3
        target.effort = [float('nan')]*3

        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]

        self._pub.publish(action_msg)
        self.last_time = time.time()


def main(args=None):
    if args is None:
        args = sys.argv

    rclpy.init(args=args)

    node = Talker()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
