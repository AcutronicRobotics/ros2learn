import rospy
from control_msgs.msg import FollowJointTrajectoryActionFeedback, FollowJointTrajectoryActionGoal
import os
import numpy as np
import time
import _thread

import copy
import rospkg
from baselines.agent.scara_arm.tree_urdf import treeFromFile # For KDL Jacobians
from PyKDL import Jacobian, Chain, ChainJntToJacSolver, JntArray # For KDL Jacobians
from baselines.agent.utility.general_utils import forward_kinematics, get_ee_points, rotation_from_matrix, \
    get_rotation_matrix,quaternion_from_matrix # For getting points and velocities.


URDF_PATH = rospkg.RosPack().get_path("mara_description") + "/urdf/mara_demo_camera_top.urdf"
_, ur_tree = treeFromFile(URDF_PATH)
LINK_NAMES = ['table', 'base_link', 'motor1_link', 'motor2_link',
                'motor3_link', 'motor4_link', 'motor5_link', 'motor6_link',
              'ee_link']
m_link_names = copy.deepcopy(LINK_NAMES)
scara_chain = ur_tree.getChain(m_link_names[0], m_link_names[-1])
EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
EE_POINTS = np.asmatrix([[0, 0, 0]])
# EE_POS_TGT = np.asmatrix([-0.421772, -0.00098445, 0.909351]) # down - home position
# EE_POS_TGT = np.asmatrix([0.000002, -0.001009, 1.34981]) # bend
# EE_POS_TGT = np.asmatrix([-0.173762, -0.0124312, 1.60415]) #easy point
# EE_POS_TGT = np.asmatrix([-0.40071, -0.0010402, 1.1485]) #new
# EE_POS_TGT = np.asmatrix([-0.40068, -0.00099772, 1.1486]) #new2
EE_POS_TGT = np.asmatrix([-0.46598, -0.0010029, 0.80828]) #bc_down
# EE_POS_TGT = np.asmatrix([-0.471771, -0.100948, 1.2094]) # 10_595 10_1189 ppo workspace
realgoal = np.ndarray.flatten(get_ee_points(EE_POINTS, EE_POS_TGT, EE_ROT_TGT).T)



observations = []
actions = []
obs_traj = []
acs_traj = []
i = 0
yes = 0
times = 0

def get_jacobians(state):
    jacobian = Jacobian(6)
    angles = JntArray(6)
    for i in range(6):
        angles[i] = state[i]
    jac_solver = ChainJntToJacSolver(scara_chain)
    jac_solver.JntToJac(angles, jacobian)
    J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(jacobian.rows())])
    ee_jacobians = J
    return ee_jacobians

def get_ee_points_jacobians(ref_jacobian, ee_points, ref_rot):
    ee_points = np.asarray(ee_points)
    ref_jacobians_trans = ref_jacobian[:3, :]
    ref_jacobians_rot = ref_jacobian[3:, :]
    end_effector_points_rot = np.expand_dims(ref_rot.dot(ee_points.T).T, axis=1)
    ee_points_jac_trans = np.tile(ref_jacobians_trans, (ee_points.shape[0], 1)) + \
                                    np.cross(ref_jacobians_rot.T, end_effector_points_rot).transpose(
                                        (0, 2, 1)).reshape(-1, 6)
    ee_points_jac_rot = np.tile(ref_jacobians_rot, (ee_points.shape[0], 1))
    return ee_points_jac_trans, ee_points_jac_rot

def get_ee_points_velocities(ref_jacobian, ee_points, ref_rot, joint_velocities):
    ref_jacobians_trans = ref_jacobian[:3, :]
    ref_jacobians_rot = ref_jacobian[3:, :]
    ee_velocities_trans = np.dot(ref_jacobians_trans, joint_velocities)
    ee_velocities_rot = np.dot(ref_jacobians_rot, joint_velocities)
    ee_velocities = ee_velocities_trans + np.cross(ee_velocities_rot.reshape(1, 3),
                                                   ref_rot.dot(ee_points.T).T)
    return ee_velocities.reshape(-1)

def obs_space(position):
    ee_link_jacobians = get_jacobians(position)

    trans, rot = forward_kinematics(scara_chain,
                                m_link_names,
                                position[:6],
                                base_link=m_link_names[0],
                                end_link=m_link_names[-1])
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rot
    rotation_matrix[:3, 3] = trans

    current_ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS,
                                                      trans,
                                                      rot).T)
    ee_points = current_ee_tgt - realgoal #self.environment['ee_points_tgt']
    ee_points_jac_trans, _ = get_ee_points_jacobians(ee_link_jacobians,
                                                           EE_POINTS,
                                                           rot)
    ee_velocities = get_ee_points_velocities(ee_link_jacobians,
                                                   EE_POINTS,
                                                   rot,
                                                   position)

    return np.r_[np.reshape(position, -1),
                  np.reshape(ee_points, -1),
                  np.reshape(ee_velocities, -1),]

def observation_callback(msg):
    global i
    global times
    times += 1
    pos = msg.feedback.actual.positions
    if len(actions) > i:
        # if actions[i][0] + 0.000001 > pos[0] > actions[i][0] - 0.000001: # down - home position
        # if actions[i][0] + 0.001 > pos[0] > actions[i][0] - 0.001: # ws - RRTConfigDefault
        if actions[i][0] + 0.00001 > pos[0] > actions[i][0] - 0.00001: #new/new2/bc_down -  RRTConfigDefault
            ob = obs_space(pos)
            observations.append(ob)
            i += 1

def actions_callback(msg):
    print(len(msg.goal.trajectory.points))
    for i in range(len(msg.goal.trajectory.points)):
        pos = msg.goal.trajectory.points[i].positions
        actions.append(pos)

def state_sub():
    rospy.init_node('state_sub', anonymous=True)
    rospy.Subscriber("/mara_controller/follow_joint_trajectory/feedback", FollowJointTrajectoryActionFeedback, observation_callback)
    rospy.Subscriber("/mara_controller/follow_joint_trajectory/goal", FollowJointTrajectoryActionGoal, actions_callback)
    rospy.spin()

def create_npz():
    while True:
        global yes
        global times
        print("times: ", times)
        if yes == 1:
            obs_traj.append(observations)
            acs_traj.append(actions)
            np.savez( os.path.join( "/home/yue/experiments/expert_data/mara/collisions_model", 'expert_data'), obs=(obs_traj), acs=(acs_traj) )
            e = np.load('/home/yue/experiments/expert_data/mara/collisions_model/expert_data.npz')
            print(len(e.f.obs[0]))
            print(len(e.f.acs[0]))
            print(e.f.obs)
            print(e.f.acs)
            print(e.f.obs.shape)
            print(e.f.acs.shape)
            break
            print("broken")
            time.sleep(40)
        else:
            yes += 1
            time.sleep(8)

if __name__ == '__main__':
    _thread.start_new_thread(create_npz, ())
    state_sub()
