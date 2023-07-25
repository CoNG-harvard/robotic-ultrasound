#! /usr/bin/env python3
from __future__ import print_function

import moveit_commander
from geometry_msgs.msg import PoseStamped
from eye_tracking_server.msg import GoToPoseAction, GoToPoseGoal
from collections import deque

import numpy as np
import rospy

import actionlib
from copy import deepcopy

import tf

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

from moveit_commander.conversions import pose_to_list

# Modules required by the get_key() function, used in the manual mode.
import os
import select
import sys
import termios
import tty


## END_SUB_TUTORIAL

def get_key(settings):
  tty.setraw(sys.stdin.fileno())
  rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
  if rlist:
    key = sys.stdin.read(1)
  else:
    key = ''

  termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
  return key

class AcquisitionControl(object):
    """AcquisitionControl"""

    def __init__(self):
        super(AcquisitionControl
, self).__init__()

        robot = moveit_commander.RobotCommander()

        group_name = "manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        print("============ Printing end factor pose")
        print(move_group.get_current_pose().pose)
        print("")

        # Misc variables
        self.box_name = "tablet"
        self.robot = robot
        self.move_group = move_group
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        self.scene = moveit_commander.PlanningSceneInterface()
        self.robot = moveit_commander.RobotCommander()


class RobotController(object):
    def __init__(self):
        self.i = 0
        self.client = actionlib.SimpleActionClient('GoTo', GoToPoseAction)
        self.client.wait_for_server()
        rospy.loginfo("Action server ready")

        self.curve = None
        self.newMats = None

        self.motion_planner = 'bezier' # bezier or moveit

        self.tf = tf.TransformListener(True, rospy.Duration(10.))
        self.goal_pose_listener = rospy.Subscriber('camera/goal_pose', PoseStamped)
        

        self.tf.waitForTransform('tool0', 'base', time=rospy.Time(), timeout=rospy.Duration(10.))
        print('all frames', self.tf.getFrameStrings())

        self.goal_pose_stack = deque(maxlen=10)

    def goal_pose_cbk(self, data):
        self.goal_pose_stack.append(data)

    def get_tool_pose(self):
        (translation, quaternion) = self.tf.lookupTransform('tool0', 'base_link', rospy.Time())
        return translation, quaternion

    def goto(self):
        if self.curve is None:
            pass 
        else: 
            goal_pos, goal_rotmat = self.curve_base[self.i], self.newMats_base[self.i]
            rospy.loginfo('get goal: %s' % goal_pos)
            rospy.loginfo('current pose %s' %self.get_tool_pose()[0] )
            r = R.from_matrix(goal_rotmat)
            orientation = r.as_quat()
            rospy.loginfo('goal quat %s' % orientation)
            goal = GoToPoseGoal()
            goal.position_x = goal_pos[0]
            goal.position_y = goal_pos[1]
            goal.position_z = goal_pos[2]
            goal.orientation_x = orientation[0]
            goal.orientation_y = orientation[1]
            goal.orientation_z = orientation[2]
            goal.orientation_w = orientation[3]
            self.client.send_goal(goal)
            self.client.wait_for_result()
            move_result = self.client.get_result()
            rospy.loginfo('move result "%s"' % move_result)
            # if move_result.waypoint_reached:
            self.i += 1

    def manual_goto(self, pos, quat):
        rospy.loginfo('manual goal set %s' % pos)
        goal = GoToPoseGoal()
        goal.position_x = pos[0]
        goal.position_y = pos[1]
        goal.position_z = pos[2]
        goal.orientation_x = quat[0]
        goal.orientation_y = quat[1]
        goal.orientation_z = quat[2]
        goal.orientation_w = quat[3]
        self.client.send_goal(goal)
        self.client.wait_for_result()
        move_result = self.client.get_result()
        rospy.loginfo('move result "%s"' % move_result)
        return move_result.waypoint_reached
        
    def is_moving(self):
        joint_states = rospy.wait_for_message('joint_states')
        velo = joint_states.velocity
        if np.max(np.abs(velo)) > 1e-5:
            return False
        else:
            return True
        
    def goto_tool_frame(self, pos, quat):
        goal = PoseStamped()
        goal.header.seq = 1
        # Need to get latest common time according to transformpose documentation
        goal.header.stamp = self.tf.getLatestCommonTime('tool0', 'base_link') 
        goal.header.frame_id = "tool0"

        goal.pose.position.x = pos[0]
        goal.pose.position.y = pos[1]
        goal.pose.position.z = pos[2]

        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]
        goal_pose_base = self.tf.transformPose('base_link', goal)
        goal_pos_base = [goal_pose_base.pose.position.x,
                        goal_pose_base.pose.position.y,
                        goal_pose_base.pose.position.z]
        goal_quat_base = [goal_pose_base.pose.orientation.x,
                        goal_pose_base.pose.orientation.y,
                        goal_pose_base.pose.orientation.z,
                        goal_pose_base.pose.orientation.w]
        result = self.manual_goto(goal_pos_base, goal_quat_base)

        return result

if __name__ == '__main__':
    rospy.init_node('controller_test', disable_signals=True)
    controller = RobotController()
    pos = [0., 0., 0.05]
    quat = [0., 0., 0., 1.,]
    def demo_control_goto(pos, quat):
        goal = PoseStamped()

        goal.header.seq = 1
        goal.header.stamp = controller.tf.getLatestCommonTime('tool0', 'base_link')
        goal.header.frame_id = "tool0"

        goal.pose.position.x = pos[0]
        goal.pose.position.y = pos[1]
        goal.pose.position.z = pos[2]

        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]
        # print(controller.tf.getFrameStrings())
        print(goal)
        goal_pose_base = controller.tf.transformPose('base_link', goal)
        goal_pos_base = [goal_pose_base.pose.position.x,
                        goal_pose_base.pose.position.y,
                        goal_pose_base.pose.position.z]
        goal_quat_base = [goal_pose_base.pose.orientation.x,
                        goal_pose_base.pose.orientation.y,
                        goal_pose_base.pose.orientation.z,
                        goal_pose_base.pose.orientation.w]
        controller.manual_goto(goal_pos_base, goal_quat_base)
    print("press w/a/s/d for moving the tool, and press i/j/k/l to tilt the tool")
    while not rospy.is_shutdown():
        try:
            key = get_key(termios.tcgetattr(sys.stdin))
            if key == 'w':
                demo_control_goto([0.05, 0., 0.], [0., 0., 0., 1.,])
            elif key == 's':
                demo_control_goto([-0.05, 0., 0.], [0., 0., 0., 1.,])
            elif key == 'a':
                demo_control_goto([0., 0.05, 0.], [0., 0., 0., 1.,])
            elif key == 'd':
                demo_control_goto([0., -0.05, 0.], [0., 0., 0., 1.,])
            elif key == 'i':
                quat = tf.transformations.quaternion_about_axis(10./180.*np.pi, (1,0,0))
                demo_control_goto([0., 0., 0.], quat)
            elif key == 'k':
                quat = tf.transformations.quaternion_about_axis(-10./180.*np.pi, (1,0,0))
                demo_control_goto([0., 0., 0.], quat)
            elif key == 'j':
                quat = tf.transformations.quaternion_about_axis(10./180.*np.pi, (0,1,0))
                demo_control_goto([0., 0., 0.], quat)
            elif key == 'l':
                quat = tf.transformations.quaternion_about_axis(-10./180.*np.pi, (0,1,0))
                demo_control_goto([0., 0., 0.], quat)
            elif key == '\x03':
                break

        except KeyboardInterrupt:
            break
            