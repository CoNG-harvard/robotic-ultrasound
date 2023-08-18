#! /usr/bin/env python3
from __future__ import print_function

import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from std_msgs.msg import Header
from eye_tracking_server.msg import GoToPoseAction, GoToPoseGoal
from collections import deque

import numpy as np
import rospy
import time

import actionlib
from copy import deepcopy
import time
import tf
import tf.transformations as tft

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

from moveit_commander.conversions import pose_to_list, list_to_pose, list_to_pose_stamped

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

        self.tf = tf.TransformListener(True, rospy.Duration(10.))
        self.goal_pose_listener = rospy.Subscriber('camera/goal_pose', PoseStamped)
        

        self.tf.waitForTransform('tool0', 'base_link', time=rospy.Time(), timeout=rospy.Duration(10.))
        rospy.loginfo('get all tf frames')
        rospy.loginfo(self.tf.getFrameStrings())

        self.goal_pose_stack = deque(maxlen=10)
        self.goal_pose_array = None

    # def goal_pose_cbk(self, data):
    #     self.goal_pose_stack.append(data)

    def hear_goal_pose_array(self, topic_name):
        """get goal pose array from ros topic topic_name.

        Args:
            topic_name (str): ros topic name to receive goal pose array.
        """
        self.goal_pose_array = rospy.wait_for_message(topic_name, PoseArray, timeout=30.)
    
    def goto_pose_array(self):
        """Go to a pose array received by self.hear_goal_pose_array.
        """
        if self.goal_pose_array is not None:
            goal_poses_base_link = self.transform_pose_array(self.goal_pose_array)
            for goal_pose in goal_poses_base_link:
                # print(goal_pose)
                # handle the camera bias. we need to let camera go to target rather than tool0.
                self.goal_manual_goto(goal_pose)
                # time.sleep(0.1)

    def show_pose_array(self, 
                        tool_bias=[0.0, 0.0, -0.172], 
                        tool_rotation=tft.quaternion_about_axis(-np.pi / 2, [0, 0, 1]).tolist(),
                        sleep_between_step=0.3):
        """show pose array in rviz.

        Args:
            tool_bias (list, optional): position from tool to flange. Defaults to [0.0, 0.0, -0.172].
            tool_rotation (list(4), quaternion, optional): rotation of flange based on tool. Defaults to tft.quaternion_about_axis(-np.pi / 2, [0, 0, 1]).tolist().
            sleep_between_step (float, optinal): sleep time between each step pose shown.
        """
        if self.goal_pose_array is not None:
            goal_pose_array_base_link = self.transform_pose_array(self.goal_pose_array)
            for goal_pose in goal_pose_array_base_link:
                if isinstance(goal_pose, PoseStamped):
                    goal_pose = goal_pose.pose
                goal_pose_list = pose_to_list(goal_pose)
                self.add_frame(goal_pose_list, 'goal')
                # self.tf_publisher.sendTransform(goal_pose_list[:3], goal_pose_list[3:],
                #                                 rospy.Time().now(), 'goal','base_link',)
                # self.tf.waitForTransform('goal', 'base_link', time=rospy.Time(), timeout=rospy.Duration(10.))
                matrix = tft.quaternion_matrix(goal_pose_list[3:])
                tool_matrix_camera = tft.quaternion_matrix(tool_rotation)
                # print(tool_rotation, goal_pose_list)
                tool_bias_base_link = np.dot(np.array(tool_bias), matrix[:3, :3])
                print(matrix)
                print(np.dot(np.array([tool_bias]), matrix[:3, :3]))
                for i in range(3):
                    goal_pose_list[i] = goal_pose_list[i] + tool_bias_base_link[i]

                tool_quaternion = tft.quaternion_from_matrix(np.dot(matrix, tool_matrix_camera))
                for j in range(4):
                    goal_pose_list[j+3] = tool_quaternion[j]

                tool_target_poselist_goal = list_to_pose_stamped(tool_bias+tool_rotation, 'goal')
                # tool_target_pose_goal.header.stamp = self.tf.getLatestCommonTime('goal', 'base_link')
                # tool_target_pose_base = self.tf.transformPose('base_link', tool_target_pose_goal)
                tool_target_pose_base = self.transform_pose(tool_target_poselist_goal)

                # for display tool target frame in rviz
                self.add_frame(pose_to_list(tool_target_pose_base.pose), 'tool_target')
                # self.tf_publisher.sendTransform(tool_target_pose_base[:3], tool_target_pose_base[3:],
                #                                 rospy.Time().now(), 'tool_target','base_link',)
                # self.tf.waitForTransform('tool_target', 'base_link', time=rospy.Time(), timeout=rospy.Duration(10.))
                # # self.tf_publisher.sendTransform([0.0, 0.0, -0.1], tft.quaternion_about_axis(-np.pi / 2, [0, 0, 1]),
                # #                                 rospy.Time().now(), 'tool_target','goal',)
                # # self.tf.waitForTransform('tool_target', 'goal', time=rospy.Time(), timeout=rospy.Duration(10.))
                rospy.loginfo('publish tool target')
                time.sleep(sleep_between_step)
                

    # ============================ high and low level goto =============================#
    
    def goal_manual_goto(self, goal_pose, tool_bias=[0.0, 0.0, -0.172], 
                         tool_rotation=tft.quaternion_about_axis(-np.pi / 2, [0, 0, 1]).tolist()):
        """Given goal position, determine the tool goal position and go to

        Args:
            goal_pose (Pose or PoseStamped): goal pose
            tool_bias (list, optional): tool_bias to camera frame.  Should be -1 * camera_bias.
                [0.0, 0.054, -0.046] for bracket.
                [0.0, 0.0, -0.172] for gripper. 
            tool_rotation (list, optional): tool rotation to camera frame. should be inv of camera rot mat.
        """
        if isinstance(goal_pose, PoseStamped):
            goal_pose = goal_pose.pose
        
        goal_pose_list = pose_to_list(goal_pose)
        self.tf_publisher.sendTransform(goal_pose_list[:3], goal_pose_list[3:],
                                        rospy.Time().now(), 'goal','base_link',)
        self.tf.waitForTransform('goal', 'base_link', time=rospy.Time(), timeout=rospy.Duration(10.))
        matrix = tft.quaternion_matrix(goal_pose_list[3:])
        tool_matrix_camera = tft.quaternion_matrix(tool_rotation)
        tool_bias_base_link = np.dot(np.array(tool_bias), matrix[:3, :3])
        tool_goal_pose = Pose()
        for i in range(3):
            goal_pose_list[i] = goal_pose_list[i] + tool_bias_base_link[i]

        tool_quaternion = tft.quaternion_from_matrix(np.dot(matrix, tool_matrix_camera))
        for j in range(4):
            goal_pose_list[j+3] = tool_quaternion[j]

        tool_target_pose_goal = list_to_pose_stamped(tool_bias+tool_rotation, 'goal')
        tool_target_pose_goal.header.stamp = self.tf.getLatestCommonTime('goal', 'base_link')
        tool_target_pose_base = self.tf.transformPose('base_link', tool_target_pose_goal)
        tool_target_pose_base = pose_to_list(tool_target_pose_base.pose)
        tool_target_pose_base[2] += 0.01
        self.manual_goto(list_to_pose(tool_target_pose_base))

    
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

    def manual_goto(self, pose: Pose or PoseStamped):
        """manually input goal pose (single) for the robot to go.

        Args:
            pose (geometry_msgs/Pose): goal pose.
        """
        if isinstance(pose, PoseStamped):
            pose = pose.pose
        rospy.loginfo('manual goal set %s' % pose)
        goal = GoToPoseGoal()
        goal.goal_poses.header.seq = 1
        goal.goal_poses.header.stamp = rospy.Time().now()
        goal.goal_poses.header.frame_id = 'base_link'
        goal.goal_poses.poses = [pose]
        self.client.send_goal(goal)
        self.client.wait_for_result()
        move_result = self.client.get_result()
        rospy.loginfo('move result "%s"' % move_result)
        
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
    
    # ==================== get or is state =====================#
    def is_moving(self):
        joint_states = rospy.wait_for_message('joint_states')
        velo = joint_states.velocity
        if np.max(np.abs(velo)) > 1e-5:
            return False
        else:
            return True
        
    def get_camera_pose(self):
        (translation, quaternion) = self.tf.lookupTransform('camera', 'base', rospy.Time())
        r = R.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        return translation, rotation_matrix
    
    # ==================== transform ===========================#
    def add_frame(self, pose: list, frame_name: str, target_frame: str='base_link'):
        """add frame by its pose to target frame.

        Args:
            pose (list(7)): pose of frame to added w.r.t. target_frame
            frame_name (str): name of frame to add
            target_frame (str, optional): Defaults to 'base_link'.
        """
        self.tf_publisher.sendTransform(pose[:3], pose[3:],
                                        rospy.Time().now(), 
                                        frame_name, 
                                        target_frame)
        self.tf.waitForTransform(frame_name, 
                                 target_frame, 
                                 time=rospy.Time(), 
                                 timeout=rospy.Duration(10.))
        

    def transform_pose_array(self, pose_array, target_frame='base_link'):
        """transform a pose array from current frame to target frame.
            the current frame is defined in the header of current pose.

        Args:
            pose_array (geometry_msgs/PoseArray): a pose array 
            target_frame (str, optional): target frame. Defaults to 'base_link'.

        Returns:
            Pose[]: A list of poses on target frame.
        """
        assert pose_array.header.frame_id != target_frame
        current_frame = pose_array.header.frame_id
        header = Header()
        header.seq = 1
        header.stamp = self.tf.getLatestCommonTime(current_frame, target_frame)
        header.frame_id = current_frame
        goal_poses = []
        for i in range(len(pose_array.poses)):
            pose_stamped = PoseStamped()
            pose_stamped.header = header
            pose_stamped.pose = pose_array.poses[i]
            goal_poses.append(self.tf.transformPose(target_frame, pose_stamped).pose)
        return goal_poses
    
    def transform_pose(self, pose: PoseStamped, target_frame : str = 'base_link') -> PoseStamped:
        """transform pose from current frame to target frame

        Args:
            pose (PoseStamped): pose to transform
            target_frame (str, optional): target frame. Defaults to 'base_link'.

        Returns:
            PoseStamped: pose on target frame.
        """
        current_frame = pose.header.frame_id
        pose.header.stamp = self.tf.getLatestCommonTime(current_frame, target_frame)
        return self.tf.transformPose(target_frame, pose)

if __name__ == '__main__':
    import rospy
    import tf
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
                quat = tft.quaternion_about_axis(10./180.*np.pi, (1,0,0))
                demo_control_goto([0., 0., 0.], quat)
            elif key == 'k':
                quat = tft.quaternion_about_axis(-10./180.*np.pi, (1,0,0))
                demo_control_goto([0., 0., 0.], quat)
            elif key == 'j':
                quat = tft.quaternion_about_axis(10./180.*np.pi, (0,1,0))
                demo_control_goto([0., 0., 0.], quat)
            elif key == 'l':
                quat = tft.quaternion_about_axis(-10./180.*np.pi, (0,1,0))
                demo_control_goto([0., 0., 0.], quat)
            elif key == '\x03':
                break

        except KeyboardInterrupt:
            break
            