#!/usr/bin/env python3

from controller import RobotController
import rospy
from moveit_commander.conversions import pose_to_list, list_to_pose

# def goback()

if __name__ == "__main__":
    rospy.init_node('track_marker')
    controller = RobotController()
    controller.hear_goal_pose_array('/camera/goal_pose_array')
    result = controller.goto_pose_array()
    rospy.loginfo(result)
