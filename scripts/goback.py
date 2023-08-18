#!/usr/bin/env python3

from controller import RobotController
import rospy
from moveit_commander.conversions import pose_to_list, list_to_pose

# def goback()

if __name__ == "__main__":
    rospy.init_node('track_marker')
    controller = RobotController()
    controller.manual_goto(list_to_pose([0.07490885255157444,
                                        -0.31445060110240575,
                                        0.8371192029122763,
                                        0.5479146916685392,
                                        -0.749988018752476,
                                        0.2851576892997476,
                                        0.2366274595534801]))
    # controller.get_goal_pose_array('/camera/goal_pose_array')
    # result = controller.goto_pose_array()
    # rospy.loginfo(result)
