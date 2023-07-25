#!/bin/bash

MAIN=$$

#conda init bash
VAL=NO
while [ $VAL != "y" ]
do
   source /opt/ros/noetic/setup.bash && \
   source ~/catkin_ws/devel/setup.bash && \
   roscore &
  read VAL
done

VAL=NO
while [ $VAL != "y" ]
do
    (
    source /opt/ros/noetic/setup.bash && \
    source ~/catkin_ws/devel/setup.bash && \
    roslaunch ur_gazebo ur5e_bringup.launch) &
  read VAL
done

VAL=NO
while [ $VAL != "y" ]
do
    (
    source /opt/ros/noetic/setup.bash && \
    source ~/catkin_ws/devel/setup.bash && \
    roslaunch ur5e_moveit_config moveit_planning_execution.launch sim:=true) &
  read VAL
done

VAL=NO
while [ $VAL != "y" ]
do
    (
    source /opt/ros/noetic/setup.bash && \
    source ~/catkin_ws/devel/setup.bash && \
    roslaunch ur5e_moveit_config moveit_rviz.launch rviz_config:=$(rospack find ur5e_moveit_config)/launch_moveit.rviz) &
  read VAL
done

trap "pkill roslaunch; pkill roscore" EXIT

wait
