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
    roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.1.13 target_filename:="${HOME}/my_robot_calibration.yaml" )&
  read VAL
done

VAL=NO
while [ $VAL != "y" ]
do
    (
    source /opt/ros/noetic/setup.bash && \
    source ~/catkin_ws/devel/setup.bash && \
    roslaunch ur5e_moveit_config moveit_planning_execution.launch) &
  read VAL
done

trap "pkill roslaunch; pkill roscore" EXIT

wait
