#!/usr/bin/env python

# Modules required by the get_key() function, used in the manual mode.
import os
import sys
import time
PKG_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PKG_ROOT_DIR)
IMAGE_DIR = os.path.join(PKG_ROOT_DIR, 'image')

import rospy
from geometry_msgs.msg import WrenchStamped, Pose
from moveit_commander.conversions import pose_to_list
from sftp import Sftp_Helper
from move_group_python_interface_tutorial import all_close

import cv2
import tf
from math import pi

ACTION_DICT = {
	'w': ([0.05, 0., 0.], [0., 0., 0., 1.,]),
	's': ([-0.05, 0., 0.], [0., 0., 0., 1.,]),
	'a': ([0., 0.05, 0.], [0., 0., 0., 1.,]),
	'd': ([0., -0.05, 0.], [0., 0., 0., 1.,]),
	'i': ([0.,0.,0.], tf.transformations.quaternion_about_axis(10./180. * pi, (1,0,0))),
	'k': ([0.,0.,0.], tf.transformations.quaternion_about_axis(-10./180. * pi, (1,0,0))),
	'j': ([0.,0.,0.], tf.transformations.quaternion_about_axis(10./180. * pi, (0,1,0))),
	'l': ([0.,0.,0.], tf.transformations.quaternion_about_axis(-10./180. * pi, (0,1,0)))

}


class WrenchSubscriber(object):

	def __init__(self) -> None:
		rospy.init_node('recorder')
		# self.sub = rospy.Subscriber('/wrench', WrenchStamped, self.force_callback)
		data = rospy.wait_for_message('/wrench', WrenchStamped, timeout=5.)
		print(data)

	def force_callback(self, data):
		rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.wrench.force)
		self.sub.unregister()

	def run(self):
		pass

def get_force():
	'''
	The data format of returned value (force and torque sensor reading) is 
	force: 
		x: -0.1924162504227007
		y: -0.6463650937001096
		z: 4.222756959216565
	torque: 
		x: -0.017902787403797825
		y: -0.003747197498168533
		z: 0.03403793105906066

	could be accessed like data.wrench.force.x , which is a float.
		
	'''
	data = rospy.wait_for_message('/wrench', WrenchStamped, timeout=5.)
	return data.wrench

def get_pose(controller):
	'''
	The data format of returned value (robot end effector pose) is
	position: 
		x: -0.391241410927265
		y: -0.052096536446966435
		z: 0.6529468999016157
	orientation: 
		x: -0.36574057519038083
		y: -0.23906563620214916
		z: -0.043007832959553904
		w: 0.8984607835352605

	could be accessed like pose.position.x , which is a float.

	'''

	pose = controller.get_tool_pose()
	return pose

def get_us_image(vid):
	# Capture the video frame
	# by frame
	ret, frame = vid.read()

def save_us_image_and_upload(sftp_helper, image, index):
	file_name = str(index) + '.png'
	cv2.imwrite(image, os.path.join(IMAGE_DIR, file_name))
	sftp_helper.Transfer_data(source_path = os.path.join(IMAGE_DIR, file_name), dest_path = '/home/local/PARTNERS/sk1064/workspace/test.png' )

def robot_move(controller, action):
	'''
	to be added
	'''
	(pos, quat) = ACTION_DICT.get(action, (None, None))
	if pos is not None:
		result = controller.goto_tool_frame(pos,quat)
	
	return result


def main():
	rospy.init_node('recording_upload')
	
	# define a video capture object
	# vid = cv2.VideoCapture('/dev/video5')

	# Define the robot pose listener
	from controller import RobotController
	controller = RobotController()

	# initialize sftp helper
	# sftp_helper = Sftp_Helper(host = 'emimdgxa100gpu3.ccds.io')
	i = 0
	force = get_force()
	print(force)
	pose = get_pose(controller)
	print(pose)
	# image = get_us_image(vid)
	robot_move(controller, 'w')
	# save_us_image_and_upload(sftp_helper, image, i)

			
if __name__ == '__main__':
	main()
