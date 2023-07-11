import time
import moveit_commander
import rospy
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list
from sftp import Sftp_Helper


import pickle as pkl


from pathlib import Path
import shutil

import cv2

# Modules required by the get_key() function, used in the manual mode.
import os
import sys


# For concurrent streaming and data saving
from threading import Thread, Event, Lock

# terminate = Event()
# data_lock = Lock()
path = os.path.dir(__file__)
sftp_helper = Sftp_Helper(host = 'emimdgxa100gpu3.ccds.io')


def catpure_video(vid, file_name):
	
	
def upload_via_sftp(file_path)
	sftp_helper.Transfer_data(source_path = os.path.join(path, 'capture.png'), dest_path = '/home/local/PARTNERS/sk1064/workspace/test.png' )


def main():	
	save_every_t = 1

	rospy.init_node('recorder')
	
	# define a video capture object
	vid = cv2.VideoCapture('/dev/video5')

	# Define the robot pose listener
	group_name = 'manipulator'
	
	# move_group = moveit_commander.MoveGroupCommander(group_name)
	# robot = moveit_commander.RobotCommander()

	i = 0

	try:
		# Capture the video frame
		# by frame
		ret, frame = vid.read()

		# pose = []
		# pose = pose_to_list(move_group.get_current_pose().pose)
		
		# print('Current pose:',pose)
		with data_lock:
			data.append((i,pose,frame))

		# Display the resulting frame
		cv2.imshow('frame', frame)
		
		cv2.imwrite(os.path.join(path, 'capture.png'), frame)
		
		# the 'q' button is set as the
		# quitting button you may use any
		# desired button of your choice
		if cv2.waitKey(1) & 0xFF == ord('q'):
			terminate.set()
		i+=1

	except KeyboardInterrupt:
		# After the loop release the cap object
		vid.release()
		# Destroy all the windows
		cv2.destroyAllWindows()

		break


			
if __name__ == '__main__':
	main()
