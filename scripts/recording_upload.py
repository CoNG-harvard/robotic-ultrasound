import time
import moveit_commander
import rospy
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list


import pickle as pkl


from pathlib import Path
import shutil

import cv2

# Modules required by the get_key() function, used in the manual mode.
import os
import sys


# For concurrent streaming and data saving
from threading import Thread, Event, Lock

terminate = Event()
data_lock = Lock()


def data_saving(dt,data_container,save_to_path=None):
	
	prev_idx = 0

	if save_to_path is None:
		save_to_path = os.getcwd()
		tmp_path = 'tmp'
	else:
		tmp_path = '{}/tmp'.format(save_to_path)
	
	#creating a new directory called pythondirectory
	Path(tmp_path).mkdir(parents=True, exist_ok=True)

	while True:
		print('save data')
		if cv2.waitKey(1) & 0xFF == ord('q'):
			terminate.set()
			
		if terminate.is_set():
			print('Saving data and shutting down.')
			
			timestamp = rospy.get_rostime()
			
			with open('{}/data_{}{}.pkl'.\
				      format(save_to_path,
				      	     timestamp.secs,
				      	     timestamp.nsecs),'wb') as f:
				
				pkl.dump(data_container,f)
			
			shutil.rmtree(tmp_path) # If the entire file is safely saved, remove the tmp files.
			break

		if prev_idx < len(data_container)-1:

			curr_idx = len(data_container)-1
			print("============ Saving data {}-{}".format(prev_idx,curr_idx))
			
			with open('{}/{}-{}.pkl'.\
						format(tmp_path,
							   prev_idx,
							   curr_idx),'wb') as f:
				
				pkl.dump(data_container[prev_idx:],f) 
				# For safety reason, we save a chunk of data periodically.
				# We can recover everything from tmp_path if streaming went wrong.

			prev_idx = curr_idx+1
			
			print("")
				
		time.sleep(dt)


def main():	
	save_every_t = 1

	rospy.init_node('streaming_recorder')
	
	data = []
	
	# define a video capture object
	vid = cv2.VideoCapture('/dev/video5')

	# Define the robot pose listener
	group_name = 'manipulator'
	
	move_group = moveit_commander.MoveGroupCommander(group_name)
	robot = moveit_commander.RobotCommander()
		
	saving_thread = Thread(target=data_saving,args = (save_every_t,data,))

	saving_thread.start()

	i = 0

	while(True):
		try:
			# Capture the video frame
			# by frame
			ret, frame = vid.read()

			# pose = []
			pose = pose_to_list(move_group.get_current_pose().pose)
			
			# print('Current pose:',pose)
			with data_lock:
				data.append((i,pose,frame))

			# Display the resulting frame
			cv2.imshow('frame', frame)
			
			# the 'q' button is set as the
			# quitting button you may use any
			# desired button of your choice
			if cv2.waitKey(1) & 0xFF == ord('q'):
				terminate.set()
				
			i+=1
		except KeyboardInterrupt:
			terminate.set()
			saving_thread.join()
			# After the loop release the cap object
			vid.release()
			# Destroy all the windows
			cv2.destroyAllWindows()

			break


			
if __name__ == '__main__':
	main()
