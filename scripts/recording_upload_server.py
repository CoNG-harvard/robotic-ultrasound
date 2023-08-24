#!/usr/bin/env python
import time
import geometry_msgs.msg

from std_srvs.srv import Empty, EmptyResponse
import pickle as pkl
import cv2
# Modules required by the get_key() function, used in the manual mode.
import os
import sys
import rospkg
root_dir = rospkg.RosPack().get_path('robotic-ultrasound')
sys.path.append(root_dir)

from sftp import Sftp_Helper
from socket_helper import Socket_Hepler_Client

import rospy
import rtde_control
import rtde_receive


class CaptureServer(object):
	def __init__(self) -> None:
		# rtde_c = rtde_control.RTDEControlInterface("192.168.1.13")
		self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.13")
		self.vid = cv2.VideoCapture('/dev/video1')
		self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
		self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

		self.save_path = '/home/mht/temp'
		self.sftp_helper = Sftp_Helper()
		self.socket_helper = Socket_Hepler_Client()
		self.idx = 0

	def save_data(self):
		ret, frame = self.vid.read()
		pose = self.rtde_r.getActualTCPPose()
		force = self.rtde_r.getActualTCPForce()
		data = {'image':frame,
				'pose':pose,
				'force':force}
		with open('{}/data_{}.pkl'.format(self.save_path, self.idx),'wb') as f:
			pkl.dump(data,f)
		self.idx += 1


	def save_data_and_upload(self, req):
		print("requesting data save and upload")
		ret, frame = self.vid.read()
		pose = self.rtde_r.getActualTCPPose()
		force = self.rtde_r.getActualTCPForce()
		data = {'image':frame,
				'pose':pose,
				'force':force}
		with open('{}/data_{}.pkl'.format(self.save_path, self.idx),'wb') as f:
			pkl.dump(data,f)
		self.sftp_helper.Transfer_data(source_path = '{}/data_{}.pkl'.format(self.save_path, self.idx), 
					dest_path = '/home/local/PARTNERS/sk1064/workspace/control/data_retrieve/data_{}.pkl'.format(self.idx) )
		# data = pkl.dumps(data)
		# self.socket_helper.Transfer_action(data)
		self.idx += 1
		return EmptyResponse()
	
	def upload(self,):
		for i in range(5):
			self.sftp_helper.Transfer_data(source_path = '{}/data_{}.pkl'.format(self.save_path, i), 
					dest_path = '/home/local/PARTNERS/sk1064/workspace/control/data_retrieve/data_{}.pkl'.format(i) )
		


def save_data_and_upload_server():
    server = CaptureServer()
    rospy.init_node('SaveDataUpload', disable_signals=True)
    s = rospy.Service('SaveDataUpload', Empty, server.save_data_and_upload)
    print("Ready save data and upload.")
    rospy.spin()
    if KeyboardInterrupt:
	    server.vid.release()
	    
def pure_upload():
	server = CaptureServer()
	server.upload()

if __name__ == "__main__":
    pure_upload()
