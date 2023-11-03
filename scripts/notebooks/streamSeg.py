import pyrealsense2 as rs
import numpy as np
import sys
import cv2
import yaml
import matplotlib.pyplot as plt

import time
import rtde_control
import rtde_receive


sys.path.append("../")

from SurfaceContact.control import SurfaceContactControl

from SurfaceContact.move import move_default_pose

try:
    
    pipeline = rs.pipeline()
    pipeline.start()

    rtde_c = rtde_control.RTDEControlInterface("192.168.1.13")
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.13")
    
    while True:
        s = input('Go to default pose?')
        move_default_pose(rtde_c)
        s = input("Start:?")
        if s=='y':
            with open('../config/pose_in_TCP_frame.yaml','r') as f:
                poses = yaml.safe_load(f)

                camera_2_tcp = poses['camera']
                probe_2_tcp = poses['probe']

            body_color_rgb = (227,124,100)
            body_hsv_rad = (5,50,100)
            sc = SurfaceContactControl(pipeline,rtde_c,rtde_r,
                                    camera_2_tcp,probe_2_tcp,
                                    body_color_rgb,body_hsv_rad)

            fig = plt.figure(dpi=100,figsize = (12,4))
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)

            for _ in range(40):
                ax1.clear()
                ax2.clear()
                dist = sc.mainloop()
                dist_threshold = 0.005
                if dist<dist_threshold:
                    break
                sc.showScene([ax1,ax2])
                plt.pause(0.005)
            print("Arrived at above target location. Start descending.")
            
            sc.loc_normal_control(hover_height=0.20)
            # s = input("Find surface:?")
        
            if sc.getContact():
                print("Contact made.")
            
        
finally:
    pipeline.stop()