import pyrealsense2 as rs
import numpy as np
import sys
import cv2
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
    move_default_pose(rtde_c)
    s = input("Start:?")
    if s=='y':
        sc = SurfaceContactControl(pipeline,rtde_c,rtde_r)

        fig = plt.figure(dpi=100,figsize = (12,4))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        for _ in range(40):
            ax1.clear()
            ax2.clear()
            dist = sc.mainloop()
            dist_threshold = 10
            if dist<dist_threshold:
                break
            sc.showScene([ax1,ax2])
            plt.pause(0.005)
        print("Arrived at above target location.")

        if sc.getContact():
            print("Contact made.")
        
        
finally:
    pipeline.stop()