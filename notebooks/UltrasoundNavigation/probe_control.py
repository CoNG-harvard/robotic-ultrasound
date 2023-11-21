import numpy as np
import time
# Move probe to align with the center of the mask pixels
def horizontal_align_step(mask,rtde_r,rtde_c):
    coords = np.argwhere(mask)
    mean_z,mean_x = np.mean(coords,axis = 0)
    center_x = mask.shape[1]/2
    x_diff = mean_x - center_x
    tcp_pose = rtde_r.getActualTCPPose()
    step_size = 0.5 * 0.01 
    target_pose = rtde_c.poseTrans(tcp_pose,[np.sign(x_diff)*step_size,0,0,0,0,0])
    vel = 0.02
    acc = 0.1
    return rtde_c.moveL(target_pose,vel,acc)
