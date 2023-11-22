import numpy as np
from tqdm import tqdm

def z_force_control(target_force,rtde_r,rtde_c,w0,force_err_tolerance=0.5):
    '''
        Drive the TCP in its z-direction until its z_force reading reaches the target_force.
        
        w0: the zero point wrench.
    '''
    MAX_ITER = 100


    steps = tqdm(range(MAX_ITER),bar_format='{desc} Time elapsed={elapsed}')
    for i in steps:
        f_min,f_max = get_z_force(rtde_r,w0)
        f_med = (f_min+f_max)/2
        steps.set_description('Current force is:{}'.format(f_med))
       
        if force_reached(target_force,f_max,f_min,force_err_tolerance):
            print('Target force reached.')
            return True
        else:
            f_diff = target_force-f_med

            move_z(rtde_c,rtde_r,0.5*np.sign(f_diff))
    else:
        print('Failed to reach target force in {} iterations.'.fomrat(MAX_ITER))
        return False

def get_z_force(rtde_r,w0):
    # Force measurement is noisy, so we take averages.
    f_z = []
    for i in range(200):
        w = rtde_r.getActualTCPForce()
        f_z.append(w[2]-w0[2])
        # mean_f.append(np.mean(f_z))
        # print('Ave force',np.mean(f_z))
    return np.max(f_z),np.min(f_z)

def move_z(rtde_c,rtde_r,d):
    ''' d: z-direction displancement in millimeters'''
    tcp_pose = rtde_r.getActualTCPPose()

    target_pose = rtde_c.poseTrans(tcp_pose,[0,0,d/1000,0,0,0])

    # When doing position-based force control, the speed has to be extremely slow for stable results.
    return rtde_c.moveL(target_pose,speed=0.001,acceleration=0.1)

def force_reached(target_force,f_max,f_min,force_error_tolerance):
    return target_force+force_error_tolerance>=f_max and target_force-force_error_tolerance<=f_min
