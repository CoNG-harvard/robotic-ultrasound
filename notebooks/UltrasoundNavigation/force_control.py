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

def get_average_force_reading(rtde_r, w0):
    # Force measurement is noisy, so we take averages.
    f_z = np.empty([200, 6])
    for i in range(200):
        w = rtde_r.getActualTCPForce()
        f_z[i] = np.array(w) - np.array(w0)
    mean_f = np.mean(f_z, axis=1)
    print('Ave force', mean_f)
    # return np.max(f_z),np.min(f_z)

def move_z(rtde_c,rtde_r, d, speed = 0.001):
    ''' d: z-direction displancement in millimeters'''
    tcp_pose = rtde_r.getActualTCPPose()

    target_pose = rtde_c.poseTrans(tcp_pose,[0,0,d/1000,0,0,0])

    # When doing position-based force control, the speed has to be extremely slow for stable results.
    return rtde_c.moveL(target_pose,speed=speed, acceleration=0.1)

def move_x(rtde_c,rtde_r, d, speed = 0.001):
    ''' d: x-direction displancement in millimeters'''
    tcp_pose = rtde_r.getActualTCPPose()

    target_pose = rtde_c.poseTrans(tcp_pose,[d/1000,0,0,0,0,0])

    # When doing position-based force control, the speed has to be extremely slow for stable results.
    return rtde_c.moveL(target_pose,speed=speed, acceleration=0.1)

def force_reached(target_force,f_max,f_min,force_error_tolerance):
    return target_force+force_error_tolerance>=f_max and target_force-force_error_tolerance<=f_min

def rotate_x(rtde_c, rtde_r, angle):
    """rotate TCP by angle on x axis.

    Args:
        rtde_c 
        rtde_r 
        angle (float): degree of rotation of x axis.
    """
    tcp_pose = rtde_r.getActualTCPPose()
    target_pose = rtde_c.poseTrans(tcp_pose, [0, 0, 0, angle / 180 * np.pi, 0, 0])
    return rtde_c.moveL(target_pose,speed=0.005,acceleration=0.1)


def rotate_y(rtde_c, rtde_r, angle, speed=0.005):
    """rotate TCP by angle on x axis.

    Args:
        rtde_c 
        rtde_r 
        angle (float): degree of rotation of x axis.
    """
    tcp_pose = rtde_r.getActualTCPPose()
    target_pose = rtde_c.poseTrans(tcp_pose, [0, 0, 0, 0, angle / 180 * np.pi, 0])
    return rtde_c.moveL(target_pose,speed=speed,acceleration=0.1)


def rotate_y_until_prependicular(rtde_r, rtde_c, w0, MAX_ITER = 15):
    """Rotate probe w.r.t. y axis until the probe is prependicular to the phantom. Decision
    made by pressing down the probe and observing the torque difference. Then implemented 
    a heuristic feedback controller to control the rotation angle.

    Args:
        rtde_r (rtde_control.RTDEControlInterface):
        rtde_c (rtde_receive.RTDEReceiveInterface): 
        w0 (list size 6): zero force sensor readings.
        MAX_ITER (int, optional): max iteration to find prependicular pose. Defaults to 15.
    """
    for iter in range(MAX_ITER):
        z_force_control(5, rtde_r, rtde_c, w0)
        x_torque_ini = get_average_force(rtde_r, w0)[3]
        z_force_control(8, rtde_r, rtde_c, w0)
        x_torque_press = get_average_force(rtde_r, w0)[3]
        diff = x_torque_press - x_torque_ini
        print(f'torque diff: {diff}')
        if np.abs(diff) >= 0.015: # TODO: heuristic values
            angle = 300 * (np.abs(diff) - 0.015) * np.sign(diff)
            print("rotate {}".format(angle))
            rotate_y(rtde_c, rtde_r, angle, speed=0.02)
        else:
            z_force_control(5, rtde_r, rtde_c, w0)
            print("find prependicular pos")
            break

def get_average_force(rtde_r, w0):
    """get average force with in 0.1s.

    Args:
        rtde_r (rtde_receive.RTDEReceiveInterface): 
        w0 (list size 6): zero force sensor readings.
    """
    f_z = np.empty([10, 6])
    for i in range(10):
        time.sleep(0.01)
        w = rtde_r.getActualTCPForce()
        f_z[i] = np.array(w) - np.array(w0)
        # print(f_z)
    mean_f = np.mean(f_z, axis=0)
    print('Ave force', mean_f)
    return(mean_f.tolist())
