import cv2
import numpy as np
import time
from collections import deque
import SimpleITK as sitk

from slice_matching import global_match,local_match
from scripts.USImageCV.utils.bunny import bunny_mask_predict,max_connected_area

def screen_brightness(ultrasound_vid):
    _,frame = ultrasound_vid.read()
    return np.mean(cv2.cvtColor(frame[200:800],cv2.COLOR_RGB2GRAY))

def find_surface(sc):
    for _ in range(40):
        dist = sc.mainloop()
        dist_threshold = 0.005
        if dist<dist_threshold:
            break
    print("Arrived at above target location. Start descending.")

    sc.loc_normal_control(hover_height=0.20)
    # s = input("Find surface:?")

    if sc.getContact():
        print("Contact made.")


def find_object(model,device,rtde_c,rtde_r,ultrasound_vid,area_threshold=5000):
    '''
        model: a neural network that detects the pixels of the object to find.
    '''

    def view_optim(start_pose,target_pose,
                   search_vel=0.001,search_acc=0.1):
        # Optimize the view by traversing the robot from a start pose to a goal pose.

        # Move the TCP to the starting pose.
        rtde_c.moveL(start_pose,0.01,0.1) 
        
        # After reaching the starting pose, fire off the TCP to move towards the target_pose.
        # Setting asynchoronous=True allows the code to fly forward while the robot execute the motion independently.
        rtde_c.moveL(target_pose,search_vel,search_acc,asynchronous=True) 
        
        # In the asynchorous mode of moveL, we have to actively monitor the robot speed to determine whether the robot has stopped.
        speeds = []

        time.sleep(0.01) # This will allow the robot to start moving a little
        area_diff = deque(maxlen=3)

        max_area = 0
        optim_pose = []
        frames = []
        for _ in range(1000):
            linear_tcp_speed = np.linalg.norm(rtde_r.getActualTCPSpeed()[:3])
            speeds.append(linear_tcp_speed)
            
            ret, frame = ultrasound_vid.read()	
            frames.append(frame)
            ###### Must record the curr_pose right after the image read and before the neural network inference.
            curr_pose = rtde_r.getActualTCPPose()	
            ######


            mask = bunny_mask_predict(model,frame,device)
            area = max_connected_area(mask)
            # print("area",area,"Time elapsed:",time.time()-start_time)
            area_diff.append(area)
            
            if area>max_area:
                max_area = area
                optim_pose = curr_pose
            
            if len(area_diff)>=3 and\
                np.min(area_diff)>4000 and \
                    area_diff[-1]-area_diff[-2]<0 and area_diff[-2]-area_diff[-3]<0:
                # If the area starts to consistently decline, stop the robot. 
                # print(list(area_diff))
                rtde_c.stopL() 
                # We can call stopL() to manually stop the robot during execution.
                # This is very useful pairing with the async move of moveL.

                # And move the robot to the optimal pose
                rtde_c.moveL(optim_pose,0.01,0.1)

                return max_area,frames
    
        if linear_tcp_speed<1e-4:
            return max_area,frames
    

    def y_direction_search():
        # y direction search has to be very careful, with low speed vel=0.001
        tcp_pose = rtde_r.getActualTCPPose()

        start_pose = rtde_c.poseTrans(tcp_pose,[0,-0.03,0,0,0,0])

        target_pose = rtde_c.poseTrans(tcp_pose,[0,+0.03,0,0,0,0])

        return view_optim(start_pose,target_pose,
                            search_vel = 0.01,
                            search_acc = 0.1)

    def x_direction_search():
        # x direction search can be more crude, with intermediate speed vel=0.01
        tcp_pose = rtde_r.getActualTCPPose()

        start_pose = rtde_c.poseTrans(tcp_pose,[-0.05,0,0,0,0,0])

        target_pose = rtde_c.poseTrans(tcp_pose,[+0.05,0,0,0,0,0])

        return view_optim(start_pose,target_pose,
                            search_vel = 0.01,
                            search_acc = 0.1)
    max_a = 0
    while True:
        y_direction_search()
        
        a,_ = x_direction_search()
        print('Max area',a)
        if a>=max_a:
            max_a = a
        if max_a>=area_threshold:
            break


def centralize_object(model,device,rtde_c,rtde_r,ultrasound_vid):
    '''
        model: the neural network responsible for outputing the segmentation mask of the desired object.
    '''
    for i in range(20):
        ret, frame = ultrasound_vid.read()	
        mask = bunny_mask_predict(model,frame,device)
        z_c,x_c=np.mean(np.argwhere(mask),axis=0)

        # We have ensured that the x axis of TCP aligns with axis 1 of frame, and z axis of TCP aligns with axis[0] of frame.
        x_dir = x_c-frame.shape[1]//2
        # print("Centralizing the object. Pixel distance remaining: ",np.linalg.norm(x_dir))

        if np.linalg.norm(x_dir)<20:
            break

        tcp_pose = rtde_r.getActualTCPPose()

        move_step = 1/200
        target_pose = rtde_c.poseTrans(tcp_pose,[move_step*np.sign(x_dir),0,0,0,0,0])


        rtde_c.moveL(target_pose,0.05,0.1)
    

def record_registration(model,device,rtde_c,rtde_r,ultrasound_vid,n_samples,
                        rec_range = 0.03, track_IVC=True):
    SCREEN_BRIGHTNESS_THRESHOLD = 20
    
    def record(start_pose,target_pose, n_waypoints = 50, remove_start=False):
        start_loc = start_pose[:3]
        target_loc = target_pose[:3]
        waypoints = np.linspace(start_loc,target_loc,n_waypoints)
        if remove_start:
            waypoints = waypoints[1:]

        waypoints = [list(w)+start_pose[3:] for w in waypoints]

        # y direction search has to be very careful, with low speed vel=0.001
        tcp_pose = rtde_r.getActualTCPPose()
        start_pose = tcp_pose

        # In the asynchorous mode of moveL, we have to actively monitor the robot speed to determine whether the robot has stopped.
        start_time = time.time()

        n_hist = 5
        areas = deque(maxlen=n_hist)

        poses=[]
        frames = []
        for w in waypoints:
            
            rtde_c.moveL(w,0.05,0.1)
            time.sleep(0.1) # This will allow the robot stablize for a bit.
       
            _, frame = ultrasound_vid.read()
            frames.append(frame)
            ###### Must record the curr_pose right after the image read and before the neural network inference.
            curr_pose = rtde_r.getActualTCPPose()	
            poses.append(curr_pose)
            ######
            if screen_brightness(ultrasound_vid)<=SCREEN_BRIGHTNESS_THRESHOLD:
                break
            
            if track_IVC:
                mask = bunny_mask_predict(model,frame,device)
                area = max_connected_area(mask)
                print("area",area,"Time elapsed:",time.time()-start_time)
                areas.append(area)
                if len(areas)>=n_hist and np.max(areas)<=2000:
                    # If the area starts to consistently be small, stop the robot.
                    # And move the robot to the start pose
                    break

        return frames,poses
    start_pose = rtde_r.getActualTCPPose()
    frames = []
    poses = []
    
    # Move and record in the feet direction
    target_pose = rtde_c.poseTrans(start_pose,[0,-rec_range,0,0,0,0])
    f,p = record(start_pose,target_pose,n_waypoints = n_samples//2+1)
    
    # Make sure the order is from feet towards head
    f.reverse()
    p.reverse()

    frames+=f
    poses+=p
    # Move and record in the head direction
    target_pose = rtde_c.poseTrans(start_pose,[0,rec_range,0,0,0,0])
    f,p = record(start_pose,target_pose,n_waypoints = n_samples//2+1, remove_start=True)
    frames+=f
    poses+=p
    return frames,poses


def slice_matching_control(model,device,rtde_c,rtde_r, ultrasound_vid, vessel_ct_slice,us_spacing,mode='global'):
    SCREEN_BRIGHTNESS_THRESHOLD = 20

    def curr_matching_score(vessel_ct_slice,us_spacing,mode='global'):
        ret, frame = ultrasound_vid.read()
        pred_mask = bunny_mask_predict(model,frame,device)
        vessel_us_slice = pred_mask.T
        vessel_us_slice = sitk.GetImageFromArray(vessel_us_slice)
        vessel_us_slice.SetSpacing(us_spacing)

        # t = time.time()
        if mode == 'global':
            return global_match(vessel_ct_slice,vessel_us_slice)
        else:
            return local_match(vessel_ct_slice,vessel_us_slice)
        
    def search(y_step,n_iter = 20,lookback = 3):

        max_am = 0
        prev_am = None
        max_pos = rtde_r.getActualTCPPose()

        score_deque = deque(maxlen=lookback)
        diff_deque = deque(maxlen=lookback)
        for _ in range(n_iter):
             # print("Matching time elapsed:",time.time()-t,"Match score:",am)
            _,am = curr_matching_score(vessel_ct_slice,us_spacing,mode)
            curr_pose = rtde_r.getActualTCPPose()
            if am>max_am:
                max_pos = curr_pose
                max_am = am


            if prev_am is not None:
                diff_deque.append(am-prev_am)
            
            prev_am = am

            if screen_brightness(ultrasound_vid)<SCREEN_BRIGHTNESS_THRESHOLD:
                break


            if len(diff_deque)>=lookback and np.max(diff_deque)<0:
                break
            
            score_deque.append(am)
            if len(score_deque)>=lookback and np.max(score_deque)<100:
                break
            
            next_pose = rtde_c.poseTrans(curr_pose,[0,y_step,0,0,0,0])
            rtde_c.moveL(next_pose,0.05,0.1)
            if mode =='global':
                centralize_object(model,device,rtde_c,rtde_r,ultrasound_vid)
            
            
        return max_am, max_pos
    
            
    initial_pos = rtde_r.getActualTCPPose()

    

    step = 1/1000
    max_am,max_pos = search(step)

    rtde_c.moveL(initial_pos,0.01,0.1)

    step = -1/1000
    a,p = search(step)

    if a>max_am:
        max_pos = p
        max_am = a
    rtde_c.moveL(max_pos,0.01,0.1)
    return max_pos

def move_horizontal_record(rtde_c, rtde_r, ultrasound_vid, 
                           CT2US, target_loc_ct,us_origin,
                           direction = 'left-right'):

    def move_xy(ref_pos,rtde_r,rtde_c):
        tcp = rtde_r.getActualTCPPose()
        target_pose = ref_pos[:2]+tcp[2:]
        return rtde_c.moveL(target_pose,0.03,0.1)

    target_loc_us = CT2US.TransformPoint(target_loc_ct)
    tx = -target_loc_us[-1]/1000
    
    ty = -target_loc_us[0]/1000
    us_x = us_origin[0]
    us_y = us_origin[1]
    curr_pose = rtde_r.getActualTCPPose()

    # x-axis in robot base frame = head-feet axis in CT frame
    # y-axis in robot base frame = left-right axis in CT frame
    if direction=='left-right':
        target_loc_robot = [curr_pose[0],us_y+ty,0,0,0,0]
    elif direction == 'head-feet':
        target_loc_robot = [us_x+tx,curr_pose[1],0,0,0,0]
    elif direction == 'xy':
        target_loc_robot = [us_x+tx,us_y+ty,0,0,0,0]
    move_xy(target_loc_robot,rtde_r,rtde_c)
    time.sleep(0.3)
    ret,frame = ultrasound_vid.read()
    return frame