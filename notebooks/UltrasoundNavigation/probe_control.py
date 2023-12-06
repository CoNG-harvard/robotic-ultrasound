import numpy as np
import time
import sys
sys.path.append("../../")

from scripts.USImageCV.utils.bunny import bunny_mask_predict,max_connected_area


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


def bunny_view_optim(rtde_c,rtde_r,start_pose,target_pose,
                     ultrasoud_vid,
                     model,device,
                     search_vel=0.001,
                     search_acc=0.1):
    # Optimize the bunny view by traversing the robot from a start pose to a goal pose.

    # Move the TCP to the starting pose.
    rtde_c.moveL(start_pose,0.01,0.1) 
    
    # After reaching the starting pose, fire off the TCP to move towards the target_pose.
    # Setting asynchoronous=True allows the code to fly forward while the robot execute the motion independently.
    rtde_c.moveL(target_pose,search_vel,search_acc,asynchronous=True) 
    
    # In the asynchorous mode of moveL, we have to actively monitor the robot speed to determine whether the robot has stopped.
    speeds = []
    start_time = time.time()

    time.sleep(0.01) # This will allow the robot to start moving a little
    area_diff = deque(maxlen=3)

    max_area = 0
    optim_pose = []
    frames = []
    for i in range(1000):
        linear_tcp_speed = np.linalg.norm(rtde_r.getActualTCPSpeed()[:3])
        speeds.append(linear_tcp_speed)
        
        ret, frame = ultrasoud_vid.read()	
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
def find_object(detect_model,
                rtde_c,rtde_r,
                ultrasound_vid,device,
                area_threshold=5000):
    '''
        detect_model: a neural network that detects the pixels of the object to find.
    '''

    def y_direction_search(model):
        # y direction search has to be very careful, with low speed vel=0.001
        tcp_pose = rtde_r.getActualTCPPose()

        start_pose = rtde_c.poseTrans(tcp_pose,[0,-0.03,0,0,0,0])

        target_pose = rtde_c.poseTrans(tcp_pose,[0,+0.03,0,0,0,0])

        return bunny_view_optim(rtde_c,rtde_r,start_pose,target_pose,
                            ultrasound_vid,
                            model,device,
                            search_vel = 0.001,
                            search_acc = 0.1)

    def x_direction_search(model):
        # x direction search can be more crude, with intermediate speed vel=0.01
        tcp_pose = rtde_r.getActualTCPPose()

        start_pose = rtde_c.poseTrans(tcp_pose,[-0.05,0,0,0,0,0])

        target_pose = rtde_c.poseTrans(tcp_pose,[+0.05,0,0,0,0,0])

        return bunny_view_optim(rtde_c,rtde_r,start_pose,target_pose,
                            ultrasound_vid,
                            model,device,
                            search_vel = 0.01,
                            search_acc = 0.1)

    def centralize_object(model):
        '''
            model: the neural network responsible for outputing the segmentation mask of the desired object.
        '''
        for i in range(20):
            ret, frame = ultrasound_vid.read()	
            mask = bunny_mask_predict(model,frame,device)
            z_c,x_c=np.mean(np.argwhere(mask),axis=0)

            # We have ensured that the x axis of TCP aligns with axis 1 of frame, and z axis of TCP aligns with axis[0] of frame.
            x_dir = x_c-frame.shape[1]//2
            print("Centralizing the object. Pixel distance remaining: ",np.linalg.norm(x_dir))

            if np.linalg.norm(x_dir)<20:
                break

            tcp_pose = rtde_r.getActualTCPPose()

            move_step = 1/100
            target_pose = rtde_c.poseTrans(tcp_pose,[move_step*np.sign(x_dir),0,0,0,0,0])


            rtde_c.moveL(target_pose,0.005,0.1)
        
    max_a = 0
    while True:
        y_direction_search(detect_model)
        
        a,_ = x_direction_search(detect_model)
        print('Max area',a)
        if a>=max_a:
            max_a = a
        if max_a>=area_threshold:
            break
    centralize_object(detect_model)



def record_registration(record_model,n_samples):
    start_pose = rtde_r.getActualTCPPose()
    frames = []
    poses = []
    
    # Move and record in the feet direction
    target_pose = rtde_c.poseTrans(start_pose,[0,-0.03,0,0,0,0])
    f,p = record_registration_frames(rtde_c,rtde_r,start_pose,target_pose,
                        ultrasoud_vid,
                        record_model,device,
                        n_waypoints = n_samples//2)
    
    # Make sure the order is from feet towards head
    f.reverse()
    p.reverse()

    frames+=f
    poses+=p
    # Move and record in the head direction
    target_pose = rtde_c.poseTrans(start_pose,[0,0.03,0,0,0,0])
    f,p = record_registration_frames(rtde_c,rtde_r,start_pose,target_pose,
                        ultrasoud_vid,
                        record_model,device,
                        n_waypoints = n_samples//2)
    frames+=f
    poses+=p
    return frames,poses
def record_registration_frames(rtde_c,rtde_r,start_pose,target_pose,
                     ultrasoud_vid,
                     model,device,
                    n_waypoints = 50):

    start_loc = start_pose[:3]
    target_loc = target_pose[:3]
    waypoints = np.linspace(start_loc,target_loc,n_waypoints)
    waypoints = [list(w)+start_pose[3:] for w in waypoints]

    rtde_c.moveL(start_pose,0.01,0.1)
    # y direction search has to be very careful, with low speed vel=0.001
    tcp_pose = rtde_r.getActualTCPPose()
    start_pose = tcp_pose

    # In the asynchorous mode of moveL, we have to actively monitor the robot speed to determine whether the robot has stopped.
    speeds = []
    start_time = time.time()

    time.sleep(0.01) # This will allow the robot to start moving a little
    n_hist = 5
    areas = deque(maxlen=n_hist)

    poses=[]
    frames = []
    for w in waypoints:
        
        ret, frame = ultrasoud_vid.read()
        frames.append(frame)
        ###### Must record the curr_pose right after the image read and before the neural network inference.
        curr_pose = rtde_r.getActualTCPPose()	
        poses.append(curr_pose)
        ######
        mask = bunny_mask_predict(model,frame,device)
        area = max_connected_area(mask)
        print("area",area,"Time elapsed:",time.time()-start_time)
        areas.append(area)
        if len(areas)>=n_hist and np.max(areas)<=2000:
            # If the area starts to consistently be small, stop the robot.
            # And move the robot to the start pose
            break
        rtde_c.moveL(w,0.01,0.1)
    rtde_c.moveL(start_pose,0.01,0.1)
    return frames,poses