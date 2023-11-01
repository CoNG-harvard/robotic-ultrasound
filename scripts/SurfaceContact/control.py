import numpy as np
import cv2
from numpy import linalg as la
import pyrealsense2 as rs
from .scene import extractBodyPixels,bodyCentroid
from .utils import patch_pixel_indx, normal_vector
        

class SurfaceContactControl:
    def __init__(self,
        rsPipeline,rtde_c,rtde_r,
        camera_pos_offset, probe_pos_offset,
        body_color_rgb = (227,124,100),
        body_hsv_rad = (50,150,150)):
        '''
           camera_pos_offset, probe_pos_offset: the relative pose of camera frame and probe tip in TCP frame.
        '''
        self.camera_2_tcp = camera_pos_offset
        self.probe_2_tcp = probe_pos_offset

        self.pipeline = rsPipeline
        self.rc = rtde_c
        self.rr = rtde_r

        self.body_color_rgb = body_color_rgb
        self.body_hsv_rad = body_hsv_rad
        
        self.rgb_img = []
        self.mask = []
        self.target_pixel_loc = []
        self.pixel_center = []
        self.pixel_offset = []

        self.points = [] 
        # The points are 3D cooridinates in the camera frame.

    def getContact(self):
        speed = [0, 0, -0.050, 0, 0, 0]
        return self.rc.moveUntilContact(speed)
    
    def fetchCameraStreamData(self):
        # Get camera stream data
        frames = self.pipeline.wait_for_frames()
        rgb = frames.get_color_frame()
        depth = frames.get_depth_frame()
        pc = rs.pointcloud()
        points = pc.calculate(depth).get_vertices()  


        self.rgb_img = np.asanyarray(rgb.get_data())


        self.mask =  extractBodyPixels(self.rgb_img,self.body_color_rgb,self.body_hsv_rad)
        self.target_pixel_loc = np.array(bodyCentroid(self.mask),dtype = np.int)     
        # self.pixel_center = np.array([w//2,h//2])
        h,w = self.mask.shape
        self.pixel_center = np.array([w//2,280]) 
        self.pixel_offset = self.target_pixel_loc-self.pixel_center
        # The center of the probe is not the same as center of the image. Can be calibrated later.

        # The points are 3D cooridinates in the camera frame.
        self.points = np.asanyarray(points).view(np.float32).reshape(self.rgb_img.shape)
        
        

    def mainloop(self):
        
        self.fetchCameraStreamData()
        
        # self.pure_pixel_control()
        return self.loc_normal_control()

    def loc_normal_control(self,hover_height = 0.50):
        # Normal vector alignment control
        patch_pixel_rad = 50

        cx,cy = np.array(self.target_pixel_loc,dtype=int)
        h,w = self.mask.shape

        patch_indx = patch_pixel_indx(cx,cy,h,w,patch_pixel_rad)

        patch_verts = [self.points[i,j] for i,j in patch_indx if np.any(self.points[i,j])]

        if len(patch_verts)>=3:

            
            camera_tcp_offset = np.array(self.camera_2_tcp) # To be determined using measurement in lab.
            tcp = self.rr.getActualTCPPose()

            # Tip: always use poseTrans to do pose addition. Directly using the "+" operator is usually incorrect.
            camera_pose = self.rc.poseTrans(tcp, camera_tcp_offset)

            body_loc_cam = np.mean(patch_verts,axis = 0) # Body centroid location in camera frame.
            

            body_normal_vec_cam = normal_vector(patch_verts) # Body surface normal vector in camera frame.
            
            # Body surface orientaion, three rotation angles, in camera frame.
            # To be calculated from body_normal_vec_cam
            body_ori_cam =  np.array([0,0,0]) 
            
            goal_pose_base = self.rc.poseTrans(camera_pose,
                                            np.hstack([body_loc_cam,body_ori_cam])+\
                                            np.array([0,0,-hover_height,0,0,0])
                                            )

            # Move the robot to the goal pose.
            speed = 0.03
            acc = 0.1
            self.rc.moveL(goal_pose_base,speed,acc)

            return np.linalg.norm(np.array(tcp[:3])-np.array(goal_pose_base[:3]))
        else:
            print("Not moving because not enough body pixels is seen.")
            return None
       

    def pure_pixel_control(self):
        
        
        pixel_dist = la.norm(self.pixel_offset)

        step_size = min(pixel_dist/1000,0.05)
        step = self.pixel_offset/pixel_dist * step_size
        
        q = self.rr.getActualTCPPose()
        target_pose = self.rc.poseTrans(q,[step[0],step[1],0,
                                                       0,0,0])
        speed = 0.03
        acc = 0.1
        self.rc.moveL(target_pose,speed,acc)

        return pixel_dist


    def showScene(self,axes):
        axes[0].imshow(self.rgb_img)
        axes[0].set_title('Camera Image')

        result = cv2.bitwise_and(self.rgb_img,self.rgb_img,mask = self.mask)
        # plt.subplot(2,2,3)
        # plt.imshow(self.mask,cmap = 'gray')

        # plt.subplot(2,2,4)
        # plt.imshow(result)

        axes[1].scatter(self.target_pixel_loc[0],self.target_pixel_loc[1],marker="x",color = 'yellow',label='target',s=100)
        # axes[1].scatter(self.pixel_center[0],self.pixel_center[1],marker="+",color = 'white',label='crosshair',s = 150)

        arrow_width = 5
        head_length = 4.5*arrow_width
        # axes[1].arrow(*self.pixel_center,
        #         *(self.pixel_offset-head_length*self.pixel_offset/la.norm(self.pixel_offset)),
        #         color = 'red',width = arrow_width,head_length = head_length,
        #         label = 'Moving direction')
        
        axes[1].imshow(result)
        axes[1].legend()
        axes[1].set_title('Body pixels')
        # axes[1].set_title("Pixel distance to target:{}".format(la.norm(self.pixel_offset)))