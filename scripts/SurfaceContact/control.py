import numpy as np
import cv2
from numpy import linalg as la
from .scene import extractBodyPixels,bodyCentroid

class SurfaceContactControl:
    def __init__(self,rsPipeline,rtde_c,rtde_r):
        self.pipeline = rsPipeline
        self.rc = rtde_c
        self.rr = rtde_r
        
        self.rgb_img = []
        self.mask = []
        self.target_loc = []
        self.center = []
        self.pixel_offset = []

    def getContact(self):
        speed = [0, 0, -0.050, 0, 0, 0]
        self.rc.moveUntilContact(speed)
    
    def mainloop(self):
        frames = self.pipeline.wait_for_frames()
        rgb = frames.get_color_frame()
        self.rgb_img = np.asanyarray(rgb.get_data())
        self.mask =  extractBodyPixels(self.rgb_img)
        self.target_loc = np.array(bodyCentroid(self.mask),dtype = np.int)

        h,w = self.mask.shape
        # self.center = np.array([w//2,h//2])
        self.center = np.array([w//2,280]) 
        # The center of the probe is not the same as center of the image. Can be calibrated later.

        self.pixel_offset = self.target_loc-self.center
        
        pixel_dist = la.norm(self.pixel_offset)
        

        step_size = min(pixel_dist/1000,0.05)
        step = self.pixel_offset/pixel_dist * step_size
        
        q = self.rr.getActualTCPPose()
        target_pose = np.array(q)
        target_pose = self.rc.poseTrans(q,[step[0],step[1],0,
                                                       0,0,0])
        speed = 0.03
        acc = 0.1
        self.rc.moveL(target_pose,speed,acc)

        return pixel_dist


    def showScene(self,axes):
        axes[0].imshow(self.rgb_img)

        result = cv2.bitwise_and(self.rgb_img,self.rgb_img,mask = self.mask)
        # plt.subplot(2,2,3)
        # plt.imshow(self.mask,cmap = 'gray')

        # plt.subplot(2,2,4)
        # plt.imshow(result)

        axes[1].scatter(self.target_loc[0],self.target_loc[1],marker="x",color = 'yellow',label='target',s=100)
        axes[1].scatter(self.center[0],self.center[1],marker="+",color = 'white',label='center',s = 150)

        arrow_width = 5
        head_length = 4.5*arrow_width
        axes[1].arrow(*self.center,
                *(self.pixel_offset-head_length*self.pixel_offset/la.norm(self.pixel_offset)),
                color = 'red',width = arrow_width,head_length = head_length,
                label = 'Moving direction')
        
        axes[1].imshow(result)
        axes[1].legend()
        axes[1].set_title("Pixel distance to target:{}".format(la.norm(self.pixel_offset)))