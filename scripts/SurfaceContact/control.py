import numpy as np
import cv2
from numpy import linalg as la
from .scene import extractBodyPixels,bodyCentroid

class SurfaceContactControl:
    def __init__(self,rsPipeline):
        self.pipeline = rsPipeline
        
        self.rgb_img = []
        self.mask = []
        self.target_loc = []
        self.center = []
        self.offset = []


    def mainloop(self):
        frames = self.pipeline.wait_for_frames()
        rgb = frames.get_color_frame()
        self.rgb_img = np.asanyarray(rgb.get_data())
        self.mask =  extractBodyPixels(self.rgb_img)
        self.target_loc = np.array(bodyCentroid(self.mask),dtype = np.int)

        h,w = self.mask.shape
        self.center = np.array([w//2,h//2])
        self.offset = self.target_loc-self.center
        return self.offset


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
                *(self.offset-head_length*self.offset/la.norm(self.offset)),
                color = 'red',width = arrow_width,head_length = head_length,
                label = 'Moving direction')
        
        axes[1].imshow(result)
        axes[1].legend()