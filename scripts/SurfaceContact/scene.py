import cv2
from matplotlib.colors import rgb_to_hsv
import numpy as np
def extractBodyPixels(rgb_scene,
                      body_color_rgb=(107, 64, 41),
                      hsv_radius=(20,30,20)):
    """
    Input: an rgb image. 
    
    Parameters:
        hsv_radius: (hue_radius>=0,saturation radius>=0,value radius>=0)~[0,255] tuple. Tune the components of hsv_radius to capture as much body as possible. A smaller hsv_radius means the color range is smaller. 
        body_color_rgb: (r,g,b)~[0,255] tuple that rounghly matches major color of the body. You may get this value of the body using any color picker app on your OS.
    
    Output: a mask indicating the pixels of the body in rgb_scene.
    """
    
    hsv_scene = cv2.cvtColor(rgb_scene,cv2.COLOR_RGB2HSV)
    
    # Hard coded extraction #
    hsv_radius = np.array(hsv_radius) # Tune the components of hsv_radius to capture as much body as possible. A smaller hsv_radius means the color range is smaller.
    body_color_rgb = np.array(body_color_rgb) # Get this value of the body using any color picker app on your OS.
    
    body_color_hsv = rgb_to_hsv(body_color_rgb/255)*255 # The rgb_to_hsv function from matplotlib uses input and output in [0,1].

    lo_hsv = body_color_hsv-hsv_radius
    hi_hsv = body_color_hsv+hsv_radius
    mask = cv2.inRange(hsv_scene,lo_hsv,hi_hsv)
    # print(hsv_scene.shape,mask.shape,mask)
    return mask
def bodyCentroid(mask):
    # mask: a binary 2-D array.
    # Output: (x,y), the centroid(mean) location of the non-zero pixels in mask. 
    # Note the y coordinate corresponds to the h direction, and x coordinate is in the w direction.

    body_pix = np.argwhere(mask)
    coord = np.mean(body_pix,axis = 0)
    return np.array(coord[::-1])