      
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch.nn.functional as F


color_list = [(255, 0, 0), (0, 0, 255), (0, 250, 0), (180, 0 , 250), (255,100, 0), (0,255,100), (100,255,100),(100,0,100),(200,255,0),(0, 20, 20), (100,0,40), (100,255,100),(100,0,100),(200,255,0),(100,0,40)]

color_dict = {'blue':(0,0,255),'green':(0,255,0),'red':(255,0,0),'yellow':(255,255,0),'cyan':(0,255,255),'deepskyblue':(0,191,255)}


def draw_mask(image, mask,color='blue'):
    masked_image = image.copy()
    masked_image[np.where(mask)] = color_dict[color]
    masked_image = masked_image.astype(np.uint8)
    masked_image = cv2.addWeighted(image, 0.4, masked_image, 0.6, 0)
    return masked_image

# def draw_contour(input_image, pred, color = 'blue'):
    

#     # mask = (F.sigmoid(mask_>0.5)).cpu()
#     # mask = np.array((F.sigmoid(mask_)>0.5).cpu()).astype(np.uint8)
#     mask = np.array( pred.astype(np.uint8))
#     contours_pred, _  = cv2.findContours(
#         (mask), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
#     )

#     # print (len(contours_pred))
    
#     if len(contours_pred) > 1:
#         pt_max = 0
#         for iidx in range(len(contours_pred)):
#             shape = np.shape(contours_pred[iidx])
#             if pt_max < np.shape(contours_pred[iidx])[0]:
#                 pt_max = np.shape(contours_pred[iidx])[0]
#                 # out = np.reshape(contours_pred[idx], (pt_max, 2))
#                 contours_pred_max = contours_pred[iidx]
#     elif len(contours_pred) == 0:
#         print ('')
#     else:
#         contours_pred_max = contours_pred[0]
    
#     cv2.drawContours(input_image, [contours_pred_max], 0, color_dict[color], 5)

#     return input_image

# def save_png(save_dir, image):
#     plt.imsave(save_dir, image)
    
# def mkdirs(dirs) -> None:
#     for dir in dirs:
#         os.makedirs(dir, exist_ok=True)
        
# def save_as_png(png_path, img, pred, fr, phase):
#     # ([1, 4, 32, 224, 224]) pred size
#     sh = np.shape(img)

#     dest_dir = os.path.join(png_path)
    
#     if not os.path.exists(dest_dir):
#         mkdirs([dest_dir])

#     if isinstance(img, np.ndarray):
#         img = img.astype(dtype=np.uint8)
#     else:
#         img = img.cpu().numpy().astype(dtype=np.uint8)
                    
#     slice_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
#     input_image_with_contour = draw_contour(
#         slice_img, pred, [1]
#         )

#     save_png(dest_dir + "/" + phase + "_" + str(fr) + ".png", input_image_with_contour)
            