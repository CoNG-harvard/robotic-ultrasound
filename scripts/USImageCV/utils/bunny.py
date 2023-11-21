import cv2
from cv2 import cvtColor
import torch
import numpy as np
import torch.nn.functional as F
from ..models.unet import UNet
from skimage.measure import label, regionprops

def max_connected_area(mask):
    label_im = label(mask)
    regions = regionprops(label_im)
    if len(regions)==0:
        return 0
    return max([r.area for r in regions])

def load_bunny_model(ckpt_path,device):
    model = UNet(2,device,enable_classification=False)

    state_dict = torch.load(ckpt_path,map_location=device)
    model.load_state_dict(state_dict["net"])

    model.to(device)
    return model

def bunny_mask_predict(model,img,device):
    '''
        The function wrapper to predict the segmentation mask for the bunny ear given an input image and the UNet model.
    '''
    original_shape = img.shape[:2]
    img = cvtColor(img,cv2.COLOR_BGR2GRAY).astype(float)
    img = cv2.resize(img,(128,128))
    img /= np.max(img)
    output = model({'input':torch.Tensor(np.array([img]))})
    output = output.argmax(1).unsqueeze(1).to(torch.float32).to(device)
        
    output = F.interpolate(output, size=original_shape, mode='bilinear', align_corners=False)
    output = output[0,0,:,:].detach().cpu().numpy()
    output[output>0]=1
    return output

