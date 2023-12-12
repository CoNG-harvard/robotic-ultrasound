import torch
from torch import nn

import SimpleITK as sitk
import numpy as np

def normalize_vessel_slice(input_img):
    output_spacing = np.array([1.0,1.0])
    output_size = (np.array(input_img.GetSize())*np.array(input_img.GetSpacing())/output_spacing).astype(np.uint32).tolist()
    output_origin = input_img.GetOrigin()

    identity_tf = sitk.Euler2DTransform() # We only crop and change the resolution of the image. No spatial transform is needed.
    return sitk.Resample(input_img,output_size, identity_tf, sitk.sitkLinear,output_origin,output_spacing)

def vessel_2D_match(fixed,moving):
    '''
        Inputs:
            fixed: binary matrix. The slice of vessel image from CT.
            moving: binary matrix. The vessel image inferred from a US image.

        Outputs:
            pos: the position within [fixed] where maximal convolution product with [moving] is achieved.
            am: the value of that maximal convolution product.
            matched_area: the pixels within [fixed] with its upper-left corner being [pos] and shape being the same as [moving].
    '''

    with torch.no_grad():
        K = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=moving.shape)
        K.weight.data = torch.Tensor(torch.Tensor(moving).reshape(1,1,*moving.shape))
        mask = K(torch.Tensor(fixed).reshape(1,1,*fixed.shape))
        mask = np.array(mask.data)

    mask = mask.reshape(mask.shape[-2:])
    am = np.max(mask)
    pos = np.argwhere(mask==am)[0]
    matched_area=fixed[pos[0]:pos[0]+moving.shape[0],pos[1]:pos[1]+moving.shape[1]]
    return  pos, am, matched_area


def local_match(vessel_ct_slice,vessel_us_slice, visualize = False):
    '''
        vessel_ct_slice: Binary mask SITK Image. A slice of CT image.
        vessel_us_slice: Binary mask SITK Image. The vessel pixels within an ultrasound image. 
        
        Output:
 
            pos: the position within the CT slice that matches the US slice.
            am: the value of the dot product between the CT slice and the US slice at [pos].
            matched_area: the sub image in CT slice matching the US slice.
    '''

    fixed = sitk.GetArrayFromImage(normalize_vessel_slice(vessel_ct_slice))
    moving = sitk.GetArrayFromImage(normalize_vessel_slice(vessel_us_slice))

    pos,am,matched = vessel_2D_match(fixed, moving)
    # if visualize:

    #     ax = plt.subplot(1,3,1)
    #     ax.imshow(fixed.T,cmap='gray')

    #     l = pos[0]+matched.shape[0]//2
    #     t = pos[1]
    #     draw_us_box(ax,l,t,*matched.shape)

    #     plt.subplot(1,3,2)
    #     plt.imshow(matched.T,cmap='gray')
    #     plt.title('Matching area in vessel_ct')
    #     plt.subplot(1,3,3)
    #     plt.imshow(moving.T,cmap='gray')
    #     plt.title('vessel_us')
    #     plt.show()

    pos = pos/np.array(fixed.shape) * np.array(vessel_ct_slice.GetSize()[::-1])
    pos = pos[::-1]
    return pos, am, matched.T