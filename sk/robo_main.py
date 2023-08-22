# Sekeun Kim 
# cleaned code @ 08152023
# --------------------------------------------------------
import numpy as np
import os
import pickle as pkl
import torch
import torch.backends.cudnn as cudnn
import sys
os.environ['CUDA_VISIBLE_DEVICES']='0'
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from socket_helper import *
import cv2

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)
    return data

def run():
    cudnn.benchmark = True
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""""""""""""""""""""""""""""INERENCE"""""""""""""""""""""""""""""""""""""""
    # socket_helper = Socket_Hepler_Host()
    
    # while True:
    #     try: 
    #         data = socket_helper.activate_server()
    #         if data is not None:
    #             break
    #         else:
    #             pass
    #     except KeyboardInterrupt:
    #         break
        
    data = load_data('/mount/home/local/PARTNERS/sk1064/workspace/control/data_retrieve/data_0.pkl')   
     
    with torch.no_grad():             
        with torch.cuda.amp.autocast():
            sam_checkpoint = "/mount/home/local/PARTNERS/sk1064/workspace/control/sam_t/segment_anything/model/ckpt/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            
            device = "cuda"

            sam = sam_model_registry[model_type]( checkpoint=sam_checkpoint)
            sam.to(device=device)
            
            print ("LOAD DONE")
            
    predictor = SamPredictor(sam)

    img = cv2.cvtColor(data["image"], cv2.COLOR_BGR2RGB)
    
    predictor.set_image(img)
    sh = np.shape(img)
    
    input_point = np.array([sh[0]//2, sh[1]//2])
    
    input_point = np.array([[500, 375], [350, 625]])
    input_label = np.array([1, 0])
    
    masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
    )

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    logits[0,:,:]
    masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    mask_input=mask_input[None, :, :],
    multimask_output=False,
    )

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(masks, plt.gca())
    plt.axis('off')
    plt.show() 
    plt.savefig('test_.jpg')
    
    ###############################
    ##### METHOD OPTIMIZATION #####
    ###############################
    ###############################
    
    socket_helper = Socket_Hepler_Host()
    socket_helper.activate_server()
    
    while True:
        try: 
            socket_helper.Transfer_action(np.random())
            pass
        except KeyboardInterrupt:
            break
                            
if __name__ == '__main__':
    run()
    
    
