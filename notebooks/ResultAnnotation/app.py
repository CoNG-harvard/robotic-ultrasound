import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import cv2
from cv2 import cvtColor
import pickle as pkl
def to_grayscale(img):
    original_shape = img.shape[:2]
    img = cvtColor(img,cv2.COLOR_BGR2GRAY).astype(float)
    img /= np.max(img)
    return img
def centroid3(img):
    nx, ny, nz = img.shape
    # print(nx,ny,nz)
    imgx = np.sum(np.sum(img, axis=1), axis=1)
    imgy = np.sum(np.sum(img, axis=2), axis=0)
    imgz = np.sum(np.sum(img, axis=0), axis=0)
    denom = np.sum(np.sum(np.sum(img, axis=0), axis=0), axis=0)
    cx = np.sum(np.linspace(0, nx-1, nx)*imgx)/denom
    cy = np.sum(np.linspace(0, ny-1, ny)*imgy)/denom
    cz = np.sum(np.linspace(0, nz-1, nz)*imgz)/denom
    
    return cx, cy, cz

def plot_img_at(img, coord, label='Target Location',
    vmin = 0.4,
    vmax = 0.8,ax=None):
    '''
        Plot the 3D slicing view of img at coord.
    '''
    img = np.array(img)
    img += -np.min(img)
    img /= np.max(img)
   
    l,p,s = coord
    
    
    target_marker = 'x'
    target_size = 50
    target_color = 'yellow'

  
    if ax is None:
        ax = plt.gca()
    ax.axis('off')
    
    # Only plot the horizontal slice.
    ax.imshow(np.squeeze(img[:,:,s]).T,cmap='gray',vmin=vmin,vmax=vmax)
    ax.scatter(l,p,marker = target_marker,s=target_size,color = target_color,label=label)
    ax.legend()    
    return ax
def visualize_body(img,target_pixel=None,label=None,
                    vmin = 0.4,
                    vmax = 0.8,ax=None):
    img = sitk.GetArrayViewFromImage(img).astype(float)
    img = np.swapaxes(img,0,2)
    if target_pixel is None:
        target_pixel = np.array(centroid3(img))
        label = 'Centroid location'
    plot_img_at(img,np.array(target_pixel,dtype=int),label,vmin,vmax,ax)
    
    
def flip_img(input_img,orders):
    flipped_ct = sitk.Flip(input_img,orders)
    flipped_ct = sitk.GetImageFromArray(sitk.GetArrayFromImage(flipped_ct).astype(np.float64))
    flipped_ct.SetOrigin(input_img.GetOrigin())
    flipped_ct.SetSpacing(input_img.GetSpacing())
    return flipped_ct
  
class ResultAnnotationApp(tk.Tk):
    def __init__(self,fs,body_ct):
        
        super().__init__()
        self.curr_obs_id = 0
        self.fs = fs
        self.body_ct = body_ct
        # Bind keypress event to handle_keypress()
        self.bind("<Key>", self.handle_keypress)

        self.combo_label = tk.Label(master=self,text='Select observation from dropdown:')
        self.combo_label.pack()

        self.combo = ttk.Combobox(
            state="readonly",
            values=list(range(len(fs))),
            master=self,
    )
        self.combo.bind("<<ComboboxSelected>>", self.observation_selected)
        self.combo.pack()

        # the figure that will contain the plot 
        self.fig = Figure(figsize = (15, 6), 
                        dpi = 100) 

        # creating the Tkinter canvas 
        # containing the Matplotlib figure 
        self.canvas = FigureCanvasTkAgg(self.fig, 
                                    master = self)   
        self.canvas.draw() 

        # placing the canvas on the Tkinter window 
        self.canvas.get_tk_widget().pack() 

        # creating the Matplotlib toolbar 
        self.toolbar = NavigationToolbar2Tk(self.canvas, 
                                        self) 
        self.toolbar.update() 

        # placing the toolbar on the Tkinter window 
        self.canvas.get_tk_widget().pack() 
        # setting the title  
        self.title('Plotting in Tkinter') 
        
        # dimensions of the main window 
        self.geometry("500x500") 
        self.displayObs()

    def displayObs(self):
        self.combo.set(self.curr_obs_id)
        self.plot(self.curr_obs_id)
    # plot function is created for  
    # plotting the graph in  
    # tkinter window 
    def plot(self,i): 
        f = self.fs[i]

        with open(obs_path+f,'rb') as fp:
            obs = pkl.load(fp)
        
        loc = obs['ct_target_loc']
        # cframe = obs['with_slice_matching']['center_frame']
        
        pix = self.body_ct.TransformPhysicalPointToIndex(loc)
        # visualize_vessel(original_vessel_ct,pix,'Target Location')
        ax = self.fig.add_subplot(1,1,1)
        visualize_body(body_ct,pix,'Target Location',vmin=0.6,vmax=0.8,ax=ax)
        self.canvas.draw() 
    
        self.toolbar.update() 


    def observation_selected(self,event):
        obs_id = int(self.combo.get())
        self.curr_obs_id = obs_id
        self.displayObs()

    def handle_keypress(self,event):
        if event.keysym == 'Right':
            self.curr_obs_id = min(len(self.fs)-1,self.curr_obs_id+1)
        elif event.keysym == 'Left':
            self.curr_obs_id = max(0,self.curr_obs_id-1)

        self.displayObs()


import os
obs_path = './data/observations/'
fs = os.listdir(obs_path)


body_ct = sitk.ReadImage('./data/nifty/CT_phantom_regular.nii.gz')
body_ct = flip_img(body_ct,[True,False,False])

# the main Tkinter window 
window = ResultAnnotationApp(fs,body_ct)

# run the gui 
window.mainloop() 