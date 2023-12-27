import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import pickle as pkl
from functools import partial

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

class US_frame:
    def __init__(self,master,status_changed_command):
        self.frame = tk.Frame(master)
        self.fig = Figure(figsize=(1,2))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig,master=self.frame)
        self.canvas.get_tk_widget().pack()
        self.status = tk.IntVar(self.frame,0)
        self.status_list = {"Not match":0, "Match":1}
        self.sbs = []
        for key,val in self.status_list.items():
             button = tk.Radiobutton(master = self.frame,
                                     text = key,
                                     variable=self.status,
                                     value= val, command=status_changed_command).pack()
             self.sbs.append(button)
            
  
class ResultAnnotationApp(tk.Tk):
    def __init__(self,fs,body_ct):
        
        super().__init__()

        self.fs = fs
        self.body_ct = body_ct

        self.curr_obs_id = 0
        
        self.curr_obs = self.load_curr_obs()
        self.save_curr_obs()

        # Bind keypress event to handle_keypress()
        self.bind("<Key>", self.handle_keypress)

        self.combo_label = tk.Label(master=self,text='Select observation from dropdown \n or press left/right arrow keys for different target locations.').grid(row=0,column=0,columnspan=2,rowspan=1)
        # self.combo_label.pack()

        self.combo = ttk.Combobox(
            state="readonly",
            values=list(range(len(fs))),
            master=self,
    )
        self.combo.grid(row = 0,column=2,columnspan=2,rowspan=1)
        self.combo.bind("<<ComboboxSelected>>", self.observation_selected)

        self.not_usable = tk.BooleanVar(self,False)
        self.not_usable_botton = ttk.Checkbutton(self,text='Observations for target location not usable',
                                                variable=self.not_usable,onvalue=True, offvalue=False,command=self.notusable_clicked).grid(row = 1, column=0,columnspan=4,rowspan=1)
        # self.combo.pack()

        # the figure that will contain the plot 
        self.fig = Figure(figsize=(4,4)) 

        # creating the Tkinter canvas 
        # containing the Matplotlib figure 
        self.canvas = FigureCanvasTkAgg(self.fig, 
                                    master = self) 
        self.canvas.get_tk_widget().grid(row=2,column=0,columnspan=4,rowspan=4)

        self.US_frames = []
        for i in range(20):
            self.US_frames.append(self.create_us_frame(i))
            self.US_frames[i].frame.grid(row = 3*(i//7),column = 5+i%7,rowspan=3)

        # setting the title  
        self.title('Plotting in Tkinter') 
        
        # dimensions of the main window 
        self.geometry("2000x2000") 
        self.displayObs()
       
    def notusable_clicked(self):
        
        self.curr_obs['not_usable'] = self.not_usable.get()

    def load_curr_obs(self):
        f = self.fs[self.curr_obs_id]

        print('Loading obs {}'.format(self.curr_obs_id),f)

        with open(obs_path+f,'rb') as fp:
            obs = pkl.load(fp)
        if 'match_status' not in obs['with_slice_matching'].keys():
            obs['with_slice_matching']['match_status'] = [0] * len(obs['with_slice_matching']['all_poses'])
        
        if 'not_usable' not in obs.keys():
            obs['not_usable'] = False
        
        return obs
    

    def save_curr_obs(self):
        print("saving curr_obs")
        f = self.fs[self.curr_obs_id]
        with open(obs_path+f,'wb') as fp:
            pkl.dump(self.curr_obs, fp)
        return 0
    
    def create_us_frame(self,i):
        return US_frame(self,partial(self.us_match_status_changed,i=i))
    
    def us_match_status_changed(self,i):
        self.curr_obs['with_slice_matching']['match_status'][i] = self.US_frames[i].status.get()

    def displayObs(self):
        self.combo.set(self.curr_obs_id)
        self.not_usable.set(self.curr_obs['not_usable'])
        
        self.plot_groundtruth(self.curr_obs)
        self.draw_us(self.curr_obs)
        

    
    # plot function is created for  
    # plotting the graph in  
    # tkinter window 
    def plot_groundtruth(self,obs): 
        
        loc = obs['ct_target_loc']
        # cframe = obs['with_slice_matching']['center_frame']
        
        pix = self.body_ct.TransformPhysicalPointToIndex(loc)
        print(pix,loc)
        ax = self.fig.gca()
        ax.clear()
        visualize_body(body_ct,pix,'Target Location',vmin=0.6,vmax=0.8,ax=ax)
        self.canvas.draw() 
    

    def draw_us(self,obs):
        for i in range(len(self.US_frames)):
            ax= self.US_frames[i].fig.gca()
            ax.clear()

        frames = obs['with_slice_matching']['all_frames']
        ss = obs['with_slice_matching']['match_status']
        
        for i in range(min(len(frames),len(self.US_frames))):
            self.US_frames[i].status.set(ss[i])
            ax = self.US_frames[i].fig.gca()
            ax.imshow(frames[i])
            ax.axis('off')
            self.US_frames[i].fig.tight_layout()
            
            self.US_frames[i].canvas.draw()

    def observation_selected(self,event):
        self.save_curr_obs()

        new_obs_id = int(self.combo.get())
        self.curr_obs_id = new_obs_id
      
        self.curr_obs = self.load_curr_obs()
        self.displayObs()

    def handle_keypress(self,event):
        self.save_curr_obs()

        if event.keysym == 'Right':
            self.curr_obs_id = min(len(self.fs)-1,self.curr_obs_id+1)
        elif event.keysym == 'Left':
            self.curr_obs_id = max(0,self.curr_obs_id-1)

        self.curr_obs = self.load_curr_obs()  
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