import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
from tqdm import trange

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
   
def plot_slice(ploting_ax, img,slice,axis):
    '''
        Plot img along a specific axis.
    '''

    if axis==0:
        ploting_ax.imshow(img[slice],cmap = 'gray')
    elif axis==1:
        ploting_ax.imshow(img[:,slice,:],cmap = 'gray')
    elif axis==2:
        ploting_ax.imshow(img[:,:,slice], cmap = 'gray')
    return ploting_ax

def visualize_vessel(vessel_img,target_pixel=None,label=None,
                    vmin = 0.4,
                    vmax = 0.8):
    img = sitk.GetArrayViewFromImage(vessel_img).astype(float)
    img = np.swapaxes(img,0,2)
    if target_pixel is None:
        target_pixel = np.array(centroid3(img))
        label = 'Centroid location'
    plt.figure(figsize=(15,6))
    plot_img_at(img,np.array(target_pixel,dtype=int),label,vmin,vmax)
    plt.show()

def plot_img_at(img, coord, label='Target Location',
    vmin = 0.4,
    vmax = 0.8):
    '''
        Plot the 3D slicing view of img at coord.
    '''
    img = np.array(img)
    img += -np.min(img)
    img /= np.max(img)
    print(np.max(img),np.min(img))

    l,p,s = coord
    
    axis_label ={"L":"L: left to right",
                 "P":"P: front to back",
                 "S":"S: feet to head"}
    
    target_marker = 'x'
    target_size = 50
    target_color = 'yellow'

    ax = plt.subplot(1, 3, 1)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel(axis_label["P"])
    ax.set_ylabel(axis_label["S"])
    # ax.invert_yaxis()

    ax.imshow(np.squeeze(img[l,:,:]).T,cmap='gray',vmin=vmin,vmax=vmax)
    ax.scatter(p,s,marker = target_marker,s=target_size,color = target_color)
    
    ax = plt.subplot(1, 3, 2)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel(axis_label["L"])
    ax.set_ylabel(axis_label["S"])
    # ax.invert_yaxis()

    
    ax.imshow(np.squeeze(img[:,p,:]).T,cmap='gray',vmin=vmin,vmax=vmax)
    ax.scatter(l,s,marker = target_marker,s=target_size,color = target_color)
    
    

    ax = plt.subplot(1, 3, 3)
    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel(axis_label["L"])
    ax.set_ylabel(axis_label["P"])
    # ax.invert_yaxis()


    ax.imshow(np.squeeze(img[:,:,s]).T,cmap='gray',vmin=vmin,vmax=vmax)
    ax.scatter(l,p,marker = target_marker,s=target_size,color = target_color,label=label)
    ax.legend()

    
    return ax

def central_normalize_img(input_img,n_side = 200):
    '''
        input_img: a sitk.Image object. 
        
        Crop and centralize the non-zero voxels of the image.
    '''
    img = sitk.GetArrayViewFromImage(input_img)
    img = np.swapaxes(img,0,2)
    centroid_pixel = centroid3(img)
    centroid_physical = input_img.TransformContinuousIndexToPhysicalPoint(list(centroid_pixel))

    ref_dx, ref_dy, ref_dz = 1.0, 1.0, 1.0
    ref_nx, ref_ny, ref_nz = n_side, n_side, n_side # Keep the grid size smaller than 150x150x150 for efficient computation
    

    output_spacing = [ref_dx,ref_dy,ref_dz]

    output_origin = centroid_physical-np.array(output_spacing)*np.array([ref_nx, ref_ny, ref_nz])/2 


    # The number of voxels in the output
    output_size = [ref_nx,ref_ny,ref_nz]

    return normalize_img(input_img, output_origin, output_spacing, output_size)

def normalize_img(input_img, output_origin, output_spacing, output_size):
    '''
        input_img: a sitk.Image object. 

        output_origin: float vector. The physical point coordinate corresponding to index (0,0,0) of the output image.
        output_spacing: float vector. The physical distance between adjacent voxels of the output image.
        output_size: int vector. The number of voxels in each dimension of the output image.
    '''
    identity_tf = sitk.Euler3DTransform() # We only crop and change the resolution of the image. No spatial transform is needed.

    output = sitk.Resample(input_img,output_size, identity_tf, sitk.sitkLinear,output_origin,output_spacing)
    return output


def get_centroid_loc(vessel_img):
    img = sitk.GetArrayViewFromImage(vessel_img)
    img = np.swapaxes(img,0,2)
    target_pixel = np.array(centroid3(img) + np.array([0,0,0]))
    return np.array(vessel_img.TransformContinuousIndexToPhysicalPoint(target_pixel))

def calculate_ct2us_transform(vessel_us,vessel_ct):
    us_centroid_physical = get_centroid_loc(vessel_us)
    ct_centroid_physical = get_centroid_loc(vessel_ct)
    centroid_offset = sitk.TranslationTransform(3, us_centroid_physical-ct_centroid_physical)
    shifted_us = sitk.Resample(vessel_us,vessel_ct, centroid_offset)
    shifted_us = central_normalize_img(shifted_us,150)

    sitk.WriteImage(shifted_us,'shifted_us.nii.gz')

    ## https://simpleitk.org/SPIE2019_COURSE/04_basic_registration.html

    # Some type casting is needed on the vessel_ct for the registration code to work
    fixed_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(vessel_ct).astype(np.float64))
    fixed_image.SetOrigin(vessel_ct.GetOrigin())
    fixed_image.SetSpacing(vessel_ct.GetSpacing())
    sitk.WriteImage(fixed_image,'resampled_CT.nii.gz')


    moving_image = shifted_us

    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                        moving_image, 
                                                        sitk.Euler3DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    num_iter = 10
    print('Start optimizing the transformation')
    for ii in trange(num_iter):
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=20)
        registration_method.SetMetricSamplingStrategy(registration_method.REGULAR )
        registration_method.SetMetricSamplingPercentage(0.2)    
        registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=30, convergenceMinimumValue=1e-7, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        
        # Don't optimize in-place. We want to run this cell multiple times.
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        
        rigid_transform = registration_method.Execute(fixed_image, moving_image)

    moving_reg = sitk.Resample(moving_image,fixed_image, rigid_transform)
    sitk.WriteImage(moving_reg,'vessel_reg.nii.gz')
    CT2US = sitk.CompositeTransform(centroid_offset)
    CT2US.AddTransform(rigid_transform)
    return CT2US

def get_centroid_loc(vessel_img):
    img = sitk.GetArrayViewFromImage(vessel_img)
    img = np.swapaxes(img,0,2)
    target_pixel = np.array(centroid3(img))
    return np.array(vessel_img.TransformContinuousIndexToPhysicalPoint(target_pixel))
    