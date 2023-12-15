
import numpy as np
import SimpleITK as sitk
def upper_boundary(slice,l): 
    m = np.argwhere(slice[:,l]>=0.6)
    if len(m)>0:
        return m[0,0]
    else:
        return np.inf
    
def lower_boundary(slice,l): 
    m = np.argwhere(slice[:,l]>=0.6)
    if len(m)>0:
        return m[-1,0]
    else:
        return np.inf
def left_boundary(slice,l): 
    m = np.argwhere(slice[l,:]>=0.6)
    if len(m)>0:
        return m[0,0]
    else:
        return np.inf
    
def right_boundary(slice,l): 
    m = np.argwhere(slice[l,:]>=0.6)
    if len(m)>0:
        return m[-1,0]
    else:
        return np.inf
def body_center(body_ct):
    center = np.array(body_ct.GetSize())/2
    _,p,_ = center.astype(int)
    s,l = xy_center(body_ct)
    return l,p,s
def xy_center(body_ct):
    img = sitk.GetArrayViewFromImage(body_ct).astype(float)
    img = np.swapaxes(img,0,2)
    img = np.array(img)
    img += -np.min(img)
    img /= np.max(img)

    center = np.array(body_ct.GetSize())/2
    center = center.astype(int)
    l,p,s = center
    slice = img[:,p,:]


    ub = []
    lb = []
    LB = []
    RB = []
    for i in range(slice.shape[1]):
        b = upper_boundary(slice,i)
        u = lower_boundary(slice,i)
        if np.isfinite(u) and 50<=i<=200:
            ub.append((i,u))
        if np.isfinite(b) and 50<=i<=200:
            lb.append((i,b))

    for i in range(slice.shape[0]):
        L = left_boundary(slice,i)
        R = right_boundary(slice,i)
        if np.isfinite(L) and 130<=i<=350:
            LB.append((L,i))
        if np.isfinite(R) and 130<=i<=350:
            RB.append((R,i))

        
    ub = np.array(ub)
    lb = np.array(lb)
    _,left_right_center = np.mean(np.vstack([lb,ub]),axis=0)
    LB = np.array(LB)
    RB = np.array(RB)
    head_feet_center,_ = np.mean(np.vstack([LB,RB]),axis=0)
    return head_feet_center,left_right_center

