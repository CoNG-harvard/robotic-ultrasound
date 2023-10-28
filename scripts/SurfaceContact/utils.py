
def patch_pixel_indx(cx,cy,h,w,patch_pixel_rad):
    '''
        Return the pixel indices that is within patch_pixel_rad pixel distance from the pixel (cy,cx).

        It's important to note here that cy is the row index, cx  is in the column index.
    '''


    xlo,xhi = max(0,cx-patch_pixel_rad),min(w,cx+patch_pixel_rad)
    ylo,yhi = max(0,cy-patch_pixel_rad),min(h,cy+patch_pixel_rad)

    patch_indx = []
    for i in range(ylo,yhi):
        for j in range(xlo,xhi):
            if (i-cy)**2+(j-cx)**2<=patch_pixel_rad**2:
                patch_indx.append((i,j))    
    return patch_indx
