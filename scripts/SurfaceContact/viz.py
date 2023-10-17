from numpy import linalg as la
from .SurfaceContact.scene import extractBodyPixels,bodyCentroid

def showScene(rgb_img):
    plt.figure(dpi=100,figsize = (12,7.5))
    plt.subplot(2,2,1)
    plt.imshow(rgb_img)

    scene = rgb_img
    mask =  extractBodyPixels(scene)
    result = cv2.bitwise_and(scene,scene,mask = mask)
    plt.subplot(2,2,3)
    plt.imshow(mask,cmap = 'gray')

    plt.subplot(2,2,4)
    plt.imshow(result)


    plt.subplot(2,2,2)
    target_loc = bodyCentroid(mask)
    target_loc = np.array(target_loc,dtype = np.int)

    offset = target_loc-center
    plt.scatter(target_loc[0],target_loc[1],marker="x",color = 'yellow',label='target',s=100)
    plt.scatter(center[0],center[1],marker="+",color = 'white',label='center',s = 150)

    arrow_width = 5
    head_length = 4.5*arrow_width
    plt.arrow(*center,
            *(offset-head_length*offset/la.norm(offset)),
            color = 'red',width = arrow_width,head_length = head_length,
            label = 'Moving direction')

    plt.imshow(result)
    plt.legend()
    plt.show()