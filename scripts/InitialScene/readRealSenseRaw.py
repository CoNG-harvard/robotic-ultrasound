import numpy as np 
with open("InitialSceneDepth_Depth.raw","rb") as f:
      array = np.fromfile(f, dtype=np.uint16) 
array=array.reshape(240,-1)
# from matplotlib import pyplot as plt
# plt.imshow(array)


