# %%
%matplotlib widget
import numpy as np
import sys
import matplotlib.pyplot as plt


# %%
verts = np.load('pointcloud.npy')


# %%
x = verts[:,0]
y = verts[:,1]
z = verts[:,2]
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)
plt.show()



