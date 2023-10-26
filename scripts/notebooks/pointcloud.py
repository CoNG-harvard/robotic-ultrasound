# %%
import pyrealsense2 as rs
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("../")

from SurfaceContact.control import SurfaceContactControl



# %%
pipeline = rs.pipeline()
pipeline.start()


# %%
frames = pipeline.wait_for_frames()
rgb = frames.get_color_frame()
rgb_img = np.asanyarray(rgb.get_data())
depth = frames.get_depth_frame()

# %%
pc = rs.pointcloud()

# %%
points = pc.calculate(depth)

# %%
v = points.get_vertices()
verts = np.asanyarray(v).view(np.float32).reshape(-1, 3) 

# %%
verts

# %%
depth_img = np.asanyarray(depth.get_data())

# %%
from mpl_toolkits.mplot3d import proj3d
x = verts[:,0]
y = verts[:,1]
z = verts[:,2]
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)
plt.show()

# %%
plt.imshow(depth_img)

# %%
plt.imshow(rgb_img)

# %%
pipeline.stop()


