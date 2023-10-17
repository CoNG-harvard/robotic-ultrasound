import pyrealsense2 as rs
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation


sys.path.append("../")

from SurfaceContact.control import SurfaceContactControl

try:
    pipeline = rs.pipeline()
    pipeline.start()
    sc = SurfaceContactControl(pipeline)

    fig = plt.figure(dpi=100,figsize = (12,4))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    def animate(i):
        ax1.clear()
        ax2.clear()
        sc.mainloop()
        sc.showScene([ax1,ax2])
        

    ani = animation.FuncAnimation(fig, animate,frames=40, interval=100)
    plt.show()
finally:
    pipeline.stop()