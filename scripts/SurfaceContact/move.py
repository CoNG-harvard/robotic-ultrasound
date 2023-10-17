import numpy as np
import yaml
def move_default_pose(rtde_c):
    with open("./config/poses.yaml","r") as f:
        d = yaml.safe_load(f)
        p = d['default_pose']
        rtde_c.moveL(p,0.05,0.1)
