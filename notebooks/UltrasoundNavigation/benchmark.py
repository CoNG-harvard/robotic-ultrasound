import yaml
import numpy as np
with open('../../config/registration_ground_truth.yaml','r') as fp: 
    truth = yaml.safe_load(fp)
    surface_center_pose = truth['surface_center_pose']
    ct_center_loc = truth['ct_center_loc']

with open('../../config/pose_in_TCP_frame.yaml','r') as f:
    poses = yaml.safe_load(f)

    camera_2_tcp = poses['camera']
    probe_2_tcp = poses['probe']

def move_to_surface_center(rtde_c):
    rtde_c.setTcp(probe_2_tcp)
    rtde_c.moveL(surface_center_pose,0.05,0.1)
    rtde_c.setTcp(np.zeros(6))
def benchmarkTargetPose(ct_target_loc):
  
    def ct2base(ct_physical_offset):
        '''
            ct_offset = (left_2_right,front_2_back,feet_2_head)
        '''
        ct_x,ct_y,ct_z = ct_physical_offset
        base_x = -ct_z
        base_y =  -ct_x
        base_z = -ct_y
        return (base_x,base_y,base_z)

    ct_offset = (np.array(ct_target_loc)-np.array(ct_center_loc))/1000
    base_offset = np.array(ct2base(ct_offset))
    base_target_pose = np.array(surface_center_pose) 
    base_target_pose[:3]+=base_offset
    return base_target_pose
