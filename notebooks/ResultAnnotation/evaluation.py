import numpy as np
def min_match_dist(obs):
    '''
        Return the minimal distance from the center_pose where a matched frame is recorded.

        e.g., if the returned value is 0.005, then it means within 0.005m from obs['with_slice_matching']['center_pose'], a US frame that matches the CT image at obs['ct_target_loc'] is found.
    '''
    ps = obs['with_slice_matching']['all_poses']
    cp = obs['with_slice_matching']['center_pose']
    match = obs['with_slice_matching']['match_status']
    match_poses = np.array(ps)[np.array(match)==1]
    if len(match_poses)==0:
        return np.inf
    return np.min(np.abs((match_poses-cp)[:,0]))
