import numpy as np
def precision(hp,hg):
    '''
        hp: binary array. The predicted segmentation.
        hg: binary array. The ground truth segmentation.

        Output: the precision value given prediction hp against ground truth hg
    '''
    denom = np.sum(hp)
    return np.sum(np.logical_and(hp,hg))/denom if denom>0 else 0

def recall(hp,hg):
    '''
        hp: binary array. The predicted segmentation.
        hg: binary array. The ground truth segmentation.

        Output: the recall value given prediction hp against ground truth hg
    '''

    denom = np.sum(hg)
    return np.sum(np.logical_and(hp,hg))/denom if denom>0 else 0

def dice(hp,hg):
    '''
        hp: binary array. The predicted segmentation.
        hg: binary array. The ground truth segmentation.

        Output: the dice value given prediction hp against ground truth hg
    '''
    denom = np.sum(hg)+np.sum(hp)
    return 2*np.sum(np.logical_and(hp,hg))/denom if denom>0 else 0

