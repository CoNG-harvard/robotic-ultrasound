from slice_matching  import match
import numpy as np
def segmentation_scores(vessel_ct_slice,vessel_us_slice):
    '''
        vessel_ct_slice: a binary SITK Image. Ground truth segmentation.
        vessel_us_slice: a binary SITK Image. Predicted segmentation.
    '''
    pos,am,Y,Yt = match(vessel_ct_slice,vessel_us_slice,padding=True, centralize = False)

    Y = np.sum(Y)
    Yt = np.sum(Yt)

    opt_mov_precision = am / Y
    opt_mov_recall = am / Yt
    opt_mov_dice = 2 * am /(Yt+Y)

    return opt_mov_precision,opt_mov_recall, opt_mov_dice, pos