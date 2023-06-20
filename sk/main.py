from utils.sftp import Sftp_Helper

# acqusition of information
# output : pos, pre, img


# save as json


# send info -> gpu 
sftp_helper = Sftp_Helper(host = 'emimdgxa100gpu3.ccds.io')
sftp_helper.Transfer_data(source_path = './test.png', dest_path = '/home/local/PARTNERS/sk1064/workspace/test.png' )