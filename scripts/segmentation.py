# /home/mht/anaconda3/envs/gpyt/bin/python

from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["vit_b"](checkpoint="/home/mht/catkin_ws/src/robotic-ultrasound/data/sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)


predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_pro