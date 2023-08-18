#!/usr/bin/env python
import time
import cv2


def main():
	
	# define a video capture object
	vid = cv2.VideoCapture('/dev/video1')
	# set video capture size for a 1920 * 1080 capture card. 
	vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
	vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
	while True:
		try:
			ret, frame = vid.read()
			# print(image)
			cv2.imshow('Video Capture', frame)
			# waits for user to press any key
			# (this is necessary to avoid Python kernel form crashing)
			cv2.waitKey(1)
			time.sleep(0.05)
		except KeyboardInterrupt:
			cv2.destroyAllWindows()
			break

			
if __name__ == '__main__':
	main()
