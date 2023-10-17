import cv2

while(True):
  
    img_color = cv2.imread('segImg.png',cv2.IMREAD_COLOR)
    cv2.imshow('Surface contact',img_color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Waits for a keystroke
cv2.waitKey(0)  
 
# Destroys all the windows created
cv2.destroyAllwindows() 