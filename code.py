# Import packages and Initialize the camera

import cv2
import time
import numpy as np

# To access the camera, we use the method cv2.VideoCapture(0) and set the capture object as vCap.
vCap = cv2.VideoCapture(0)       

# Store a single frame before starting the infinite loop

# vCap.read() function captures frames from webcam
_, background = vCap.read()    

# 2-second delay between two captures are for adjusting camera auto exposure
time.sleep(2)
_, background = vCap.read()

# Define all the kernels size  
open_kernel = np.ones((5,5),np.uint8)
close_kernel = np.ones((7,7),np.uint8)
dilation_kernel = np.ones((10, 10), np.uint8)

# Function for remove noise from mask 
def filter_mask(mask):

# cv2.MORPH_CLOSE removes unnecessary black noise from the white region in the mask. And how much noise to remove that is defined by kernel size.
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

# cv2.MORPH_OPEN removes unnecessary white noise from the black region.
    open_mask = cv2.morphologyEx(close_mask, cv2.MORPH_OPEN, open_kernel)

#cv2.dilate increases white region in the image.
    dilation = cv2.dilate(open_mask, dilation_kernel, iterations= 1)
    return dilation

# vCap.isOpened() function checks if the camera is open or not and returns true if the camera is open and false if the camera is not open.
while vCap.isOpened():
    _, frame = vCap.read()       # Capture every frame

# Detect the cloth:

# cv2.cvtColor() function converts colorspace.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Lower bound and Upper bound are the boundaries of red color.
    lower_bound = np.array([0, 125, 50])     
    upper_bound = np.array([10, 255,255])

# cv2.inRange() function returns a segmented binary mask of the frame where the red color is present.
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Filter mask
    mask = filter_mask(mask)

# Apply the mask to take only those region from the saved background 
# where our cloak is present in the current frame
    cloak = cv2.bitwise_and(background, background, mask=mask)

# create inverse mask 
    inverse_mask = cv2.bitwise_not(mask)  

# Apply the inverse mask to take those region of the current frame where cloak is not present 
    current_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    combined = cv2.add(cloak, current_background)


    cv2.imshow("Final result", combined)


    if cv2.waitKey(1) == ord('q'):
        break
    
vCap.release()
cv2.destroyAllWindows()
