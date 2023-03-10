import imghdr
import cv2
import numpy as np
import os

# ------OVERVIEW OF CODE-------
# 1) Procure files, iterate files function
# 2) Create a cropping mask of 256x256
# 3) Create a circle mask with radius 16- sqrt(256)
# 4) Pass image input with above as a binary mask
# 5) Display code

# ---------------------*1*-------------------------
# Procure files, iterate files function
# This loops through the directory and adds the file names to the list

dirnames = ''
for dirname in dirnames:
    # This loops through the files in the directory
    for filename in os.listdir(dirname):
        file = open(filename)
        # TODO: this will iterate file functions
        pass

# create a file name iterator:
# idea 1- try os file calling function, list all files, avoid pro?

# This is the directory where the images are stored

image = cv2.imread('20120101073000.raw.jpg')
print(image.shape)  # Print image shape
cv2.imshow("original", image)

# ---------------------*2*-------------------------

# Prepare crop area
x, y = 20, 49
xe, ye = 276, 305

# Crop image to specified area using slicing
cropped_image = image[y:ye, x:xe]
cv2.imwrite("Cropped Image.jpg", cropped_image)
image2 = cv2.imread('Cropped Image.jpg')
# ---------------------*3*-------------------------

# display shape of image
# print(image.shape)
print("break1")
print(image2.shape)
print("break1")
print(cropped_image)

# Load image, create mask, and draw white circle on mask
mask = np.zeros(image2.shape, dtype=np.uint8)
mask = cv2.circle(mask, (128, 128), 128, (255, 255, 255), -1)

# ---------------------*4*-------------------------

# Mask input image with binary mask
result = cv2.bitwise_and(image2, mask)
# Color background white
result[mask == 0] = 255  # Optional

# ---------------------*5*-------------------------

# Show cropped image
cv2.imshow("cropped", cropped_image)

# Show masked image
cv2.imshow("croppeddisp", image2)

# Show mask image
cv2.imshow('mask', mask)

# Show final result of binary mask
cv2.imshow('result', result)

# Wait for key press, then close all windows
cv2.waitKey()
