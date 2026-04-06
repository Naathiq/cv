import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. LOAD AS GRAYSCALE (IMPORTANT)
imgL = cv2.imread("im0.png", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread("im1.png", cv2.IMREAD_GRAYSCALE)

# 2. FIX SIZE (MANDATORY)
h = min(imgL.shape[0], imgR.shape[0])
w = min(imgL.shape[1], imgR.shape[1])
imgL = imgL[:h, :w]
imgR = imgR[:h, :w]

# 3. ENSURE TYPE uint8
imgL = np.uint8(imgL)
imgR = np.uint8(imgR)

# 4. STEREO MATCHING
stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16*6, blockSize=5)

# Calculate the disparity (depth)
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# 5. DEPTH MAP CLEANUP
disparity[disparity < 0] = 0 # Remove errors
disp = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp = np.uint8(disp)

# 6. DISPLAY & SAVE
plt.imshow(disp, cmap='gray')
plt.title("Depth Map")
plt.axis('off')
plt.show()

cv2.imwrite("depth_map.png", disp)