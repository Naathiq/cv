import cv2 as cv
import numpy as np

# 1. Load Image
img_path = r'C:\Users\acer\Desktop\computer vision\image.png'  # Replace with your actual image path
image = cv.imread(img_path)
cv.imshow('Original Image', image)

# 2. Crop Image
cropped_img = image[50:300, 100:400]
cv.imshow('Cropped Image', cropped_img)

# 3. Resize Image
resized_img = cv.resize(image, (400, 400))
cv.imshow('Resized Image', resized_img)

# 4. Thresholding (Grayscale -> Black & White)
gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
_, thresh_img = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY)
cv.imshow('Thresholded Image', thresh_img)

# 5. Contour Analysis
contours, _ = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contour_img = image.copy()
cv.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
cv.imshow('Contour Analysis', contour_img)

# 6. Blob detection
params = cv.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 30       # Minimum area of blobs to be detected
params.maxArea = 5000     # Maximum area of blobs to be detected
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.minThreshold = 10
params.maxThreshold = 200

# Create detector and find keypoints
detector = cv.SimpleBlobDetector_create(params)
keypoints = detector.detect(thresh_img)

# Draw red circles around the blobs
blob_img = cv.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('Blob Detection', blob_img)

# Wait for any key press, then close all windows
cv.waitKey(0)
cv.destroyAllWindows()