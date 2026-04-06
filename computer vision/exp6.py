import numpy as np
import cv2 as cv

# 1. Math rules to tell the sub-pixel algorithm when to stop calculating
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 2. Prepare the "perfect" 3D real-world points (0,0,0), (1,0,0), etc.
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Lists to store the points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in the image

# 3. Load the image and convert to grayscale
img = cv.imread('chessbd.jpg') # Replace with your image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 4. Find the chessboard corners
ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

# 5. If it successfully found the corners (ret == True)
if ret == True:
    objpoints.append(objp) # Save the 3D points
    
    # Refine the 2D corners to sub-pixel accuracy
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2) # Save the refined 2D points
    
    # Draw the colorful lines connecting the corners on the image
    cv.drawChessboardCorners(img, (7, 6), corners2, ret)

# Display the result
cv.imshow('calibrated image', img)
cv.waitKey(0)
cv.destroyAllWindows()