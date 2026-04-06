import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Images
img1 = cv2.imread('image.png')
img2 = cv2.imread('image.png')

# Convert to RGB (VERY IMPORTANT for matplotlib)
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# 1. CLONING
cloned_image = np.copy(img1_rgb)

# 2. FOURIER TRANSFORM
f = np.fft.fft2(gray_img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# 3. HOUGH LINES
edges = cv2.Canny(gray_img, 50, 150)
line_img = img1_rgb.copy()
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 4. HOUGH CIRCLES
circle_img = img1_rgb.copy()
blurred = cv2.GaussianBlur(gray_img, (9, 9), 2)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                           param1=50, param2=30, minRadius=10, maxRadius=100)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for c in circles[0, :]:
        cv2.circle(circle_img, (c[0], c[1]), c[2], (0, 255, 0), 2)

# 5. ORB KEYPOINTS
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

orb_img = cv2.drawKeypoints(img1_rgb, kp1, None, color=(0,255,0))

# 6. FEATURE MATCHING
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img_matches = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2,
                             matches[:20], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 7. IMAGE ALIGNMENT
src_pts = np.float32([[50, 60], [100, 40], [80, 120], [150, 100]]).reshape(-1, 1, 2)
dst_pts = np.float32([[55, 65], [110, 50], [90, 130], [160, 110]]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h, w = img1.shape[:2]
img_aligned = cv2.warpPerspective(img1_rgb, M, (w, h))

# ---------------- DISPLAY USING MATPLOTLIB ----------------

plt.figure(figsize=(12, 10))

plt.subplot(3, 3, 1)
plt.imshow(img1_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(cloned_image)
plt.title("Cloned Image")
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Fourier Transform")
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(edges, cmap='gray')
plt.title("Edges")
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(line_img)
plt.title("Hough Lines")
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(circle_img)
plt.title("Hough Circles")
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(orb_img)
plt.title("ORB Keypoints")
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(img_matches)
plt.title("Feature Matching")
plt.axis('off')

plt.subplot(3, 3, 9)
plt.imshow(img_aligned)
plt.title("Aligned Image")
plt.axis('off')

plt.tight_layout()
plt.show()