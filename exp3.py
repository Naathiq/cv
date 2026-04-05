import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('image.png')

# Color space conversions
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cmy = 255 - image # Simulated CMY

# Histogram Equalization
hist_eq = cv2.equalizeHist(gray)

# Smoothing filters
blur_avg = cv2.blur(image, (5, 5))
blur_gaussian = cv2.GaussianBlur(image, (5, 5), 0)
blur_median = cv2.medianBlur(image, 5)

# Convolution: Sharpening filter
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened = cv2.filter2D(image, -1, sharpen_kernel)

# Gradient filters
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(sobelx, sobely)

# Edge detection
edges = cv2.Canny(gray, 100, 200)

# --- MATPLOTLIB DISPLAY SECTION ---
plt.figure(figsize=(14, 10))

plt.subplot(3, 5, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(3, 5, 2)
plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
plt.title('RGB')
plt.axis('off')

plt.subplot(3, 5, 3)
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB))
plt.title('HSV')
plt.axis('off')

plt.subplot(3, 5, 4)
plt.imshow(cv2.cvtColor(cmy, cv2.COLOR_BGR2RGB))
plt.title('CMY')
plt.axis('off')

plt.subplot(3, 5, 5)
plt.imshow(hist_eq, cmap='gray')
plt.title('Histogram Equalization')
plt.axis('off')

plt.subplot(3, 5, 6)
plt.imshow(cv2.cvtColor(blur_avg, cv2.COLOR_BGR2RGB))
plt.title('Avg Blur')
plt.axis('off')

plt.subplot(3, 5, 7)
plt.imshow(cv2.cvtColor(blur_gaussian, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Blur')
plt.axis('off')

plt.subplot(3, 5, 8)
plt.imshow(cv2.cvtColor(blur_median, cv2.COLOR_BGR2RGB))
plt.title('Median Blur')
plt.axis('off')

plt.subplot(3, 5, 9)
plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
plt.title('Sharpened (Convolution)')
plt.axis('off')

plt.subplot(3, 5, 10)
plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X')
plt.axis('off')

plt.subplot(3, 5, 11)
plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y')
plt.axis('off')

plt.subplot(3, 5, 12)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.subplot(3, 5, 13)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.tight_layout()
plt.show()