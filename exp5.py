import numpy as np
import cv2

# 1. Load the image
img = cv2.imread('image.png') # Replace with your image

# 2. Setup the empty arrays for the math
mask = np.zeros(img.shape[:2], np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 3. Define the bounding box (x, y, width, height)
rect = (5, 20, 450, 600)

# 4. Run the GrabCut algorithm
cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 5. Filter the mask (throw away background, keep foreground)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 6. Apply the mask to the original image
segmented_image = img * mask2[:, :, np.newaxis]

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Segmented Image', segmented_image)

cv2.waitKey(0)
cv2.destroyAllWindows()