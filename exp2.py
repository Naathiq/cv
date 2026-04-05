import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image and convert BGR to RGB for matplotlib
image = cv2.imread("image.png")  # Replace with your image name
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Draw a Line (Green)
imageline = img_rgb.copy()
cv2.line(imageline, (75, 100), (150, 150), (0, 255, 0), 3)

# 3. Draw a Circle (Red)
imagecircle = img_rgb.copy()
cv2.circle(imagecircle, (75, 75), 25, (255, 0, 0), 3) 

# 4. Draw a Rectangle (Green)
imagerect = img_rgb.copy()
cv2.rectangle(imagerect, (100, 100), (200, 200), (0, 255, 0), 3)

# 5. Draw an Ellipse (Red)
image_ellip = img_rgb.copy()
cv2.ellipse(image_ellip, (100, 70), (75, 10), 0, 0, 360, (255, 0, 0), 3)

# 6. Add Text (Blue)
image_text = img_rgb.copy()
cv2.putText(image_text, 'DOG', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

# --- Matplotlib Display Setup ---
titles = ['Original Image', 'Line', 'Circle', 'Rectangle', 'Ellipse', 'Text']
images = [img_rgb, imageline, imagecircle, imagerect, image_ellip, image_text]

plt.figure(figsize=(15, 10))

# Loop through and plot all 6 images
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off') # Hides the axis numbers

plt.tight_layout()
plt.show()