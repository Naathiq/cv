import cv2

# 1. Load the image
img = cv2.imread("image.png") # Replace with your group photo

# Resize it to make detection a bit easier (optional but good practice)
img = cv2.resize(img, None, fx=1.5, fy=1.5)

# 2. Preprocess the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# 3. Load the Haar Cascade Rulebook
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 4. Find the faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(30, 30)
)

print("Faces detected:", len(faces))

# 5. Draw a green rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
cv2.imshow("Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()