import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("video.mp4")

# Read first frame
ret, frame = cap.read()

# Select object
x, y, w, h = cv2.selectROI(frame)

# Selected object
roi = frame[y:y+h, x:x+w]

# Convert to HSV
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Create histogram
hist = cv2.calcHist([hsv], [0], None, [180], [0,180])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

# Track object
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Find matching color
    back = cv2.calcBackProject([hsv], [0], hist, [0,180], 1)

    # Track movement
    _, (x,y,w,h) = cv2.meanShift(back, (x,y,w,h),
                                 (cv2.TERM_CRITERIA_EPS |
                                  cv2.TERM_CRITERIA_COUNT,10,1))

    # Draw rectangle
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Tracking",frame)

    if cv2.waitKey(30)==27:
        break

cap.release()
cv2.destroyAllWindows()
