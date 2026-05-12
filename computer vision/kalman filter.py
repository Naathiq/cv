import cv2
import numpy as np

# Open video
cap = cv2.VideoCapture("video1.mp4")

# Create Kalman Filter
kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)

# First frame
ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Motion detection
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Predict next position
    pred = kalman.predict()
    px, py = int(pred[0]), int(pred[1])

    if contours:
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)

            cx = x + w//2
            cy = y + h//2

            # Correct prediction
            kalman.correct(np.array([[np.float32(cx)],
                                     [np.float32(cy)]]))

            # Red = detected
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

    # Green = predicted
    cv2.circle(frame, (px, py), 5, (0,255,0), -1)

    cv2.imshow("Tracking", frame)

    prev_gray = gray

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
