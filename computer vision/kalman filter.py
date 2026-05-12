import cv2
import numpy as np

cap = cv2.VideoCapture(r"D:\Users\yesur\Program Files - PHY\video1.mp4")   # or 0

# Kalman Filter
kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]], np.float32)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5

kalman.statePre = np.zeros((4,1), np.float32)
initialized = False

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    cv2.namedWindow("Motion", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 🔹 Motion Detection (Frame Difference)
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Predict
    predicted = kalman.predict()
    px, py = int(predicted[0][0]), int(predicted[1][0])

    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)

        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)

            mx = x + w//2
            my = y + h//2

            measurement = np.array([[np.float32(mx)],
                                    [np.float32(my)]])

            if not initialized:
                kalman.statePre = np.array([[mx],
                                            [my],
                                            [0],
                                            [0]], np.float32)
                initialized = True

            kalman.correct(measurement)

            # Red = detected
            cv2.circle(frame, (mx, my), 5, (0,0,255), -1)

    # Green = predicted
    cv2.circle(frame, (px, py), 5, (0,255,0), -1)

    cv2.imshow("Motion", thresh)
    cv2.imshow("Tracking", frame)

    prev_gray = gray.copy()

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
