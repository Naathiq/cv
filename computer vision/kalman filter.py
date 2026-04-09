import cv2
import numpy as np

# --- STEP 1: Setup Video & Background ---
cap = cv2.VideoCapture("video.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

# --- STEP 2: Setup Kalman Filter ---
# 4 states (x, y, dx, dy) and 2 measurements (x, y)
kf = cv2.KalmanFilter(4, 2)

kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                 [0, 1, 0, 0]], np.float32)

kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                [0, 1, 0, 1], 
                                [0, 0, 1, 0], 
                                [0, 0, 0, 1]], np.float32)

# --- STEP 3: Main Loop & Masking ---
while True:
    ret, frame = cap.read()
    if not ret: break
    
    mask = fgbg.apply(frame)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    
    # --- STEP 4: Find Contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- STEP 5: Track the Largest Object ---
    if contours:
        # Find the biggest blob to avoid the multi-car bug
        c = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(c) > 1500:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w//2, y + h//2  # Center point
            
            # Kalman: Correct -> Predict -> Draw
            kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
            pred = kf.predict()
            px, py = int(pred[0]), int(pred[1])
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(30) == 27: break # Press ESC to exit

cap.release()
cv2.destroyAllWindows()
