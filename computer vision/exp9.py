import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("Traffic IP Camera video.mp4")

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=25,
    detectShadows=True
)

# Tracker storage
trackers = {}
next_id = 0

# Parameters
distance_threshold = 80
min_area = 1500
max_missed = 5

# Kalman Filter function
def create_kalman(x, y):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix = np.array([
        [1,0,1,0],
        [0,1,0,1],
        [0,0,1,0],
        [0,0,0,1]
    ], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.statePre = np.array([[x],[y],[0],[0]], np.float32)
    return kf

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    # Remove old predictions
    for data in trackers.values():
        data.pop('pred', None)

    # Foreground mask
    fgmask = fgbg.apply(frame)

    # Remove shadows
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    fgmask = cv2.dilate(fgmask, np.ones((5,5), np.uint8), iterations=2)

    # Contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
            detections.append((cx, cy, x, y, w, h))

    updated_trackers = {}
    matched_ids = set()

    # Match detections to trackers
    for det in detections:
        cx, cy, x, y, w, h = det

        min_dist = float('inf')
        matched_id = None

        for tid, data in trackers.items():
            if tid in matched_ids:
                continue

            if 'pred' not in data:
                data['pred'] = data['kf'].predict()

            px, py = int(data['pred'][0]), int(data['pred'][1])
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)

            if dist < min_dist and dist < distance_threshold:
                min_dist = dist
                matched_id = tid

        if matched_id is not None:
            kf = trackers[matched_id]['kf']
            kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))

            old_box = trackers[matched_id]['bbox']
            alpha = 0.7
            new_box = [x, y, w, h]

            smoothed_box = [
                int(alpha*old_box[0] + (1-alpha)*new_box[0]),
                int(alpha*old_box[1] + (1-alpha)*new_box[1]),
                int(alpha*old_box[2] + (1-alpha)*new_box[2]),
                int(alpha*old_box[3] + (1-alpha)*new_box[3]),
            ]

            updated_trackers[matched_id] = {
                'kf': kf,
                'bbox': smoothed_box,
                'missed': 0
            }

            matched_ids.add(matched_id)

        else:
            kf = create_kalman(cx, cy)
            updated_trackers[next_id] = {
                'kf': kf,
                'bbox': [x, y, w, h],
                'missed': 0
            }
            next_id += 1

    # Handle lost trackers
    for tid, data in trackers.items():
        if tid not in matched_ids:
            data['missed'] += 1
            if data['missed'] <= max_missed:
                updated_trackers[tid] = data

    trackers = updated_trackers

    # Draw results
    for tid, data in trackers.items():
        x, y, w, h = data['bbox']

        if 'pred' not in data:
            data['pred'] = data['kf'].predict()

        px, py = int(data['pred'][0]), int(data['pred'][1])

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(frame, (px, py), 5, (0,0,255), -1)
        cv2.putText(frame, f"ID {tid}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Car Tracking", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# csrt

import cv2

# Load video
cap = cv2.VideoCapture("videoplayback.mp4")

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read frame")
    exit()

frame = cv2.resize(frame, (960, 540))

# Select ROI
bbox = cv2.selectROI("Select Object to Track", frame, False, False)
x, y, w, h = bbox

if w == 0 or h == 0:
    print("No ROI selected")
    exit()

# Create CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Initialize tracker
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(frame, "Tracking", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost Object", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("CSRT Tracker", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()