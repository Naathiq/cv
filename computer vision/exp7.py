import cv2 as cv

# 1. Setup Anatomy Maps (Abbreviated for studying)
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4} # ... continues for 18 parts
POSE_PAIRS = [["Neck", "RShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"]] # ... continues

# 2. Load the AI Model
net = cv.dnn.readNetFromTensorflow('graph_opt1.pb')

# 3. Read Video/Image
cap = cv.VideoCapture('Sr.jpg')
hasFrame, frame = cap.read()
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

# 4. Create the Blob and feed it to the AI
blob = cv.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
net.setInput(blob)

# 5. Run the AI (Forward Pass)
out = net.forward()
out = out[:, :19, :, :] # Keep only the first 19 elements (the body parts)

points = []

# 6. Find the exact coordinate for each body part
for i in range(len(BODY_PARTS)):
    heatMap = out[0, i, :, :]
    
    # Finds the brightest spot (highest probability) on the heatmap
    _, conf, _, point = cv.minMaxLoc(heatMap)
    
    # Scale the point back to the original image size
    x = (frameWidth * point[0]) / out.shape[3]
    y = (frameHeight * point[1]) / out.shape[2]
    
    # If confidence is > 0.2, save the point. Otherwise, save None.
    if conf > 0.2:
        points.append((int(x), int(y)))
    else:
        points.append(None)

# 7. Draw the Skeleton
for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    
    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]
    
    # If both points exist, draw a line between them
    if points[idFrom] and points[idTo]:
        cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
        cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
        cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

cv.imshow('OpenPose', frame)
cv.waitKey(0)
cv.destroyAllWindows()