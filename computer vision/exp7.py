import cv2 as cv

# Body parts
BODY_PARTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7
}

# Connections between body parts
POSE_PAIRS = [
    ["Neck", "RShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["Neck", "LShoulder"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"]
]

# Load AI model
net = cv.dnn.readNetFromTensorflow("graph_opt1.pb")

# Load image
img = cv.imread("body.jpg")

if img is None:
    print("Image not found!")
    exit()

h, w = img.shape[:2]

# Convert image to blob
blob = cv.dnn.blobFromImage(img, 1.0, (368, 368),
                            (127.5, 127.5, 127.5),
                            swapRB=True, crop=False)

net.setInput(blob)

# Run model
output = net.forward()

points = []

# Find body points
for i in range(8):
    heatMap = output[0, i, :, :]
    _, conf, _, point = cv.minMaxLoc(heatMap)

    x = int((w * point[0]) / output.shape[3])
    y = int((h * point[1]) / output.shape[2])

    if conf > 0.2:
        points.append((x, y))
    else:
        points.append(None)

# Draw skeleton
for pair in POSE_PAIRS:
    partA = BODY_PARTS[pair[0]]
    partB = BODY_PARTS[pair[1]]

    if points[partA] and points[partB]:
        cv.line(img, points[partA], points[partB], (0, 255, 0), 2)
        cv.circle(img, points[partA], 4, (0, 0, 255), -1)
        cv.circle(img, points[partB], 4, (0, 0, 255), -1)

# Show result
cv.imshow("Pose Detection", img)
cv.waitKey(0)
cv.destroyAllWindows()
