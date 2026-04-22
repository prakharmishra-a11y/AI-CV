"""Live webcam face recognition test.

Opens the webcam, detects faces, and draws bounding boxes with
the recognized person's name and confidence score.

Press 'q' to quit.
"""

import warnings
warnings.filterwarnings("ignore")

import cv2
import time
import database
from recognizer import FaceRecognizer

database.create_tables()

print("Loading InsightFace model...")
recognizer = FaceRecognizer()
print(f"Ready — {len(recognizer._known_embeddings)} face(s) enrolled.\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit(1)

print("Webcam opened. Press 'q' to quit.\n")

frame_count = 0
fps_start = time.time()
display_fps = 0

# Process every Nth frame to keep it smooth (recognition is heavy on CPU)
PROCESS_EVERY = 5
last_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Calculate FPS
    elapsed = time.time() - fps_start
    if elapsed >= 1.0:
        display_fps = frame_count / elapsed
        frame_count = 0
        fps_start = time.time()

    # Run recognition every Nth frame
    if frame_count % PROCESS_EVERY == 0:
        faces = recognizer.detect_faces(frame)
        last_results = []
        for face in faces:
            pid, name, conf = recognizer.identify(face)
            bbox = face.bbox.astype(int)
            last_results.append((bbox, name, conf, face.det_score))

    # Draw results on every frame (using last detection)
    for bbox, name, conf, det_score in last_results:
        x1, y1, x2, y2 = bbox

        if name != "Unknown":
            # Green box for recognized
            color = (0, 200, 0)
            label = f"{name} ({conf:.2f})"
        else:
            # Red box for unknown
            color = (0, 0, 200)
            label = f"Unknown ({conf:.2f})"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw FPS
    cv2.putText(frame, f"FPS: {display_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw enrolled count
    cv2.putText(frame, f"Enrolled: {len(recognizer._known_embeddings)} face(s)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Face Recognition - Live Test (press 'q' to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
