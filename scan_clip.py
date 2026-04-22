"""Scan a clip of a video file: 1 frame per 5 seconds, log all detections."""

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recognizer import FaceRecognizer
import database

database.create_tables()
rec = FaceRecognizer()

VIDEO = "/home/Vinny/Downloads/slack_downloads/20260422174940448.dav"
START_SEC = 300  # 5:00
END_SEC = 420    # 7:00
INTERVAL = 5     # 1 frame every 5 seconds
LOG_FILE = "scan_results.txt"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 15.0

# Seek to start
start_frame = int(START_SEC * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

log_lines = []
frame_idx = 0

print(f"Scanning {VIDEO}")
print(f"Range: {START_SEC}s ({START_SEC//60}:{START_SEC%60:02d}) → {END_SEC}s ({END_SEC//60}:{END_SEC%60:02d})")
print(f"Interval: 1 frame every {INTERVAL}s")
print(f"FPS: {fps}, skip {int(INTERVAL * fps)} frames between samples")
print()

skip = int(INTERVAL * fps)

while True:
    pos_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    pos_sec = pos_frames / fps

    if pos_sec >= END_SEC:
        break

    ret, frame = cap.read()
    if not ret:
        break

    # Only process at the interval
    frame_idx += 1
    if frame_idx % skip != 0:
        continue

    timestamp = f"{int(pos_sec)//60}:{int(pos_sec)%60:02d}"

    faces = rec.detect_faces(frame)
    if not faces:
        line = f"[{timestamp}] No faces detected"
        print(line)
        log_lines.append(line)
        continue

    for face in faces:
        bbox = face.bbox.astype(int)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        det_score = face.det_score

        if face.embedding is None:
            line = f"[{timestamp}] Face at {bbox.tolist()} area={area} — no embedding"
            print(line)
            log_lines.append(line)
            continue

        # Match against all known
        query = face.embedding.reshape(1, -1)
        best_score = 0.0
        best_name = "Unknown"

        for known in rec._known_embeddings:
            ref = known["embedding"].reshape(1, -1)
            score = cosine_similarity(query, ref)[0][0]
            if score > best_score:
                best_score = score
                best_name = known["name"]

        if best_score < 0.40:
            match_label = f"Unknown (best: {best_name} @ {best_score:.3f})"
        else:
            match_label = f"{best_name} (conf={best_score:.3f})"

        line = f"[{timestamp}] {match_label}  |  area={area}  det={det_score:.3f}  bbox={bbox.tolist()}"
        print(line)
        log_lines.append(line)

cap.release()

with open(LOG_FILE, "w") as f:
    f.write(f"Video: {VIDEO}\n")
    f.write(f"Range: {START_SEC//60}:{START_SEC%60:02d} → {END_SEC//60}:{END_SEC%60:02d}\n")
    f.write(f"Interval: 1 frame / {INTERVAL}s\n")
    f.write(f"Threshold: 0.40\n")
    f.write(f"{'='*80}\n\n")
    for line in log_lines:
        f.write(line + "\n")

print(f"\nSaved to {LOG_FILE}")
