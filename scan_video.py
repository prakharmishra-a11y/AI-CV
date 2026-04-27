"""Scan a video: 1 frame per 5 seconds, log all face detections with match scores."""

import warnings
warnings.filterwarnings("ignore")

import cv2
import sys
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recognizer import FaceRecognizer
import database

database.create_tables()
print("Loading model...")
rec = FaceRecognizer()

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/home/Vinny/Downloads/YTDown.com_YouTube_Varshitap-Parna-Mahotsav-Param-Namramuni_Media_qJqOvp-G3OQ_002_720p.mp4"
INTERVAL = 5  # seconds between frames
THRESHOLD = 0.30
LOG_FILE = "video_scan_results.txt"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
skip = max(1, int(INTERVAL * fps))

print(f"Video: {VIDEO}")
print(f"Duration: {int(duration//60)}m {int(duration%60)}s | FPS: {fps} | Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
print(f"Sampling: 1 frame every {INTERVAL}s (skip {skip} frames)")
print(f"Threshold: {THRESHOLD}")
print(f"Known faces: {len(rec._known_embeddings)}")
print()

log_lines = []
matches_by_person = {}
frame_count = 0
sample_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip != 0:
        continue

    sample_count += 1
    pos_sec = frame_count / fps
    timestamp = f"{int(pos_sec)//60}:{int(pos_sec)%60:02d}"

    faces = rec.detect_faces(frame)
    if not faces:
        continue

    for face in faces:
        bbox = face.bbox.astype(int)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        if area < 400:  # skip tiny faces
            continue

        if face.embedding is None:
            continue

        query = face.embedding.reshape(1, -1)

        # Get top 3 matches
        scores = []
        for known in rec._known_embeddings:
            ref = known["embedding"].reshape(1, -1)
            score = cosine_similarity(query, ref)[0][0]
            scores.append((known["name"], known["person_id"], score))

        scores.sort(key=lambda x: -x[2])
        top = scores[0]
        top3 = scores[:3]

        if top[2] >= THRESHOLD:
            # MATCHED
            name = top[0]
            conf = top[2]
            line = f"[{timestamp}] MATCH: {name} (conf={conf:.3f})  area={area}  bbox={bbox.tolist()}"
            print(line)
            log_lines.append(line)

            if name not in matches_by_person:
                matches_by_person[name] = []
            matches_by_person[name].append({"time": timestamp, "conf": conf, "area": area})
        else:
            # Show top 3 closest for debugging
            top3_str = " | ".join([f"{n} {s:.3f}" for n, _, s in top3])
            line = f"[{timestamp}] Unknown  top3=[{top3_str}]  area={area}"
            log_lines.append(line)

    if sample_count % 50 == 0:
        elapsed = time.time() - start_time
        pct = pos_sec / duration * 100
        print(f"  ... {timestamp} ({pct:.0f}%) — {len(matches_by_person)} persons matched so far")

cap.release()

# Write log file
with open(LOG_FILE, "w") as f:
    f.write(f"Video: {VIDEO}\n")
    f.write(f"Duration: {int(duration//60)}m {int(duration%60)}s\n")
    f.write(f"Interval: 1 frame / {INTERVAL}s\n")
    f.write(f"Threshold: {THRESHOLD}\n")
    f.write(f"Known faces: {len(rec._known_embeddings)}\n")
    f.write(f"{'='*80}\n\n")

    # Summary
    f.write("MATCHED PERSONS SUMMARY\n")
    f.write(f"{'-'*80}\n")
    for name in sorted(matches_by_person.keys()):
        hits = matches_by_person[name]
        avg_conf = sum(h["conf"] for h in hits) / len(hits)
        times = ", ".join(h["time"] for h in hits)
        f.write(f"{name}: {len(hits)} detections, avg_conf={avg_conf:.3f}\n")
        f.write(f"  timestamps: {times}\n\n")

    f.write(f"\n{'='*80}\n")
    f.write("FULL LOG\n")
    f.write(f"{'='*80}\n\n")
    for line in log_lines:
        f.write(line + "\n")

elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"Done in {elapsed:.0f}s")
print(f"Frames sampled: {sample_count}")
print(f"Persons matched: {len(matches_by_person)}")
print()

if matches_by_person:
    print("MATCHED PERSONS:")
    print(f"{'-'*60}")
    for name in sorted(matches_by_person.keys()):
        hits = matches_by_person[name]
        avg_conf = sum(h["conf"] for h in hits) / len(hits)
        times = [h["time"] for h in hits]
        print(f"  {name}: {len(hits)} hits, avg={avg_conf:.3f}")
        print(f"    at: {', '.join(times[:10])}{'...' if len(times)>10 else ''}")

print(f"\nFull log saved to {LOG_FILE}")
