"""Process chunk frames: run recognizer + speaker_tracker, produce speaker timeline."""

from datetime import datetime, timezone
from collections import defaultdict

import database
from recognizer import FaceRecognizer
from speaker_tracker import SpeakerTracker


def process_chunk(chunk_data, recognizer, stream_id="default"):
    """Process a recorded chunk and store speaker sightings in DB.

    Args:
        chunk_data: dict from recorder.record_chunk() with keys:
            frames, started_at, ended_at, chunk_index
        recognizer: FaceRecognizer instance
        stream_id: identifier for this stream

    Returns:
        list of sighting dicts for this chunk
    """
    frames = chunk_data["frames"]
    started_at = chunk_data["started_at"]
    ended_at = chunk_data["ended_at"]
    chunk_index = chunk_data["chunk_index"]

    if not frames:
        print(f"Chunk #{chunk_index}: no frames to process.")
        return []

    # Insert chunk record
    chunk_id = database.insert_chunk(stream_id, chunk_index, started_at, ended_at)

    tracker = SpeakerTracker()

    # Track speaker segments: list of (person_id, person_name, confidence, timestamp)
    timeline = []

    for frame, timestamp in frames:
        faces = recognizer.detect_faces(frame)
        if not faces:
            continue

        # Identify each face
        faces_with_identity = []
        for face in faces:
            pid, name, conf = recognizer.identify(face)
            faces_with_identity.append((face, pid, name, conf))
            print(f"  [{timestamp.strftime('%H:%M:%S')}] Detected: {name} (conf={conf:.3f})")

        # Determine active speaker
        speaker = tracker.update(faces_with_identity)
        if speaker is not None:
            pid, name, conf = speaker
            timeline.append({
                "person_id": pid,
                "person_name": name,
                "confidence": conf,
                "timestamp": timestamp,
            })

    # Collapse timeline into contiguous speaker segments
    sightings = _collapse_timeline(timeline)

    # Store sightings in DB
    for s in sightings:
        database.insert_sighting(
            chunk_id=chunk_id,
            person_id=s["person_id"],
            spoke_from=s["spoke_from"],
            spoke_until=s["spoke_until"],
            duration_seconds=s["duration_seconds"],
            avg_confidence=s["avg_confidence"],
            detection_count=s["detection_count"],
        )

    database.mark_chunk_processed(chunk_id)

    # Print summary
    print(f"\nChunk #{chunk_index} processed — {len(sightings)} speaker segment(s):")
    for s in sightings:
        print(f"  {s['person_name']}: {s['spoke_from'].strftime('%H:%M:%S')} → "
              f"{s['spoke_until'].strftime('%H:%M:%S')} "
              f"({s['duration_seconds']}s, avg_conf={s['avg_confidence']:.3f}, "
              f"detections={s['detection_count']})")

    return sightings


def _collapse_timeline(timeline):
    """Collapse a frame-by-frame timeline into contiguous speaker segments.

    Input: list of dicts with person_id, person_name, confidence, timestamp
    Output: list of segment dicts with spoke_from, spoke_until, duration, avg_confidence, count
    """
    if not timeline:
        return []

    segments = []
    current = {
        "person_id": timeline[0]["person_id"],
        "person_name": timeline[0]["person_name"],
        "spoke_from": timeline[0]["timestamp"],
        "spoke_until": timeline[0]["timestamp"],
        "confidences": [timeline[0]["confidence"]],
        "count": 1,
    }

    for entry in timeline[1:]:
        if entry["person_id"] == current["person_id"]:
            # Same speaker — extend segment
            current["spoke_until"] = entry["timestamp"]
            current["confidences"].append(entry["confidence"])
            current["count"] += 1
        else:
            # Speaker changed — finalize current segment
            segments.append(_finalize_segment(current))
            current = {
                "person_id": entry["person_id"],
                "person_name": entry["person_name"],
                "spoke_from": entry["timestamp"],
                "spoke_until": entry["timestamp"],
                "confidences": [entry["confidence"]],
                "count": 1,
            }

    # Finalize last segment
    segments.append(_finalize_segment(current))
    return segments


def _finalize_segment(seg):
    """Convert a raw segment into the final sighting dict."""
    duration = (seg["spoke_until"] - seg["spoke_from"]).total_seconds()
    avg_conf = sum(seg["confidences"]) / len(seg["confidences"]) if seg["confidences"] else 0.0
    return {
        "person_id": seg["person_id"],
        "person_name": seg["person_name"],
        "spoke_from": seg["spoke_from"],
        "spoke_until": seg["spoke_until"],
        "duration_seconds": int(duration),
        "avg_confidence": round(avg_conf, 4),
        "detection_count": seg["count"],
    }
