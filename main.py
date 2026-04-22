"""Main orchestrator: record + process chunks in parallel threads.

Usage:
  Live webcam:     python main.py
  RTSP stream:     python main.py --stream rtsp://192.168.1.1/stream
  File playback:   python main.py --stream /path/to/video.dav
"""

import argparse
import json
import os
import threading
import queue
import signal
from datetime import datetime, timezone

import database
from recorder import ChunkRecorder
from recognizer import FaceRecognizer
from processor import process_chunk
from config import STREAM_URL


# Graceful shutdown
shutdown_event = threading.Event()


def signal_handler(sig, frame):
    print("\nShutdown requested...")
    shutdown_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def recording_thread(recorder, chunk_queue, start_index=0):
    """Thread 1: record chunks and put them on the queue."""
    chunk_index = start_index
    while not shutdown_event.is_set():
        try:
            chunk_data = recorder.record_chunk(chunk_index=chunk_index)
            if chunk_data is None:
                # File ended
                chunk_queue.put(None)
                break
            chunk_queue.put(chunk_data)
            chunk_index += 1
        except Exception as e:
            print(f"Recording error: {e}")
            if shutdown_event.is_set():
                break
            print("Retrying in 5 seconds...")
            shutdown_event.wait(5)

    recorder.release()
    print("Recording thread stopped.")


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Speaker Tracker")
    parser.add_argument("--stream", type=str, default=None,
                        help="Stream URL, webcam index, or file path (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    source = args.stream or STREAM_URL
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    is_file = isinstance(source, str) and os.path.isfile(source)
    stream_id = f"stream_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    print("=== Face Recognition Speaker Tracker ===")
    print(f"Source: {source} ({'file' if is_file else 'live'})")
    print(f"Session: {stream_id}")
    print()

    print("Creating database tables...")
    database.create_tables()

    print("Loading InsightFace model...")
    recognizer = FaceRecognizer()

    recorder = ChunkRecorder(stream_url=source)
    chunk_queue = queue.Queue(maxsize=2)

    all_sightings = []
    sightings_lock = threading.Lock()

    def processing_worker():
        while not shutdown_event.is_set():
            try:
                chunk_data = chunk_queue.get(timeout=2)
            except queue.Empty:
                continue
            if chunk_data is None:
                break
            try:
                sightings = process_chunk(chunk_data, recognizer, stream_id=stream_id)
                with sightings_lock:
                    all_sightings.extend(sightings)
            except Exception as e:
                print(f"Processing error on chunk #{chunk_data['chunk_index']}: {e}")
                import traceback
                traceback.print_exc()
        print("Processing thread stopped.")

    rec_thread = threading.Thread(target=recording_thread,
                                  args=(recorder, chunk_queue), daemon=True)
    proc_thread = threading.Thread(target=processing_worker, daemon=True)

    print("Starting threads...")
    rec_thread.start()
    proc_thread.start()

    try:
        if is_file:
            rec_thread.join()
            proc_thread.join()
        else:
            while not shutdown_event.is_set():
                shutdown_event.wait(1)
            rec_thread.join(timeout=10)
            proc_thread.join(timeout=10)
    except KeyboardInterrupt:
        shutdown_event.set()
        rec_thread.join(timeout=10)
        proc_thread.join(timeout=10)

    # Print summary
    print("\n=== Summary ===")
    if not all_sightings:
        print("No speakers detected.")
    else:
        person_totals = {}
        for s in all_sightings:
            name = s["person_name"]
            if name not in person_totals:
                person_totals[name] = {"duration": 0, "segments": 0, "detections": 0}
            person_totals[name]["duration"] += s["duration_seconds"]
            person_totals[name]["segments"] += 1
            person_totals[name]["detections"] += s["detection_count"]

        for name, info in sorted(person_totals.items()):
            mins, secs = divmod(info["duration"], 60)
            print(f"  {name}: {mins}m {secs}s "
                  f"({info['segments']} segment(s), {info['detections']} detections)")

    # Save JSON if requested
    output_path = args.output
    if is_file and output_path is None:
        output_path = "results.json"

    if output_path and all_sightings:
        output = {
            "stream_id": stream_id,
            "source": str(source),
            "total_segments": len(all_sightings),
            "segments": [
                {
                    "person_name": s["person_name"],
                    "person_id": s["person_id"],
                    "spoke_from": s["spoke_from"].isoformat(),
                    "spoke_until": s["spoke_until"].isoformat(),
                    "duration_seconds": s["duration_seconds"],
                    "avg_confidence": s["avg_confidence"],
                    "detection_count": s["detection_count"],
                }
                for s in all_sightings
            ],
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
