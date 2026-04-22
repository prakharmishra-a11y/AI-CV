"""OpenCV stream capture: record 5-minute chunks, sample every Nth frame.

Supports:
  - Live streams (webcam / RTSP): chunks by wall-clock time
  - File playback (.mp4, .dav, etc.): chunks by video timestamps, full speed
"""

import os
import time
from datetime import datetime, timedelta, timezone

import cv2

from config import STREAM_URL, CHUNK_DURATION, FRAME_SAMPLE_RATE


class ChunkRecorder:
    def __init__(self, stream_url=STREAM_URL):
        self.stream_url = stream_url
        self.cap = None
        self.is_file = isinstance(stream_url, str) and os.path.isfile(stream_url)
        self.video_ended = False

    def _open_stream(self):
        """Open or reconnect to the video stream / file."""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            raise ConnectionError(f"Cannot open: {self.stream_url}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / self.fps if self.fps > 0 else 0
        mode = "file" if self.is_file else "live"
        print(f"Opened ({mode}): {self.stream_url}  "
              f"fps={self.fps:.1f}  frames={total_frames}  duration={duration:.0f}s")

    def record_chunk(self, chunk_index=0):
        """Record one chunk of CHUNK_DURATION seconds.

        Returns dict with frames, timestamps, chunk_index.
        Returns None if file has ended.
        """
        if self.video_ended:
            return None

        if self.cap is None or not self.cap.isOpened():
            self._open_stream()

        if self.is_file:
            return self._record_chunk_file(chunk_index)
        else:
            return self._record_chunk_live(chunk_index)

    def _record_chunk_file(self, chunk_index):
        """Read a chunk from a video file at full speed using video timestamps."""
        frames = []
        frame_count = 0
        chunk_start_sec = chunk_index * CHUNK_DURATION
        chunk_end_sec = chunk_start_sec + CHUNK_DURATION

        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        started_at = base_time + timedelta(seconds=chunk_start_sec)
        pos_sec = chunk_start_sec

        print(f"Reading chunk #{chunk_index} [{chunk_start_sec}s - {chunk_end_sec}s] ...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.video_ended = True
                break

            pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            pos_sec = pos_msec / 1000.0

            if pos_sec >= chunk_end_sec:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
                break

            if pos_sec < chunk_start_sec:
                continue

            frame_count += 1

            if frame_count % FRAME_SAMPLE_RATE == 0:
                timestamp = base_time + timedelta(seconds=pos_sec)
                frames.append((frame, timestamp))

        ended_at = base_time + timedelta(seconds=min(pos_sec, chunk_end_sec))
        print(f"Chunk #{chunk_index} done: {len(frames)} sampled frames "
              f"from {frame_count} total")

        return {
            "frames": frames,
            "started_at": started_at,
            "ended_at": ended_at,
            "chunk_index": chunk_index,
        }

    def _record_chunk_live(self, chunk_index):
        """Record a chunk from a live stream using wall-clock time."""
        frames = []
        frame_count = 0
        started_at = datetime.now(timezone.utc)
        start_time = time.time()

        print(f"Recording chunk #{chunk_index} "
              f"(duration={CHUNK_DURATION}s, sample_rate=1/{FRAME_SAMPLE_RATE})...")

        while True:
            elapsed = time.time() - start_time
            if elapsed >= CHUNK_DURATION:
                break

            ret, frame = self.cap.read()
            if not ret:
                print("Stream read failed, attempting reconnect...")
                try:
                    self._open_stream()
                except ConnectionError:
                    print("Reconnect failed, ending chunk early.")
                    break
                continue

            frame_count += 1

            if frame_count % FRAME_SAMPLE_RATE == 0:
                timestamp = datetime.now(timezone.utc)
                frames.append((frame, timestamp))

        ended_at = datetime.now(timezone.utc)
        print(f"Chunk #{chunk_index} done: {len(frames)} sampled frames "
              f"from {frame_count} total in {elapsed:.1f}s")

        return {
            "frames": frames,
            "started_at": started_at,
            "ended_at": ended_at,
            "chunk_index": chunk_index,
        }

    def release(self):
        """Release the video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
