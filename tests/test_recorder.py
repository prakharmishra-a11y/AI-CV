"""Tests for recorder.py — live stream recording."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from recorder import ChunkRecorder


class TestChunkRecorderInit:
    def test_webcam_init(self):
        rec = ChunkRecorder(stream_url=0)
        assert rec.stream_url == 0
        assert rec.cap is None

    def test_rtsp_init(self):
        rec = ChunkRecorder(stream_url="rtsp://192.168.1.1/stream")
        assert rec.stream_url == "rtsp://192.168.1.1/stream"


class TestChunkRecorderOpenStream:
    def test_open_failure_raises(self):
        rec = ChunkRecorder(stream_url="rtsp://invalid_url_that_wont_connect")
        with pytest.raises(ConnectionError):
            rec._open_stream()

    @patch("recorder.cv2.VideoCapture")
    def test_open_success(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap_cls.return_value = mock_cap

        rec = ChunkRecorder(stream_url=0)
        rec._open_stream()
        assert rec.cap is not None
        mock_cap_cls.assert_called_once_with(0)


class TestChunkRecorderRecordChunk:
    @patch("recorder.cv2.VideoCapture")
    @patch("recorder.time")
    def test_record_chunk_returns_dict(self, mock_time, mock_cap_cls):
        """Simulate a short recording with mocked time and capture."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0

        # Return 60 frames then simulate time elapsed
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        mock_cap_cls.return_value = mock_cap

        # Simulate time: first call returns start, subsequent calls exceed chunk duration
        call_count = [0]

        def fake_time():
            call_count[0] += 1
            if call_count[0] <= 2:
                return 0.0
            return 999.0  # exceed CHUNK_DURATION immediately

        mock_time.time = fake_time

        rec = ChunkRecorder(stream_url=0)
        rec.cap = mock_cap

        chunk = rec.record_chunk(chunk_index=0)
        assert chunk is not None
        assert "frames" in chunk
        assert "started_at" in chunk
        assert "ended_at" in chunk
        assert chunk["chunk_index"] == 0

    @patch("recorder.cv2.VideoCapture")
    @patch("recorder.time")
    def test_frame_sampling(self, mock_time, mock_cap_cls):
        """Should only keep every FRAME_SAMPLE_RATE-th frame."""
        from config import FRAME_SAMPLE_RATE

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Return 90 frames (3 seconds at 30fps)
        read_count = [0]

        def fake_read():
            read_count[0] += 1
            return (True, frame.copy())

        mock_cap.read = fake_read
        mock_cap_cls.return_value = mock_cap

        # Time: let 90 reads happen then stop
        time_call = [0]

        def fake_time():
            time_call[0] += 1
            if time_call[0] <= 2:
                return 0.0
            # Stop after 90 reads
            if read_count[0] >= 90:
                return 999.0
            return 1.0  # still within CHUNK_DURATION

        mock_time.time = fake_time

        rec = ChunkRecorder(stream_url=0)
        rec.cap = mock_cap

        chunk = rec.record_chunk(chunk_index=0)
        # 90 frames / 30 sample rate = 3 sampled frames
        assert len(chunk["frames"]) == 90 // FRAME_SAMPLE_RATE

    @patch("recorder.cv2.VideoCapture")
    def test_reconnect_on_read_failure(self, mock_cap_cls):
        """Should attempt reconnect when read() fails."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        # First read fails, reconnect also fails → ends chunk early
        mock_cap.read.return_value = (False, None)
        mock_cap_cls.return_value = mock_cap

        rec = ChunkRecorder(stream_url=0)
        rec.cap = mock_cap

        # Make _open_stream raise on reconnect
        with patch.object(rec, '_open_stream', side_effect=ConnectionError("reconnect failed")):
            chunk = rec.record_chunk(chunk_index=0)
            assert chunk is not None
            assert len(chunk["frames"]) == 0


class TestChunkRecorderRelease:
    def test_release_when_not_opened(self):
        rec = ChunkRecorder(stream_url=0)
        rec.release()  # should not raise
        assert rec.cap is None

    def test_release_after_open(self):
        rec = ChunkRecorder(stream_url=0)
        mock_cap = MagicMock()
        rec.cap = mock_cap
        rec.release()
        mock_cap.release.assert_called_once()
        assert rec.cap is None
