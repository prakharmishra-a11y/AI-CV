"""Tests for processor.py — timeline collapsing and chunk processing."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from processor import _collapse_timeline, _finalize_segment, process_chunk
from tests.conftest import FakeFace


class TestCollapseTimeline:
    def test_empty_timeline(self):
        assert _collapse_timeline([]) == []

    def test_single_entry(self):
        t = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timeline = [{"person_id": 1, "person_name": "alice", "confidence": 0.9, "timestamp": t}]
        segments = _collapse_timeline(timeline)
        assert len(segments) == 1
        assert segments[0]["person_name"] == "alice"
        assert segments[0]["duration_seconds"] == 0
        assert segments[0]["detection_count"] == 1

    def test_same_speaker_collapses(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timeline = [
            {"person_id": 1, "person_name": "alice", "confidence": 0.9, "timestamp": base},
            {"person_id": 1, "person_name": "alice", "confidence": 0.85, "timestamp": base + timedelta(seconds=1)},
            {"person_id": 1, "person_name": "alice", "confidence": 0.88, "timestamp": base + timedelta(seconds=2)},
        ]
        segments = _collapse_timeline(timeline)
        assert len(segments) == 1
        assert segments[0]["duration_seconds"] == 2
        assert segments[0]["detection_count"] == 3
        assert 0.87 < segments[0]["avg_confidence"] < 0.89

    def test_two_speakers_creates_two_segments(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timeline = [
            {"person_id": 1, "person_name": "alice", "confidence": 0.9, "timestamp": base},
            {"person_id": 1, "person_name": "alice", "confidence": 0.85, "timestamp": base + timedelta(seconds=1)},
            {"person_id": 2, "person_name": "bob", "confidence": 0.8, "timestamp": base + timedelta(seconds=2)},
            {"person_id": 2, "person_name": "bob", "confidence": 0.82, "timestamp": base + timedelta(seconds=3)},
        ]
        segments = _collapse_timeline(timeline)
        assert len(segments) == 2
        assert segments[0]["person_name"] == "alice"
        assert segments[1]["person_name"] == "bob"

    def test_speaker_returns_creates_three_segments(self):
        """A-B-A pattern should create 3 segments, not merge the two A's."""
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timeline = [
            {"person_id": 1, "person_name": "alice", "confidence": 0.9, "timestamp": base},
            {"person_id": 2, "person_name": "bob", "confidence": 0.8, "timestamp": base + timedelta(seconds=10)},
            {"person_id": 1, "person_name": "alice", "confidence": 0.85, "timestamp": base + timedelta(seconds=20)},
        ]
        segments = _collapse_timeline(timeline)
        assert len(segments) == 3

    def test_unknown_speaker_segments(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timeline = [
            {"person_id": None, "person_name": "Unknown", "confidence": 0.3, "timestamp": base},
            {"person_id": None, "person_name": "Unknown", "confidence": 0.25, "timestamp": base + timedelta(seconds=1)},
        ]
        segments = _collapse_timeline(timeline)
        assert len(segments) == 1
        assert segments[0]["person_name"] == "Unknown"
        assert segments[0]["person_id"] is None


class TestFinalizeSegment:
    def test_basic_segment(self):
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        seg = {
            "person_id": 1,
            "person_name": "alice",
            "spoke_from": base,
            "spoke_until": base + timedelta(seconds=30),
            "confidences": [0.8, 0.9, 0.85],
            "count": 3,
        }
        result = _finalize_segment(seg)
        assert result["duration_seconds"] == 30
        assert result["detection_count"] == 3
        assert 0.84 < result["avg_confidence"] < 0.86

    def test_zero_duration_segment(self):
        t = datetime(2025, 1, 1, tzinfo=timezone.utc)
        seg = {
            "person_id": 1,
            "person_name": "alice",
            "spoke_from": t,
            "spoke_until": t,
            "confidences": [0.9],
            "count": 1,
        }
        result = _finalize_segment(seg)
        assert result["duration_seconds"] == 0


class TestProcessChunk:
    def test_empty_frames(self, test_db):
        """Chunk with no frames should return empty sightings."""
        recognizer = MagicMock()
        chunk_data = {
            "frames": [],
            "started_at": datetime.now(timezone.utc),
            "ended_at": datetime.now(timezone.utc),
            "chunk_index": 0,
        }
        sightings = process_chunk(chunk_data, recognizer, stream_id="test_empty")
        assert sightings == []

    def test_no_faces_in_frames(self, test_db):
        """Frames with no detected faces should produce no sightings."""
        recognizer = MagicMock()
        recognizer.detect_faces.return_value = []

        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        now = datetime.now(timezone.utc)
        chunk_data = {
            "frames": [(frame, now), (frame, now + timedelta(seconds=1))],
            "started_at": now,
            "ended_at": now + timedelta(seconds=2),
            "chunk_index": 0,
        }
        sightings = process_chunk(chunk_data, recognizer, stream_id="test_noface")
        assert sightings == []

    def test_single_speaker_produces_sighting(self, test_db):
        """Single recognized person across frames should produce one sighting."""
        pid = test_db.insert_person("process_test_alice")

        face = FakeFace(bbox=[0, 0, 300, 300])
        recognizer = MagicMock()
        recognizer.detect_faces.return_value = [face]
        recognizer.identify.return_value = (pid, "process_test_alice", 0.9)

        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        now = datetime.now(timezone.utc)
        chunk_data = {
            "frames": [
                (frame, now),
                (frame, now + timedelta(seconds=1)),
                (frame, now + timedelta(seconds=2)),
            ],
            "started_at": now,
            "ended_at": now + timedelta(seconds=3),
            "chunk_index": 0,
        }
        sightings = process_chunk(chunk_data, recognizer, stream_id="test_single")
        assert len(sightings) >= 1
        assert sightings[0]["person_name"] == "process_test_alice"
        assert sightings[0]["detection_count"] >= 1
