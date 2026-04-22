"""Tests for speaker_tracker.py — largest face logic + transition guards."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from speaker_tracker import SpeakerTracker, _face_area
from tests.conftest import FakeFace


class TestFaceArea:
    def test_normal_bbox(self):
        face = FakeFace(bbox=[0, 0, 100, 200])
        assert _face_area(face) == 20000

    def test_zero_area(self):
        face = FakeFace(bbox=[50, 50, 50, 50])
        assert _face_area(face) == 0

    def test_small_face(self):
        face = FakeFace(bbox=[10, 10, 30, 30])
        assert _face_area(face) == 400


class TestSpeakerTrackerInit:
    def test_initial_state(self):
        tracker = SpeakerTracker()
        assert tracker.current_speaker is None
        assert tracker._candidate is None
        assert tracker._candidate_count == 0

    def test_reset(self):
        tracker = SpeakerTracker()
        tracker.current_speaker = (1, "alice")
        tracker._candidate = (2, "bob")
        tracker._candidate_count = 10
        tracker.reset()
        assert tracker.current_speaker is None
        assert tracker._candidate is None
        assert tracker._candidate_count == 0


class TestSpeakerTrackerEmptyInput:
    def test_no_faces(self):
        tracker = SpeakerTracker()
        assert tracker.update([]) is None

    def test_none_like_empty(self):
        tracker = SpeakerTracker()
        assert tracker.update([]) is None


class TestSpeakerTrackerSingleFace:
    def test_first_face_accepted_immediately(self):
        tracker = SpeakerTracker()
        face = FakeFace(bbox=[0, 0, 300, 300])
        result = tracker.update([(face, 1, "alice", 0.9)])
        assert result == (1, "alice", 0.9)
        assert tracker.current_speaker == (1, "alice")

    def test_same_face_stays(self):
        tracker = SpeakerTracker()
        face = FakeFace(bbox=[0, 0, 300, 300])
        tracker.update([(face, 1, "alice", 0.9)])
        result = tracker.update([(face, 1, "alice", 0.85)])
        assert result == (1, "alice", 0.85)

    def test_single_face_skips_ratio_check(self):
        """With only 1 face, size ratio guard doesn't apply."""
        tracker = SpeakerTracker()
        small_face = FakeFace(bbox=[0, 0, 50, 50])
        result = tracker.update([(small_face, 1, "alice", 0.8)])
        assert result is not None
        assert result[1] == "alice"


class TestSpeakerTrackerSizeRatioGuard:
    def test_transition_detected_when_faces_similar_size(self):
        """When two faces are nearly the same size, return None (transition)."""
        tracker = SpeakerTracker()
        face_a = FakeFace(bbox=[0, 0, 200, 200])  # area = 40000
        face_b = FakeFace(bbox=[0, 0, 190, 190])  # area = 36100

        # First, establish a speaker
        tracker.update([(face_a, 1, "alice", 0.9)])

        # Now both faces are similar size → ratio < 1.5 → transition
        result = tracker.update([
            (face_a, 1, "alice", 0.9),
            (face_b, 2, "bob", 0.8),
        ])
        assert result is None

    def test_clear_speaker_when_ratio_above_threshold(self):
        """When largest face is clearly bigger (>1.5x), speaker is identified."""
        tracker = SpeakerTracker()
        large = FakeFace(bbox=[0, 0, 300, 300])  # area = 90000
        small = FakeFace(bbox=[0, 0, 100, 100])  # area = 10000, ratio = 9.0

        result = tracker.update([
            (large, 1, "alice", 0.9),
            (small, 2, "bob", 0.8),
        ])
        assert result is not None
        assert result[1] == "alice"

    def test_ratio_exactly_at_threshold(self):
        """Ratio exactly at 1.5 should NOT be transition (< 1.5 triggers it)."""
        tracker = SpeakerTracker()
        large = FakeFace(bbox=[0, 0, 300, 200])   # area = 60000
        small = FakeFace(bbox=[0, 0, 200, 200])   # area = 40000, ratio = 1.5

        result = tracker.update([
            (large, 1, "alice", 0.9),
            (small, 2, "bob", 0.8),
        ])
        assert result is not None

    def test_second_face_zero_area(self):
        """Second face with zero area shouldn't cause division error."""
        tracker = SpeakerTracker()
        large = FakeFace(bbox=[0, 0, 300, 300])
        zero = FakeFace(bbox=[50, 50, 50, 50])  # zero area

        result = tracker.update([
            (large, 1, "alice", 0.9),
            (zero, 2, "bob", 0.8),
        ])
        # zero area → ratio check skipped (second_area <= 0)
        assert result is not None


class TestSpeakerTrackerStabilityBuffer:
    def test_speaker_switch_requires_consecutive_frames(self):
        """New speaker must be largest for SPEAKER_SWITCH_FRAMES consecutive frames."""
        from config import SPEAKER_SWITCH_FRAMES

        tracker = SpeakerTracker()
        face_a = FakeFace(bbox=[0, 0, 300, 300])
        face_b = FakeFace(bbox=[0, 0, 400, 400])
        face_small = FakeFace(bbox=[0, 0, 50, 50])

        # Establish alice as speaker
        tracker.update([(face_a, 1, "alice", 0.9)])

        # Bob becomes largest but shouldn't switch until threshold
        for i in range(SPEAKER_SWITCH_FRAMES - 1):
            result = tracker.update([
                (face_b, 2, "bob", 0.85),
                (face_small, 1, "alice", 0.7),
            ])
            assert result[1] == "alice", f"Should still be alice on frame {i}"

        # One more frame → switch happens
        result = tracker.update([
            (face_b, 2, "bob", 0.85),
            (face_small, 1, "alice", 0.7),
        ])
        assert result[1] == "bob"
        assert tracker.current_speaker == (2, "bob")

    def test_interrupted_candidate_resets_count(self):
        """If original speaker becomes largest again, candidate count resets."""
        tracker = SpeakerTracker()
        face_a = FakeFace(bbox=[0, 0, 300, 300])
        face_b = FakeFace(bbox=[0, 0, 400, 400])
        face_small = FakeFace(bbox=[0, 0, 50, 50])

        tracker.update([(face_a, 1, "alice", 0.9)])

        # Bob is largest for 5 frames
        for _ in range(5):
            tracker.update([
                (face_b, 2, "bob", 0.85),
                (face_small, 1, "alice", 0.7),
            ])

        # Alice comes back as largest — should reset candidate count
        tracker.update([(face_a, 1, "alice", 0.9)])
        assert tracker._candidate is None
        assert tracker._candidate_count == 0

    def test_different_candidate_resets_count(self):
        """If a third person becomes largest, previous candidate count resets."""
        tracker = SpeakerTracker()
        face_a = FakeFace(bbox=[0, 0, 300, 300])
        face_b = FakeFace(bbox=[0, 0, 400, 400])
        face_c = FakeFace(bbox=[0, 0, 450, 450])
        face_small = FakeFace(bbox=[0, 0, 50, 50])

        tracker.update([(face_a, 1, "alice", 0.9)])

        # Bob for 5 frames
        for _ in range(5):
            tracker.update([
                (face_b, 2, "bob", 0.85),
                (face_small, 1, "alice", 0.7),
            ])
        assert tracker._candidate_count == 5

        # Charlie takes over — count resets to 1
        tracker.update([
            (face_c, 3, "charlie", 0.8),
            (face_small, 1, "alice", 0.7),
        ])
        assert tracker._candidate == (3, "charlie")
        assert tracker._candidate_count == 1

    def test_current_speaker_not_in_frame_during_buffer(self):
        """During buffer period, if current speaker leaves frame, still return them."""
        tracker = SpeakerTracker()
        face_a = FakeFace(bbox=[0, 0, 300, 300])
        face_b = FakeFace(bbox=[0, 0, 400, 400])

        tracker.update([(face_a, 1, "alice", 0.9)])

        # Bob is alone in frame (alice left) but hasn't reached threshold
        result = tracker.update([(face_b, 2, "bob", 0.85)])
        assert result[1] == "alice"
        assert result[2] == 0.0  # confidence = 0 since alice not in frame


class TestSpeakerTrackerUnknown:
    def test_unknown_speaker(self):
        """Unknown (person_id=None) should still work as a speaker."""
        tracker = SpeakerTracker()
        face = FakeFace(bbox=[0, 0, 300, 300])
        result = tracker.update([(face, None, "Unknown", 0.3)])
        assert result == (None, "Unknown", 0.3)
