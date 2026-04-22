"""Tests for config.py — verify all config constants are set."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


class TestConfigValues:
    def test_cosine_threshold_range(self):
        assert 0.0 < config.COSINE_THRESHOLD < 1.0

    def test_insightface_model_set(self):
        assert isinstance(config.INSIGHTFACE_MODEL, str)
        assert len(config.INSIGHTFACE_MODEL) > 0

    def test_ctx_id_is_cpu(self):
        assert config.CTX_ID == -1

    def test_speaker_size_ratio(self):
        assert config.SPEAKER_SIZE_RATIO > 1.0

    def test_speaker_switch_frames(self):
        assert config.SPEAKER_SWITCH_FRAMES > 0
        assert isinstance(config.SPEAKER_SWITCH_FRAMES, int)

    def test_chunk_duration_positive(self):
        assert config.CHUNK_DURATION > 0

    def test_frame_sample_rate_positive(self):
        assert config.FRAME_SAMPLE_RATE > 0

    def test_database_url_set(self):
        assert isinstance(config.DATABASE_URL, str)
        assert "face_recognition" in config.DATABASE_URL

    def test_known_faces_dir_set(self):
        assert isinstance(config.KNOWN_FACES_DIR, str)

    def test_min_det_score_range(self):
        assert 0.0 < config.MIN_DET_SCORE <= 1.0

    def test_no_video_file_config(self):
        """VIDEO_FILE should not exist — video input was removed."""
        assert not hasattr(config, "VIDEO_FILE")
