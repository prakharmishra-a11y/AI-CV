"""Tests for recognizer.py — face detection, embedding, identification."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from tests.conftest import FakeFace


class TestFaceRecognizerIdentify:
    """Test identification logic without loading the real InsightFace model."""

    def _make_recognizer_with_known(self, known_embeddings):
        """Create a FaceRecognizer-like object with mocked internals."""
        from recognizer import FaceRecognizer

        with patch.object(FaceRecognizer, '__init__', lambda self: None):
            rec = FaceRecognizer()
            rec.app = MagicMock()
            rec._known_embeddings = known_embeddings
        return rec

    def test_identify_no_known_embeddings(self):
        rec = self._make_recognizer_with_known([])
        face = FakeFace(bbox=[0, 0, 100, 100])
        pid, name, conf = rec.identify(face)
        assert pid is None
        assert name == "Unknown"
        assert conf == 0.0

    def test_identify_matching_embedding(self):
        """A face with the same embedding should match with high confidence."""
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # normalize

        rec = self._make_recognizer_with_known([
            {"person_id": 1, "name": "alice", "embedding": emb.copy()},
        ])

        face = FakeFace(bbox=[0, 0, 100, 100], embedding=emb.copy())
        pid, name, conf = rec.identify(face)
        assert pid == 1
        assert name == "alice"
        assert conf > 0.99  # same embedding → cosine sim ≈ 1.0

    def test_identify_different_embedding_below_threshold(self):
        """Orthogonal embeddings should produce low similarity → Unknown."""
        emb_known = np.zeros(512, dtype=np.float32)
        emb_known[0] = 1.0  # unit vector along dim 0

        emb_query = np.zeros(512, dtype=np.float32)
        emb_query[1] = 1.0  # unit vector along dim 1 (orthogonal)

        rec = self._make_recognizer_with_known([
            {"person_id": 1, "name": "alice", "embedding": emb_known},
        ])

        face = FakeFace(bbox=[0, 0, 100, 100], embedding=emb_query)
        pid, name, conf = rec.identify(face)
        assert pid is None
        assert name == "Unknown"

    def test_identify_best_match_across_persons(self):
        """Should match the person with highest cosine similarity."""
        emb_a = np.random.randn(512).astype(np.float32)
        emb_a = emb_a / np.linalg.norm(emb_a)

        emb_b = np.random.randn(512).astype(np.float32)
        emb_b = emb_b / np.linalg.norm(emb_b)

        rec = self._make_recognizer_with_known([
            {"person_id": 1, "name": "alice", "embedding": emb_a},
            {"person_id": 2, "name": "bob", "embedding": emb_b},
        ])

        # Query is very close to alice
        query = emb_a + np.random.randn(512).astype(np.float32) * 0.01
        face = FakeFace(bbox=[0, 0, 100, 100], embedding=query)
        pid, name, conf = rec.identify(face)
        assert name == "alice"

    def test_identify_multiple_embeddings_per_person(self):
        """Should take the best score across all embeddings for a person."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)

        rec = self._make_recognizer_with_known([
            {"person_id": 1, "name": "alice", "embedding": emb1},
            {"person_id": 1, "name": "alice", "embedding": emb2},
        ])

        # Query matches emb2 closely
        query = emb2 + np.random.randn(512).astype(np.float32) * 0.01
        face = FakeFace(bbox=[0, 0, 100, 100], embedding=query)
        pid, name, conf = rec.identify(face)
        assert pid == 1
        assert name == "alice"

    def test_identify_null_embedding_on_face(self):
        rec = self._make_recognizer_with_known([
            {"person_id": 1, "name": "alice", "embedding": np.random.randn(512).astype(np.float32)},
        ])
        face = FakeFace(bbox=[0, 0, 100, 100], embedding=None)
        pid, name, conf = rec.identify(face)
        assert pid is None
        assert name == "Unknown"


class TestFaceRecognizerFaceArea:
    def test_face_area(self):
        from recognizer import FaceRecognizer
        with patch.object(FaceRecognizer, '__init__', lambda self: None):
            rec = FaceRecognizer()
        face = FakeFace(bbox=[10, 20, 110, 220])
        assert rec.face_area(face) == 100 * 200


class TestFaceRecognizerGetEmbedding:
    def test_get_embedding_no_faces(self):
        from recognizer import FaceRecognizer
        with patch.object(FaceRecognizer, '__init__', lambda self: None):
            rec = FaceRecognizer()
            rec.app = MagicMock()
            rec.app.get.return_value = []
            rec._known_embeddings = []

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        emb, score = rec.get_embedding(frame)
        assert emb is None
        assert score is None

    def test_get_embedding_picks_best_det_score(self):
        from recognizer import FaceRecognizer
        with patch.object(FaceRecognizer, '__init__', lambda self: None):
            rec = FaceRecognizer()
            rec._known_embeddings = []

        face_low = FakeFace(bbox=[0, 0, 50, 50], det_score=0.5)
        face_high = FakeFace(bbox=[0, 0, 100, 100], det_score=0.95)

        rec.app = MagicMock()
        rec.app.get.return_value = [face_low, face_high]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        emb, score = rec.get_embedding(frame)
        assert score == 0.95
        np.testing.assert_array_equal(emb, face_high.embedding)
