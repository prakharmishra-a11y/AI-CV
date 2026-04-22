"""Shared fixtures for face recognition tests."""

import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeFace:
    """Lightweight mock of an InsightFace face object."""

    def __init__(self, bbox, embedding=None, det_score=0.9):
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.embedding = embedding if embedding is not None else np.random.randn(512).astype(np.float32)
        self.det_score = det_score


@pytest.fixture
def fake_face_small():
    """A small face (100x100) — background/audience."""
    return FakeFace(bbox=[500, 500, 600, 600])


@pytest.fixture
def fake_face_large():
    """A large face (300x300) — active speaker."""
    return FakeFace(bbox=[100, 100, 400, 400])


@pytest.fixture
def fake_face_medium():
    """A medium face (200x200) — transition zone."""
    return FakeFace(bbox=[200, 200, 400, 400])


@pytest.fixture
def person_a_embedding():
    """Deterministic embedding for person A."""
    np.random.seed(42)
    return np.random.randn(512).astype(np.float32)


@pytest.fixture
def person_b_embedding():
    """Deterministic embedding for person B (different from A)."""
    np.random.seed(99)
    return np.random.randn(512).astype(np.float32)


@pytest.fixture
def sample_frame():
    """A synthetic BGR image (720p) with no real faces."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def test_db():
    """Create a clean test database schema (uses the real DB connection).

    Yields the module, then cleans up test data after the test.
    """
    import database
    database.create_tables()
    yield database

    # Cleanup: remove test data
    conn = database.get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM speaker_sightings;")
    cur.execute("DELETE FROM stream_chunks;")
    cur.execute("DELETE FROM person_embeddings;")
    cur.execute("DELETE FROM persons;")
    conn.commit()
    cur.close()
    conn.close()
