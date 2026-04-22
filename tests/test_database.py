"""Tests for database.py — CRUD operations on PostgreSQL."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from datetime import datetime, timezone


class TestPersonsCRUD:
    def test_insert_person(self, test_db):
        pid = test_db.insert_person("alice")
        assert pid is not None
        assert isinstance(pid, int)

    def test_get_person_by_name(self, test_db):
        test_db.insert_person("bob")
        row = test_db.get_person_by_name("bob")
        assert row is not None
        assert row[1] == "bob"

    def test_get_person_not_found(self, test_db):
        row = test_db.get_person_by_name("nonexistent_person_xyz")
        assert row is None

    def test_get_or_create_new(self, test_db):
        pid = test_db.get_or_create_person("charlie")
        assert pid is not None
        # Creating again should return same id
        pid2 = test_db.get_or_create_person("charlie")
        assert pid == pid2

    def test_get_all_persons(self, test_db):
        test_db.insert_person("dave")
        test_db.insert_person("eve")
        persons = test_db.get_all_persons()
        names = [p["name"] for p in persons]
        assert "dave" in names
        assert "eve" in names

    def test_delete_person(self, test_db):
        pid = test_db.insert_person("to_delete")
        assert test_db.delete_person(pid) is True
        assert test_db.get_person_by_name("to_delete") is None

    def test_delete_nonexistent_person(self, test_db):
        assert test_db.delete_person(99999) is False


class TestEmbeddingsCRUD:
    def test_insert_and_retrieve_embedding(self, test_db):
        pid = test_db.insert_person("embed_test")
        emb = np.random.randn(512).astype(np.float32)
        test_db.insert_embedding(pid, emb, "/path/to/photo.jpg")

        results = test_db.get_all_embeddings()
        found = [r for r in results if r[1] == "embed_test"]
        assert len(found) == 1
        assert found[0][0] == pid
        assert len(found[0][2]) == 512

    def test_multiple_embeddings_per_person(self, test_db):
        pid = test_db.insert_person("multi_emb")
        for i in range(3):
            emb = np.random.randn(512).astype(np.float32)
            test_db.insert_embedding(pid, emb, f"/path/photo_{i}.jpg")

        results = test_db.get_all_embeddings()
        found = [r for r in results if r[1] == "multi_emb"]
        assert len(found) == 3

    def test_embedding_values_preserved(self, test_db):
        """Verify embedding values survive the round-trip through DB."""
        pid = test_db.insert_person("precision_test")
        original = np.array([0.1, -0.5, 0.999] + [0.0] * 509, dtype=np.float32)
        test_db.insert_embedding(pid, original, "/path.jpg")

        results = test_db.get_all_embeddings()
        found = [r for r in results if r[1] == "precision_test"]
        retrieved = np.array(found[0][2], dtype=np.float32)
        np.testing.assert_allclose(retrieved[:3], [0.1, -0.5, 0.999], atol=1e-4)

    def test_delete_person_cascades_embeddings(self, test_db):
        pid = test_db.insert_person("cascade_test")
        emb = np.random.randn(512).astype(np.float32)
        test_db.insert_embedding(pid, emb, "/path.jpg")

        test_db.delete_person(pid)
        results = test_db.get_all_embeddings()
        found = [r for r in results if r[1] == "cascade_test"]
        assert len(found) == 0


class TestStreamChunks:
    def test_insert_chunk(self, test_db):
        now = datetime.now(timezone.utc)
        chunk_id = test_db.insert_chunk("test_stream", 0, now, now)
        assert chunk_id is not None

    def test_mark_chunk_processed(self, test_db):
        now = datetime.now(timezone.utc)
        chunk_id = test_db.insert_chunk("test_stream", 0, now, now)
        test_db.mark_chunk_processed(chunk_id)
        # No error means success — processed_at is set


class TestSpeakerSightings:
    def test_insert_sighting(self, test_db):
        pid = test_db.insert_person("sighting_test")
        now = datetime.now(timezone.utc)
        chunk_id = test_db.insert_chunk("test_stream", 0, now, now)
        test_db.insert_sighting(chunk_id, pid, now, now, 60, 0.85, 30)

    def test_get_sightings_by_stream(self, test_db):
        pid = test_db.insert_person("stream_q_test")
        now = datetime.now(timezone.utc)
        chunk_id = test_db.insert_chunk("query_stream", 0, now, now)
        test_db.insert_sighting(chunk_id, pid, now, now, 45, 0.75, 20)

        sightings = test_db.get_sightings_by_stream("query_stream")
        assert len(sightings) >= 1
        assert sightings[0]["person_name"] == "stream_q_test"
        assert sightings[0]["duration_seconds"] == 45

    def test_get_sightings_empty_stream(self, test_db):
        sightings = test_db.get_sightings_by_stream("nonexistent_stream")
        assert sightings == []


class TestStreamIndex:
    def test_get_all_stream_ids(self, test_db):
        now = datetime.now(timezone.utc)
        test_db.insert_chunk("idx_stream_1", 0, now, now)
        test_db.insert_chunk("idx_stream_2", 0, now, now)

        streams = test_db.get_all_stream_ids()
        ids = [s["stream_id"] for s in streams]
        assert "idx_stream_1" in ids
        assert "idx_stream_2" in ids
