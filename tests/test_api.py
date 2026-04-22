"""Tests for api.py — FastAPI endpoints."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from tests.conftest import FakeFace


@pytest.fixture(scope="module")
def client():
    """Create a test client with mocked recognizer."""
    mock_recognizer = MagicMock()
    mock_recognizer.get_embedding.return_value = (
        np.random.randn(512).astype(np.float32),
        0.9,
    )
    mock_recognizer.detect_faces.return_value = []
    mock_recognizer.identify.return_value = (None, "Unknown", 0.0)

    with patch("api.FaceRecognizer", return_value=mock_recognizer):
        import api
        api.recognizer = mock_recognizer
        with TestClient(api.app) as c:
            yield c


@pytest.fixture
def face_image_bytes():
    """Generate a JPEG image in memory."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = (200, 180, 160)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class TestPersonsEndpoint:
    def test_list_persons(self, client):
        resp = client.get("/persons")
        assert resp.status_code == 200
        assert "persons" in resp.json()
        assert isinstance(resp.json()["persons"], list)

    def test_delete_nonexistent_person(self, client):
        resp = client.delete("/persons/99999")
        assert resp.status_code == 404


class TestEnrollEndpoint:
    def test_enroll_success(self, client, face_image_bytes):
        resp = client.post(
            "/enroll",
            data={"name": "api_test_user"},
            files=[("photos", ("test.jpg", face_image_bytes, "image/jpeg"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "enrolled" in data
        assert len(data["enrolled"]) == 1
        assert data["enrolled"][0]["embedding_dim"] == 512

    def test_enroll_multiple_photos(self, client, face_image_bytes):
        resp = client.post(
            "/enroll",
            data={"name": "multi_photo_user"},
            files=[
                ("photos", ("front.jpg", face_image_bytes, "image/jpeg")),
                ("photos", ("side.jpg", face_image_bytes, "image/jpeg")),
            ],
        )
        assert resp.status_code == 200
        assert len(resp.json()["enrolled"]) == 2

    def test_enroll_no_face_detected(self, client):
        """When recognizer finds no face, should return 400."""
        import api
        api.recognizer.get_embedding.return_value = (None, None)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)

        resp = client.post(
            "/enroll",
            data={"name": "noface"},
            files=[("photos", ("blank.jpg", buf.tobytes(), "image/jpeg"))],
        )
        assert resp.status_code == 400

        # Restore
        api.recognizer.get_embedding.return_value = (
            np.random.randn(512).astype(np.float32), 0.9,
        )

    def test_enroll_low_det_score_warning(self, client, face_image_bytes):
        """Low detection score should produce a warning but still enroll."""
        import api
        api.recognizer.get_embedding.return_value = (
            np.random.randn(512).astype(np.float32), 0.5,
        )

        resp = client.post(
            "/enroll",
            data={"name": "low_score_user"},
            files=[("photos", ("low.jpg", face_image_bytes, "image/jpeg"))],
        )
        assert resp.status_code == 200
        assert len(resp.json()["warnings"]) > 0

        # Restore
        api.recognizer.get_embedding.return_value = (
            np.random.randn(512).astype(np.float32), 0.9,
        )


class TestResultsEndpoint:
    def test_list_results(self, client):
        resp = client.get("/results")
        assert resp.status_code == 200
        assert "sessions" in resp.json()

    def test_get_nonexistent_result(self, client):
        resp = client.get("/results/nonexistent_stream_id")
        assert resp.status_code == 404


class TestOpenAPISchema:
    def test_openapi_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Face Recognition Speaker Tracker"

    def test_all_endpoints_in_schema(self, client):
        resp = client.get("/openapi.json")
        paths = resp.json()["paths"]
        assert "/enroll" in paths
        assert "/persons" in paths
        assert "/persons/{person_id}" in paths
        assert "/results" in paths
        assert "/results/{stream_id}" in paths
        # /process should NOT exist
        assert "/process" not in paths

    def test_swagger_ui_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower()
