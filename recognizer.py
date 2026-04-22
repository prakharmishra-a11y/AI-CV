"""InsightFace wrapper: detection, embedding generation, and cosine matching."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

from config import INSIGHTFACE_MODEL, CTX_ID, COSINE_THRESHOLD
import database


class FaceRecognizer:
    def __init__(self):
        self.app = FaceAnalysis(name=INSIGHTFACE_MODEL, root="~/.insightface/models")
        self.app.prepare(ctx_id=CTX_ID)
        self._known_embeddings = []  # loaded from DB
        self._refresh_known()

    def _refresh_known(self):
        """Load all known embeddings from DB into memory."""
        rows = database.get_all_embeddings()
        self._known_embeddings = [
            {
                "person_id": pid,
                "name": name,
                "embedding": np.array(emb, dtype=np.float32),
            }
            for pid, name, emb in rows
        ]
        print(f"Loaded {len(self._known_embeddings)} known embeddings.")

    def refresh(self):
        """Public method to reload known embeddings (e.g. after enrollment)."""
        self._refresh_known()

    def detect_faces(self, frame):
        """Detect faces in a BGR frame. Returns list of face objects.

        Each face object has:
          - .bbox: [x1, y1, x2, y2]
          - .embedding: 512d numpy array
          - .det_score: detection confidence
        """
        faces = self.app.get(frame)
        return faces

    def get_embedding(self, frame):
        """Get embedding for the largest face in a frame (used for enrollment).

        Returns (embedding, det_score) or (None, None) if no face found.
        """
        faces = self.detect_faces(frame)
        if not faces:
            return None, None
        # Pick the face with highest detection score
        best = max(faces, key=lambda f: f.det_score)
        return best.embedding, best.det_score

    def identify(self, face):
        """Match a face against known embeddings.

        Returns (person_id, person_name, confidence) or (None, "Unknown", 0.0).
        """
        if face.embedding is None or len(self._known_embeddings) == 0:
            return None, "Unknown", 0.0

        query = face.embedding.reshape(1, -1)
        best_score = 0.0
        best_person = None
        best_name = "Unknown"

        # Group embeddings by person, take best score per person
        person_scores = {}
        for known in self._known_embeddings:
            ref = known["embedding"].reshape(1, -1)
            score = cosine_similarity(query, ref)[0][0]
            pid = known["person_id"]
            if pid not in person_scores or score > person_scores[pid]["score"]:
                person_scores[pid] = {
                    "score": score,
                    "name": known["name"],
                }

        for pid, info in person_scores.items():
            if info["score"] > best_score:
                best_score = info["score"]
                best_person = pid
                best_name = info["name"]

        if best_score >= COSINE_THRESHOLD:
            return best_person, best_name, float(best_score)

        return None, "Unknown", float(best_score)

    def face_area(self, face):
        """Calculate face bounding box area."""
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height
