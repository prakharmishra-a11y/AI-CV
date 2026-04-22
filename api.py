"""FastAPI REST API for the face recognition speaker tracker.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST   /enroll           — Upload a photo + person name to enroll
    GET    /persons          — List all enrolled persons
    DELETE /persons/{id}     — Delete a person and their embeddings
    GET    /results          — List all processed sessions
    GET    /results/{id}     — Get sightings for a specific session
"""

import os
import uuid
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, UploadFile, File, Query, HTTPException

import database
from config import KNOWN_FACES_DIR, MIN_DET_SCORE
from recognizer import FaceRecognizer

# ── Globals ──────────────────────────────────────────────────
recognizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and DB on startup."""
    global recognizer
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    database.create_tables()
    print("Loading InsightFace model...")
    recognizer = FaceRecognizer()
    print("API ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Face Recognition Speaker Tracker",
    description="""
## Overview
Enroll faces via photo upload and query speaker tracking results from live camera streams.

## How it works
1. **Enroll** known people by uploading their face photos
2. **Run the live pipeline** (`python main.py --stream <source>`) to process camera streams
3. **Query results** to see who spoke, when, and for how long

## Models
- **RetinaFace** (detection) — finds faces in each frame
- **ArcFace** (recognition) — generates 512D face embeddings, matches via cosine similarity (threshold ≥ 0.40)

## CLI Usage
```
python main.py                                    # webcam
python main.py --stream rtsp://192.168.1.1/stream  # RTSP
python main.py --stream /path/to/video.dav         # file playback
python webcam_test.py                              # live overlay demo
```
""",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Enrollment ───────────────────────────────────────────────

@app.post("/enroll", tags=["enrollment"])
async def enroll_person(
    name: str = Query(..., description="Person's name (e.g. 'Prakhar')"),
    photo: UploadFile = File(..., description="Face photo (jpg/png)"),
):
    """Enroll a person by uploading their face photo.

    - Upload one photo at a time
    - Call multiple times for multiple photos of the same person
    - Detects the face, generates a 512D embedding, stores in DB
    """
    # Save upload to disk
    ext = os.path.splitext(photo.filename)[1] or ".jpg"
    save_name = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = os.path.join(KNOWN_FACES_DIR, save_name)

    content = await photo.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Read with OpenCV and extract embedding
    img = cv2.imread(save_path)
    if img is None:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail="Could not read image file.")

    embedding, det_score = recognizer.get_embedding(img)
    if embedding is None:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail="No face detected in the photo.")

    warning = None
    if det_score < MIN_DET_SCORE:
        warning = f"Low detection score ({det_score:.3f}) — may be low quality."

    person_id = database.get_or_create_person(name)
    database.insert_embedding(person_id, embedding, save_path)
    recognizer.refresh()

    result = {
        "message": f"Enrolled '{name}' successfully.",
        "person_id": person_id,
        "det_score": round(float(det_score), 4),
        "embedding_dim": len(embedding),
    }
    if warning:
        result["warning"] = warning

    return result


# ── Persons ──────────────────────────────────────────────────

@app.get("/persons", tags=["persons"])
async def list_persons():
    """List all enrolled persons."""
    return {"persons": database.get_all_persons()}


@app.delete("/persons/{person_id}", tags=["persons"])
async def delete_person(person_id: int):
    """Delete a person and all their embeddings."""
    deleted = database.delete_person(person_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Person not found.")
    recognizer.refresh()
    return {"message": f"Person {person_id} deleted."}


# ── Results ──────────────────────────────────────────────────

@app.get("/results", tags=["results"])
async def list_results():
    """List all processed stream sessions."""
    return {"sessions": database.get_all_stream_ids()}


@app.get("/results/{stream_id}", tags=["results"])
async def get_results(stream_id: str):
    """Get speaker sightings for a specific session."""
    sightings = database.get_sightings_by_stream(stream_id)
    if not sightings:
        raise HTTPException(status_code=404, detail="No results found for this session.")
    return {"stream_id": stream_id, "sightings": sightings}
