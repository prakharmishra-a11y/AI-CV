# Face Recognition Speaker Tracker

Detect and identify enrolled people in live camera or video streams, and track who spoke (or appeared most prominently) and for how long. Built on **RetinaFace** (face detection) + **ArcFace** (512-D embeddings, cosine similarity) via the `insightface` toolkit, with a FastAPI HTTP layer and a PostgreSQL backend for persistence.

## Features

- Enroll people from photos via CLI (`enroll.py`) or REST API (`POST /enroll`).
- Live pipeline (`main.py`) that records the source in chunks and processes them in parallel for low-latency speaker timeline output.
- One-shot scanners for offline files (`scan_video.py`, `scan_clip.py`).
- Speaker tracking heuristic — the largest stable face on screen is treated as the active speaker, with hysteresis to avoid flicker.
- Sightings (who, when, duration, confidence, frame count) persisted to Postgres and queryable per session via the API.

## Stack

- **Python 3.10+**
- `insightface` (`buffalo_l` model) + `onnxruntime` (CPU by default; flip `CTX_ID` in `config.py` for GPU)
- `opencv-python` for capture/decoding
- `scikit-learn` for cosine similarity
- `fastapi` + `uvicorn` for the HTTP layer
- `psycopg2-binary` + PostgreSQL for storage

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Postgres is required. Edit `DATABASE_URL` in `config.py` to match your local instance — the default uses a Unix socket and database `face_recognition`. Tables are created automatically on first run.

Other tunables in `config.py`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `COSINE_THRESHOLD` | `0.40` | Minimum cosine similarity to count as a match |
| `INSIGHTFACE_MODEL` | `"buffalo_l"` | InsightFace model pack |
| `CTX_ID` | `-1` | `-1` = CPU, `0` = first GPU |
| `SPEAKER_SIZE_RATIO` | `1.5` | How much larger a face must be to be the speaker |
| `SPEAKER_SWITCH_FRAMES` | `15` | Stable frames required before switching speaker |
| `CHUNK_DURATION` | `300` | Seconds per recorded chunk |
| `FRAME_SAMPLE_RATE` | `30` | Process every Nth frame |
| `STREAM_URL` | `0` | Default source: webcam index, RTSP URL, or file path |
| `MIN_DET_SCORE` | `0.7` | Warn if enrollment photo's face score is below this |

## Enrollment

**CLI** — drop photos into `known_faces/` named `<person>.jpg` (or `<person>_1.jpg`, `<person>_2.jpg` for multiple shots), then:

```bash
python enroll.py
```

**API** — start the server (see below) and:

```bash
curl -X POST "http://localhost:8000/enroll?name=Prakhar" \
     -F "photo=@/path/to/face.jpg"
```

## Live pipeline

```bash
python main.py                                      # webcam (index 0)
python main.py --stream rtsp://192.168.1.1/stream   # RTSP
python main.py --stream /path/to/video.mp4          # file
python main.py --output results.json                # save sightings to JSON
```

The recorder thread saves chunks while the processor thread runs detection, recognition, and speaker attribution on each. Sightings stream into Postgres as they're produced; `Ctrl+C` shuts down gracefully.

There's also `webcam_test.py` for a quick live-overlay sanity check.

## Offline scanning

```bash
python scan_video.py /path/to/video.mp4    # full video, sampled frames
python scan_clip.py /path/to/clip.mp4      # short clip
```

## REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/enroll?name=<name>` (multipart `photo`) | Enroll one photo |
| `GET` | `/persons` | List enrolled persons |
| `DELETE` | `/persons/{id}` | Remove a person and their embeddings |
| `GET` | `/results` | List processed sessions |
| `GET` | `/results/{stream_id}` | Sightings for a session |

Interactive docs at `http://localhost:8000/docs`.

## Tests

```bash
pytest
```

Test suite covers the API, recognizer, processor, recorder, database layer, and config — using `unittest.mock` to stub the InsightFace model and a `FakeFace` fixture from `tests/conftest.py`.

## Repository layout

```
api.py                FastAPI app + endpoints
main.py               Live pipeline orchestrator (recorder + processor threads)
recorder.py           Chunked video capture
processor.py          Per-chunk detection, recognition, speaker timeline
recognizer.py         InsightFace wrapper + cosine matching
speaker_tracker.py    "Who is currently speaking" heuristic
database.py           Postgres schema + queries
enroll.py             CLI enrollment from known_faces/
scan_video.py         Offline video scan
scan_clip.py          Offline short-clip scan
webcam_test.py        Live overlay debug tool
config.py             All tunables
tests/                pytest suite
```
