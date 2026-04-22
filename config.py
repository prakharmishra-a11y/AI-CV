"""Configuration constants for the face recognition speaker tracker."""

# ── Recognition ──────────────────────────────────────────────
COSINE_THRESHOLD = 0.40
INSIGHTFACE_MODEL = "buffalo_l"
CTX_ID = -1  # -1 = CPU, 0 = GPU

# ── Speaker tracking ────────────────────────────────────────
SPEAKER_SIZE_RATIO = 1.5     # largest / second-largest face area
SPEAKER_SWITCH_FRAMES = 15   # consecutive stable frames before switching speaker

# ── Recording ────────────────────────────────────────────────
CHUNK_DURATION = 300         # seconds (5 minutes)
FRAME_SAMPLE_RATE = 30       # sample every Nth frame (1 fps at 30fps)
STREAM_URL = 0               # 0 = webcam, or "rtsp://..." string

# ── Database ─────────────────────────────────────────────────
DATABASE_URL = "dbname=face_recognition user=Vinny host=/var/run/postgresql"

# ── Enrollment ───────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"
MIN_DET_SCORE = 0.7          # warn if reference photo detection score is below this
