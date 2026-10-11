"""
Configuration Management for Portable Clinic.
Loads settings from environment variables with robust defaults.
"""

import os
from pathlib import Path

# Base Paths
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_DIR / "data")))

# Sub-directories
KNOWN_FACES_DIR = Path(os.getenv("KNOWN_FACES_DIR", str(DATA_DIR / "known_faces")))
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", str(DATA_DIR / "embeddings")))
EMBEDDINGS_FILE = Path(os.getenv("EMBEDDINGS_FILE", str(EMBEDDINGS_DIR / "face_embeddings.npy")))
NAMES_FILE = Path(os.getenv("NAMES_FILE", str(EMBEDDINGS_DIR / "face_names.json")))
DATABASE_FILE = Path(os.getenv("DATABASE_FILE", str(DATA_DIR / "clinic.db")))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# Ensure necessary directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Clinic & Doctor Business Logic
# ---------------------------------------------------------------------------
# Default designated doctor name(s), comma-separated (e.g. "bharat,doctor,dr-smith")
# Any enrolled person whose name is in this list is treated as a DOCTOR.
# Others are treated as CLINIC STAFF / PATIENTS / VISITORS.
DOCTOR_NAMES_RAW = os.getenv("DOCTOR_NAMES", "bharat,doctor,puli")
DOCTOR_NAMES = [name.strip().lower() for name in DOCTOR_NAMES_RAW.split(",") if name.strip()]

# Grace period (seconds) before marking the doctor as "EXITED" after disappearing from camera.
# Prevents false exit events when doctor turns around or momentarily steps out of frame.
DOCTOR_EXIT_GRACE_SEC = float(os.getenv("DOCTOR_EXIT_GRACE_SEC", "12.0"))

# Grace period (seconds) before marking a visitor as "EXITED" after leaving camera view.
VISITOR_EXIT_GRACE_SEC = float(os.getenv("VISITOR_EXIT_GRACE_SEC", "8.0"))

# ---------------------------------------------------------------------------
# Camera / Video Feed Settings
# ---------------------------------------------------------------------------
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")
RTSP_URL = os.getenv(
    "RTSP_URL",
    "",
)
RTSP_TIMEOUT_MS = int(os.getenv("RTSP_TIMEOUT_MS", "5000"))
STREAM_STALE_SEC = float(os.getenv("STREAM_STALE_SEC", "3.0"))

# Frame Resolutions
CAM_WIDTH = int(os.getenv("CAM_WIDTH", "1280"))
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "720"))
DISPLAY_MAX_WIDTH = int(os.getenv("DISPLAY_MAX_WIDTH", "1280"))
INFER_MAX_WIDTH = int(os.getenv("INFER_MAX_WIDTH", "0"))

# ---------------------------------------------------------------------------
# ArcFace & SCRFD Detection Settings
# ---------------------------------------------------------------------------
DET_WIDTH = int(os.getenv("DET_WIDTH", "640"))
DET_HEIGHT = int(os.getenv("DET_HEIGHT", "640"))
DET_SIZE = (DET_WIDTH, DET_HEIGHT)
_DET_SIZE_EXPLICIT = "DET_WIDTH" in os.environ or "DET_HEIGHT" in os.environ
DET_THRESH = float(os.getenv("DET_THRESH", "0.35"))
MIN_FACE_PX = int(os.getenv("MIN_FACE_PX", "40"))

# ---------------------------------------------------------------------------
# Matching & Thresholds
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.38"))
MATCH_MARGIN = float(os.getenv("MATCH_MARGIN", "0.05"))
SMOOTH_FRAMES = int(os.getenv("SMOOTH_FRAMES", "5"))

# ---------------------------------------------------------------------------
# Face Tracker & Identity Relinking
# ---------------------------------------------------------------------------
TRACK_IOU_THRESH = float(os.getenv("TRACK_IOU_THRESH", "0.25"))
MAX_MISSES = int(os.getenv("MAX_MISSES", "20"))
CENTROID_MATCH_FACTOR = float(os.getenv("CENTROID_MATCH_FACTOR", "0.9"))
RELINK_GRACE_FRAMES = int(os.getenv("RELINK_GRACE_FRAMES", "45"))
RELINK_MAX_DIST_FACTOR = float(os.getenv("RELINK_MAX_DIST_FACTOR", "1.75"))

# ---------------------------------------------------------------------------
# Server / API Settings
# ---------------------------------------------------------------------------
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "1"))

# Supported image file extensions
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
