"""
Face Enrollment & Database Management for Portable Clinic.
Supports multi-pose enrollment, subfolders, and mirrored pose augmentation.
"""

import json
from pathlib import Path
import cv2
import numpy as np
from .config import KNOWN_FACES_DIR, EMBEDDINGS_FILE, NAMES_FILE, EXTS


def _person_name(stem: str) -> str:
    """Extract base person name before first underscore: 'bharat_2' -> 'bharat'."""
    return stem.split("_", 1)[0] if "_" in stem else stem


def save_and_enroll_face(app, image: np.ndarray, name: str) -> bool:
    """
    Detects the largest face in image, crops with margin, and saves to known_faces/<name>.jpg.
    Returns True if face found and saved.
    """
    clean_name = name.strip().replace(" ", "-").lower()
    if not clean_name:
        return False

    faces = app.get(image)
    if not faces:
        return False

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = face.bbox.astype(int)
    h, w = image.shape[:2]
    pad = 30
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = image[y1:y2, x1:x2]

    KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = KNOWN_FACES_DIR / f"{clean_name}.jpg"
    n = 2
    while out_path.exists():
        out_path = KNOWN_FACES_DIR / f"{clean_name}_{n}.jpg"
        n += 1

    cv2.imwrite(str(out_path), crop)
    print(f"[Enrollment] Saved {out_path.name} for identity '{clean_name}'")
    return True


def enroll_known_faces(app) -> dict[str, list[np.ndarray]]:
    """
    Builds embedding database from known_faces/ directory.
    Every image contributes 2 embeddings (original + horizontal flip) for robust pose matching.
    """
    db: dict[str, list[np.ndarray]] = {}
    if not KNOWN_FACES_DIR.exists():
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
        return db

    def add_photo(name: str, path: Path) -> bool:
        img = cv2.imread(str(path))
        if img is None:
            return False
        faces = app.get(img)
        if not faces:
            print(f"[Enrollment] No face detected in: {path.name}")
            return False

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.normed_embedding
        if emb is None:
            return False
        db.setdefault(name.lower(), []).append(emb.astype(np.float32))

        # Horizontal flip augmentation
        flipped = cv2.flip(img, 1)
        flipped_faces = app.get(flipped)
        if flipped_faces:
            fface = max(flipped_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if fface.normed_embedding is not None:
                db[name.lower()].append(fface.normed_embedding.astype(np.float32))
        return True

    added_count = 0
    # Process flat images
    for path in sorted(KNOWN_FACES_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTS:
            if add_photo(_person_name(path.stem), path):
                added_count += 1

    # Process subfolders (known_faces/Name/*.jpg)
    for sub in sorted(KNOWN_FACES_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for path in sorted(sub.iterdir()):
            if path.is_file() and path.suffix.lower() in EXTS:
                if add_photo(sub.name, path):
                    added_count += 1

    print(f"[Enrollment] Enrolled {added_count} photos -> {len(db)} identities: {list(db.keys())}")
    if db:
        save_face_database(db)
    return db


def save_face_database(db: dict[str, list[np.ndarray]]) -> None:
    """Save embeddings array to .npy and names to .json."""
    names: list[str] = []
    rows: list[np.ndarray] = []
    for name, embs in db.items():
        for e in embs:
            names.append(name.lower())
            rows.append(e)

    if not rows:
        return

    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, np.array(rows, dtype=np.float32))
    with open(NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)
    print(f"[Enrollment] Saved database to {EMBEDDINGS_FILE} and {NAMES_FILE}")


def load_face_database() -> dict[str, list[np.ndarray]]:
    """Load embeddings and names from files."""
    if not EMBEDDINGS_FILE.exists() or not NAMES_FILE.exists():
        return {}
    try:
        embs = np.load(EMBEDDINGS_FILE)
        with open(NAMES_FILE, "r", encoding="utf-8") as f:
            names = json.load(f)
        db: dict[str, list[np.ndarray]] = {}
        for name, emb in zip(names, embs):
            db.setdefault(name.lower(), []).append(emb)
        return db
    except Exception as e:
        print(f"[Enrollment] Error loading database: {e}")
        return {}


def get_enrolled_people() -> list[str]:
    """Return sorted list of all enrolled person names."""
    db = load_face_database()
    return sorted(db.keys())
