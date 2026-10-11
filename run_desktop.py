"""
Portable Clinic — Desktop OpenCV Application.
Adapted from arcface_accurate.py with complete clinic attendance & occupancy HUD.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from clinic.config import (
    VIDEO_SOURCE,
    RTSP_URL,
    DISPLAY_MAX_WIDTH,
    SIDEBAR_WIDTH,
    CAM_WIDTH,
    CAM_HEIGHT,
)
from clinic.database import ClinicDatabase
from clinic.face_engine import FaceEngine
from clinic.matcher import Matcher
from clinic.tracker import FaceTracker, PersonRegistry, AdaptiveFrameSkip, BoxSmoother
from clinic.enrollment import (
    enroll_known_faces,
    save_and_enroll_face,
    load_face_database,
)
from clinic.clinic_engine import ClinicEngine
from clinic.camera import CameraManager


def draw_clinic_hud(frame: np.ndarray, status: dict, fps: float, gpu_in_use: bool, provider: str):
    """Draw clinic attendance, doctor status, and occupancy overlay on OpenCV frame."""
    doc = status.get("doctor", {})
    clinic = status.get("clinic", {})
    summary = status.get("summary_today", {})

    is_present = doc.get("present", False)
    doc_name = doc.get("name", "Doctor")
    dur_str = doc.get("current_duration_formatted", "0s")
    occupancy = clinic.get("occupancy", 0)
    unattended = summary.get("unattended_visitors", 0)

    # Top-Left Hardware HUD
    hw_text = f"FPS: {fps:.1f} | GPU: {'ON' if gpu_in_use else 'OFF'} ({provider})"
    cv2.putText(frame, hw_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Top-Right Doctor Status Banner
    banner_text = f"Dr. {doc_name.capitalize()}: {'PRESENT (' + dur_str + ')' if is_present else 'ABSENT'}"
    color = (0, 220, 100) if is_present else (50, 50, 240)
    cv2.putText(frame, banner_text, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Occupancy & Unattended count
    occ_text = f"Occupancy: {occupancy} inside | Unattended Visits: {unattended}"
    cv2.putText(frame, occ_text, (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 200, 0), 2)


def main(source: str | None = None):
    print("=" * 60)
    print("🏥 PORTABLE CLINIC — COMPUTER VISION SYSTEM")
    print("=" * 60)

    if source is None:
        print("\nSelect video source:")
        print("  1 - Built-in / USB Webcam")
        print(f"  2 - RTSP Clinic Camera ({RTSP_URL})")
        choice = input("Choice [1/2, default=1]: ").strip()
        source = "0" if choice != "2" else RTSP_URL

    db = ClinicDatabase()
    engine = FaceEngine()

    face_db = load_face_database()
    if not face_db:
        print("[Info] No existing database. Scanning known_faces/...")
        face_db = enroll_known_faces(engine.app)

    matcher = Matcher(face_db)
    registry = PersonRegistry()
    tracker = FaceTracker(registry)
    clinic = ClinicEngine(db)

    cam = CameraManager(source=source)
    if not cam.connect():
        print("[Error] Could not open camera source. Exiting.")
        return

    print("\n[Controls] N: Enroll new face | R: Rescan database | Q/ESC: Quit\n")
    cv2.namedWindow("Portable Clinic Monitor", cv2.WINDOW_NORMAL)

    fps_count = 0
    fps_start = time.time()
    fps_val = 0.0

    while True:
        ret, frame = cam.read_frame()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Inference
        detections = engine.detect_faces(frame)
        raw = []
        for det in detections:
            box = det["box"]
            emb = det["embedding"]
            name, sim = matcher.match(emb)
            raw.append({"box": box, "name": name, "sim": sim})

        tracked = tracker.update(raw)
        clinic_status = clinic.process_frame(tracked)

        # FPS calculation
        fps_count += 1
        now = time.time()
        if now - fps_start >= 1.0:
            fps_val = round(fps_count / (now - fps_start), 1)
            fps_count = 0
            fps_start = now

        # Draw bounding boxes
        doc_names = clinic.doctor_names
        for item in tracked:
            x1, y1, x2, y2 = item["box"]
            name = item["name"]
            is_doc = clinic.is_doctor(name)

            if is_doc:
                box_color = (0, 255, 100)
                label = f"[DOCTOR] Dr. {name.upper()}"
            elif name != "Unknown":
                box_color = (255, 200, 0)
                label = f"{name} ({item['person_id']})"
            else:
                box_color = (0, 165, 255)
                label = f"Visitor {item['person_id']}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(frame, (x1, y1 - 24), (x1 + max(len(label) * 11, 140), y1), (20, 20, 20), -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        draw_clinic_hud(frame, clinic_status, fps_val, engine.gpu_in_use, engine.provider)

        cv2.imshow("Portable Clinic Monitor", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('r'), ord('R')):
            print("[Action] Rescanning face database...")
            face_db = enroll_known_faces(engine.app)
            matcher = Matcher(face_db)
            registry = PersonRegistry()
            tracker = FaceTracker(registry)
        elif key in (ord('n'), ord('N')):
            name_in = input("\nEnter name for face in camera: ").strip()
            if name_in:
                if save_and_enroll_face(engine.app, frame, name_in):
                    face_db = enroll_known_faces(engine.app)
                    matcher = Matcher(face_db)
                    print(f"[Success] Enrolled '{name_in}'!")

    cam.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
