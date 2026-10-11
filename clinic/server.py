"""
FastAPI Server & WebSocket Streaming for Portable Clinic.
Serves the web dashboard, REST APIs for audit logs, and real-time inference streaming.
"""

import asyncio
import base64
import io
import json
import threading
import time
from urllib.parse import quote, urlsplit, urlunsplit
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .config import FASTAPI_HOST, FASTAPI_PORT, KNOWN_FACES_DIR, RTSP_URL
from .database import ClinicDatabase
from .camera import CameraManager
from .face_engine import FaceEngine
from .matcher import Matcher
from .tracker import FaceTracker, PersonRegistry, AdaptiveFrameSkip, BoxSmoother
from .enrollment import (
    enroll_known_faces,
    save_and_enroll_face,
    load_face_database,
    get_enrolled_people,
)
from .clinic_engine import ClinicEngine

# ============================================================================
# Application Setup
# ============================================================================
app = FastAPI(
    title="Portable Clinic — Computer Vision System",
    description="Automated Doctor Attendance, Patient/Visitor Tracking, and Occupancy Monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def normalize_camera_source(source: str) -> str:
        """Convert a CP Plus web root URL into its RTSP video stream URL."""
        source = source.strip()
        parsed = urlsplit(source)
        if parsed.scheme in {"http", "https"} and parsed.hostname and parsed.path in {"", "/"}:
            return urlunsplit((
                "rtsp",
                parsed.netloc,
                "/cam/realmonitor",
                "channel=3&subtype=1",
                "",
            ))
        return source

# Shared State
db: Optional[ClinicDatabase] = None
face_engine: Optional[FaceEngine] = None
camera_manager: Optional[CameraManager] = None
matcher: Optional[Matcher] = None
tracker: Optional[FaceTracker] = None
person_registry: Optional[PersonRegistry] = None
clinic_engine: Optional[ClinicEngine] = None

# Thread Control
inference_thread: Optional[threading.Thread] = None
inference_stop_event = threading.Event()

# Latest Frame & Real-time State
latest_frame: Optional[np.ndarray] = None
latest_tracked_items: list[dict] = []
latest_clinic_status: dict = {}
fps_counter = {"count": 0, "start": time.time(), "fps": 0.0}


# ============================================================================
# Lifecycle
# ============================================================================
@app.on_event("startup")
async def startup_event():
    global db, face_engine, camera_manager, matcher, tracker, person_registry, clinic_engine
    global inference_thread, inference_stop_event

    print("[Startup] Initializing Clinic Database...")
    db = ClinicDatabase()
    registered_cameras = db.get_camera_sources()
    if not any(camera["source_url"] == RTSP_URL for camera in registered_cameras):
        db.add_camera_source("Camera 01", "VSR Boys Hostel Clinic", RTSP_URL)

    print("[Startup] Initializing Face Recognition Engine (ArcFace)...")
    face_engine = FaceEngine()

    print("[Startup] Loading Enrolled Faces...")
    face_db = load_face_database()
    if not face_db:
        print(
            "[Startup] No embeddings found. Performing initial scan of known_faces/...")
        face_db = enroll_known_faces(face_engine.app)

    matcher = Matcher(face_db)
    person_registry = PersonRegistry()
    tracker = FaceTracker(person_registry)

    print("[Startup] Initializing Clinic Engine...")
    clinic_engine = ClinicEngine(db)

    print("[Startup] Connecting Camera...")
    camera_manager = CameraManager()
    if not camera_manager.connect():
        print("[Startup] Camera connection deferred — background retry loop active.")

    print("[Startup] Starting Background Inference Loop...")
    inference_stop_event.clear()
    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    inference_thread.start()
    print("[Startup] Portable Clinic System Ready!")


@app.on_event("shutdown")
async def shutdown_event():
    global inference_stop_event, camera_manager
    print("[Shutdown] Stopping inference loop...")
    inference_stop_event.set()
    if inference_thread:
        inference_thread.join(timeout=3.0)
    if camera_manager:
        camera_manager.disconnect()
    print("[Shutdown] Completed.")


# ============================================================================
# Background Inference Loop
# ============================================================================
def inference_loop():
    global latest_frame, latest_tracked_items, latest_clinic_status, fps_counter
    reconnect_delay = 2.0

    while not inference_stop_event.is_set():
        if not camera_manager.is_connected():
            time.sleep(reconnect_delay)
            if not inference_stop_event.is_set():
                camera_manager.connect()
            continue

        ret, frame = camera_manager.read_frame()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        try:
            # 1. Run face detection & ArcFace feature extraction
            detections = face_engine.detect_faces(frame)
            raw = []
            for det in detections:
                box = det["box"]
                emb = det["embedding"]
                name, sim = matcher.match(emb)
                raw.append({"box": box, "name": name, "sim": sim})

            # 2. Update multi-person face tracker
            tracked = tracker.update(raw)
            latest_tracked_items = tracked

            # 3. Process Clinic Business Logic
            status = clinic_engine.process_frame(tracked)
            latest_clinic_status = status
            latest_frame = frame.copy()

            # 4. Update FPS metric
            fps_counter["count"] += 1
            now = time.time()
            if now - fps_counter["start"] >= 1.0:
                fps_counter["fps"] = round(
                    fps_counter["count"] / (now - fps_counter["start"]), 1)
                fps_counter["count"] = 0
                fps_counter["start"] = now

        except Exception as e:
            print(f"[Inference Error] {e}")
            time.sleep(0.02)

        time.sleep(0.005)


# ============================================================================
# REST Endpoints
# ============================================================================
@app.get("/")
async def get_index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"message": "Portable Clinic API is Running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gpu_in_use": face_engine.gpu_in_use if face_engine else False,
        "onnx_provider": face_engine.provider if face_engine else "unknown",
        "camera_connected": camera_manager.is_connected() if camera_manager else False,
        "database_backend": "postgresql" if db and db.database_url else "sqlite",
    }


@app.get("/api/doctor/photo")
async def get_doctor_photo(name: str = Query("bharat", min_length=1, max_length=100)):
    """Return the first enrolled photo for a doctor, when available."""
    safe_name = Path(name).name.lower()
    person_dir = KNOWN_FACES_DIR / safe_name
    candidates = []
    if person_dir.exists():
        preferred_names = (
            (("meeee.jpeg",) if safe_name == "bharat" else ())
            + (
                f"{safe_name}1.jpeg",
                f"{safe_name}1.jpg",
                f"{safe_name}.jpeg",
                f"{safe_name}.jpg",
            )
        )
        for preferred_name in preferred_names:
            preferred = person_dir / preferred_name
            if preferred.is_file():
                return FileResponse(str(preferred))
        candidates.extend(
            path for path in person_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
    if not candidates:
        raise HTTPException(status_code=404, detail="No enrolled photo found")
    return FileResponse(str(sorted(candidates)[0]))


@app.get("/api/clinic/status")
async def get_clinic_status():
    """Get current snapshot of doctor presence, occupancy, and today's summary."""
    if not clinic_engine:
        raise HTTPException(status_code=503, detail="System initializing")
    return {
        "status": latest_clinic_status,
        "camera_connected": camera_manager.is_connected() if camera_manager else False,
        "fps": fps_counter["fps"],
    }


@app.get("/api/clinic/events")
async def get_events(limit: int = Query(50, ge=1, le=500), date: Optional[str] = None):
    """Retrieve audit timeline events."""
    if not db:
        raise HTTPException(status_code=503, detail="Database initializing")
    events = db.get_recent_events(limit=limit, date_filter=date)
    return {"events": events}


@app.get("/api/clinic/sessions")
async def get_sessions(date: Optional[str] = None):
    """Retrieve doctor and visitor session history."""
    if not db:
        raise HTTPException(status_code=503, detail="Database initializing")
    doc_sessions = db.get_doctor_sessions_today(target_date=date)
    vis_sessions = db.get_visitor_sessions_today(target_date=date)
    return {
        "doctor_sessions": doc_sessions,
        "visitor_sessions": vis_sessions,
    }


@app.get("/api/clinic/daily-summary")
async def get_daily_summary(date: Optional[str] = None):
    """Retrieve aggregated daily attendance and clinic occupancy summary."""
    if not db:
        raise HTTPException(status_code=503, detail="Database initializing")
    return db.get_daily_summary(target_date=date)


@app.get("/api/clinic/analytics")
async def get_clinic_analytics(date: Optional[str] = None):
    """Return summary and chart data for the analytics view."""
    if not db:
        raise HTTPException(status_code=503, detail="Database initializing")
    return {"summary": db.get_daily_summary(target_date=date), "analytics": db.get_analytics(target_date=date)}


@app.get("/api/system/status")
async def get_system_status():
    """Return operational health data for the system status view."""
    return {
        "service": "Portable Clinic",
        "status": "online",
        "camera": camera_manager.get_status() if camera_manager else {"connected": False},
        "fps": fps_counter["fps"],
        "gpu_in_use": face_engine.gpu_in_use if face_engine else False,
        "onnx_provider": face_engine.provider if face_engine else "unknown",
        "tracked_people": len(latest_tracked_items),
        "database": "ready" if db else "initializing",
        "database_backend": "postgresql" if db and db.database_url else "sqlite",
    }


@app.get("/api/admin/cameras")
async def list_camera_configuration():
    """Expose the configured camera source for administration screens."""
    if not camera_manager:
        raise HTTPException(
            status_code=503, detail="Camera manager initializing")
    status = camera_manager.get_status()
    return {"cameras": [{"id": "camera-01", "name": "Primary clinic camera", **status}]}


@app.post("/api/assistant")
async def clinic_assistant(question: str = Form(...)):
    """Answer common clinic questions from current system data without inventing facts."""
    if not db:
        raise HTTPException(status_code=503, detail="Database initializing")
    text = question.lower()
    summary = db.get_daily_summary()
    current = latest_clinic_status or {}
    clinic = current.get("clinic", {})
    doctor = current.get("doctor", {})
    if "occup" in text or "inside" in text or "visitor" in text:
        answer = f"There are {clinic.get('occupancy', 0)} people inside, including {clinic.get('active_visitors_count', 0)} visitors or patients."
    elif "doctor" in text or "attendance" in text:
        answer = f"The doctor is currently {'present' if doctor.get('present') else 'absent'}. Today's recorded doctor sessions: {summary['doctor_sessions_count']}."
    elif "peak" in text or "today" in text or "report" in text:
        answer = f"Today has {summary['total_visitors']} visitor sessions. Average visit duration is {db.format_seconds(summary['avg_visitor_duration_sec'])}."
    else:
        answer = "Ask about current occupancy, visitor count, doctor attendance, peak time, or today's report."
    return {"question": question, "answer": answer}


@app.post("/api/settings/grace")
async def set_grace_periods(doc_grace: float = Form(...), vis_grace: float = Form(...)):
    """Update doctor and visitor exit grace thresholds."""
    if not clinic_engine:
        raise HTTPException(status_code=503, detail="System initializing")
    clinic_engine.set_grace_periods(doc_grace, vis_grace)
    return {
        "success": True,
        "doctor_exit_grace_sec": clinic_engine.doctor_exit_grace_sec,
        "visitor_exit_grace_sec": clinic_engine.visitor_exit_grace_sec,
    }


@app.post("/api/doctor/set")
async def set_registered_doctors(names: str = Form(...)):
    """Configure list of recognized doctor names (comma-separated)."""
    if not clinic_engine:
        raise HTTPException(status_code=503, detail="System initializing")
    name_list = [n.strip() for n in names.split(",") if n.strip()]
    clinic_engine.set_doctor_names(name_list)
    return {"success": True, "registered_doctors": clinic_engine.doctor_names}


@app.get("/api/doctor/list")
async def list_doctors():
    """List designated doctors and all enrolled persons."""
    enrolled = get_enrolled_people()
    registered = clinic_engine.doctor_names if clinic_engine else []
    return {
        "registered_doctors": registered,
        "all_enrolled_persons": enrolled,
    }


@app.post("/api/enroll")
async def enroll_person(name: str = Form(...), image: UploadFile = File(...), is_doctor: bool = Form(False)):
    """Enroll a new face (Doctor or Staff/Patient)."""
    if not face_engine:
        raise HTTPException(status_code=503, detail="Face engine initializing")

    try:
        content = await image.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(
                status_code=400, detail="Invalid image file format.")

        saved = save_and_enroll_face(face_engine.app, img, name)
        if not saved:
            raise HTTPException(
                status_code=400, detail="No face detected in the uploaded photo.")

        # Reload database
        face_db = load_face_database()
        global matcher, tracker, person_registry
        matcher = Matcher(face_db)
        person_registry = PersonRegistry()
        tracker = FaceTracker(person_registry)

        if is_doctor and clinic_engine:
            if name.lower() not in clinic_engine.doctor_names:
                clinic_engine.doctor_names.append(name.lower())
                clinic_engine.set_doctor_names(clinic_engine.doctor_names)

        return {
            "success": True,
            "name": name.lower(),
            "enrolled_count": len(face_db),
            "people": sorted(face_db.keys()),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/rescan")
async def rescan_database():
    """Rescan known_faces directory to rebuild embedding database."""
    if not face_engine:
        raise HTTPException(status_code=503, detail="Face engine initializing")

    face_db = enroll_known_faces(face_engine.app)
    global matcher, tracker, person_registry
    matcher = Matcher(face_db)
    person_registry = PersonRegistry()
    tracker = FaceTracker(person_registry)

    return {"success": True, "enrolled_count": len(face_db), "people": sorted(face_db.keys())}


@app.post("/api/camera/connect")
async def camera_connect(source: Optional[str] = Form(None)):
    if not camera_manager:
        raise HTTPException(
            status_code=503, detail="Camera manager initializing")
    if source:
        camera_manager.source = normalize_camera_source(source)
    ok = camera_manager.connect()
    return {"success": ok, "status": camera_manager.get_status()}


@app.post("/api/camera/disconnect")
async def camera_disconnect():
    if not camera_manager:
        raise HTTPException(
            status_code=503, detail="Camera manager initializing")
    camera_manager.disconnect()
    return {"success": True, "status": camera_manager.get_status()}


@app.get("/api/camera/status")
async def camera_status():
    if not camera_manager:
        raise HTTPException(
            status_code=503, detail="Camera manager initializing")
    return {
        "camera": camera_manager.get_status(),
        "fps": fps_counter["fps"],
        "gpu_in_use": face_engine.gpu_in_use if face_engine else False,
    }


@app.get("/api/cameras")
async def list_camera_sources():
    if not db:
        raise HTTPException(status_code=503, detail="Database initializing")
    return {"cameras": db.get_camera_sources()}


@app.post("/api/cameras")
async def register_camera(
    camera_name: str = Form(...),
    location: str = Form(...),
    source_url: str = Form(...),
    username: str = Form(""),
    password: str = Form(""),
):
    if not db:
        raise HTTPException(status_code=503, detail="Database initializing")
    source_url = normalize_camera_source(source_url)
    if username or password:
        parsed = urlsplit(source_url)
        if parsed.scheme and parsed.hostname:
            auth = quote(username, safe="")
            if password:
                auth += ":" + quote(password, safe="")
            source_url = urlunsplit((
                parsed.scheme,
                f"{auth}@{parsed.hostname}:{parsed.port}" if parsed.port else f"{auth}@{parsed.hostname}",
                parsed.path,
                parsed.query,
                parsed.fragment,
            ))
    if not camera_name.strip() or not location.strip() or not source_url:
        raise HTTPException(
            status_code=400, detail="Camera name, location, and RTSP URL are required")
    camera_id = db.add_camera_source(camera_name, location, source_url)
    return {"success": True, "camera_id": camera_id, "cameras": db.get_camera_sources()}


@app.delete("/api/cameras/{camera_id}")
async def remove_camera(camera_id: int):
    if not db:
        raise HTTPException(status_code=503, detail="Database initializing")
    db.deactivate_camera_source(camera_id)
    return {"success": True, "cameras": db.get_camera_sources()}


@app.get("/api/export/csv")
async def export_audit_csv(date: Optional[str] = None):
    """Export complete clinic events log as CSV."""
    if not db:
        raise HTTPException(status_code=503, detail="Database initializing")

    events = db.get_recent_events(limit=5000, date_filter=date)
    output = io.StringIO()
    output.write("ID,Timestamp,Date,EventType,PersonName,PersonID,Details\n")
    for ev in events:
        safe_details = str(ev['details']).replace('"', '""')
        output.write(
            f'{ev["id"]},{ev["timestamp"]},{ev["date"]},{ev["event_type"]},{ev["person_name"]},{ev["person_id"]},"{safe_details}"\n')

    return PlainTextResponse(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=clinic_audit_{date or 'all'}.csv"}
    )


def json_safe_default(obj):
    """Ensure numpy and special types serialize cleanly to JSON."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, date)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    return str(obj)


# ============================================================================
# Real-Time WebSocket Streaming
# ============================================================================
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Dashboard client connected.")

    try:
        last_sent = 0.0
        while True:
            now = time.time()
            if (now - last_sent) < 0.066:  # ~15 FPS WebSocket throttle
                await asyncio.sleep(0.01)
                continue

            last_sent = now

            # Prepare real-time packet
            packet = {
                "timestamp": now,
                "fps": fps_counter["fps"],
                "camera_connected": camera_manager.is_connected() if camera_manager else False,
                "gpu_in_use": face_engine.gpu_in_use if face_engine else False,
                "tracked_items": latest_tracked_items,
                "clinic_status": latest_clinic_status,
            }

            # Encode frame to JPEG
            if latest_frame is not None:
                ret, buf = cv2.imencode(".jpg", latest_frame, [
                                        cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    packet["frame"] = base64.b64encode(buf).decode("utf-8")

            payload = json.dumps(packet, default=json_safe_default)
            await websocket.send_text(payload)

    except Exception as e:
        print(f"[WebSocket Loop Note] Client session ended: {e}")
    finally:
        print("[WebSocket] Dashboard client disconnected.")


def run_server():
    uvicorn.run(
        app,
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    run_server()
