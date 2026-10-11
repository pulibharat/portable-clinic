# 🏥 Portable Clinic — Computer Vision System

A real-time computer vision system tailored for clinical containers. Designed to automatically monitor and record **Doctor Attendance** (Entry/Exit timestamps and total duration), track **Clinic Occupancy**, identify **Visitor & Patient Sessions**, and detect **Unattended Clinic Visits** (people entering while the doctor is absent).

Powered by **InsightFace (SCRFD + ArcFace `buffalo_l`)**, **FastAPI**, **WebSockets**, and an interactive real-time **Dashboard**.

---

## 🌟 Key Features

- 👨‍⚕️ **Automated Doctor Attendance**: Real-time identification using ArcFace. Logs doctor entry timestamp, exit timestamp, and active shift duration with debounce grace protection.
- 👥 **Clinic Occupancy & Visitor Tracking**: Tracks the count and dwell time of people inside the clinic.
- ⚠️ **Unattended Visit Alerts**: Automatically flags and logs when visitors enter the container while the registered doctor is away.
- 🎯 **High Pose Tolerance & Sticky Identity**: Uses multi-pose horizontal mirroring and IoU + centroid tracking to maintain recognition through turns and tilts.
- 📊 **Real-time Web Dashboard**: Interactive UI with live video stream, KPI metrics, live event timeline, and session logs.
- 🖥️ **Desktop OpenCV Viewer Option**: Standalone window with live clinic HUD and terminal shortcuts (`N` to enroll, `R` to rescan).
- 🗄️ **Persistent SQLite Database**: Structured audit logs for shifts, visitors, and security events.
- 📥 **One-Click CSV Export**: Download attendance and audit trails for clinic compliance.

---

## 📁 Repository Structure

```
portable_clinic/
├── clinic/
│   ├── __init__.py
│   ├── camera.py             # RTSP / Webcam stream grabber (zero-latency buffering)
│   ├── clinic_engine.py      # Attendance, Occupancy & Unattended visit state machine
│   ├── config.py             # Environment configuration & defaults
│   ├── database.py           # SQLite database persistence layer
│   ├── enrollment.py         # Face enrollment with pose mirroring
│   ├── face_engine.py        # InsightFace / ArcFace model wrapper
│   ├── matcher.py            # Cosine similarity matrix matcher
│   ├── server.py             # FastAPI backend & WebSocket streaming
│   ├── tracker.py            # Multi-person tracking with sticky identity
│   └── static/
│       └── index.html        # Modern Clinic Dashboard UI
├── data/
│   ├── known_faces/          # Enrolled doctor and staff photos
│   ├── embeddings/           # Persistent .npy embeddings & .json name maps
│   └── clinic.db             # SQLite audit database
├── run.py                    # Main launcher (Web Dashboard or Desktop)
├── run_desktop.py            # Standalone OpenCV Desktop viewer
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── Dockerfile                # NVIDIA GPU container build
├── docker-compose.yml        # Multi-container orchestration
└── README.md                 # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Virtual Environment

```bash
# Navigate to the portable_clinic directory
cd portable_clinic

# Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

# For NVIDIA GPU support:
pip install onnxruntime-gpu>=1.15.0

# For Windows DirectML (AMD/Intel/NVIDIA without CUDA toolkit):
pip install onnxruntime-directml>=1.15.0
```

### 3. Configure Camera & Doctor Settings

Copy `.env.example` to `.env` and set your video stream:

```bash
cp .env.example .env
```

Key variables in `.env`:
```ini
# 0 for USB webcam, or RTSP URL for CCTV/NVR camera
VIDEO_SOURCE=0
RTSP_URL=rtsp://admin:password@192.168.1.100:554/stream

# Comma-separated names of designated doctors
DOCTOR_NAMES=bharat,doctor,puli

# Debounce grace time before marking exited (seconds)
DOCTOR_EXIT_GRACE_SEC=12.0
VISITOR_EXIT_GRACE_SEC=8.0
```

---

## 💻 Running the System

### Option A: Web Server & Live Dashboard (Recommended)

```bash
python run.py
```

Then open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

### Option B: Standalone Desktop OpenCV Window

```bash
python run.py --desktop
```

**Desktop Keyboard Shortcuts:**
- `N` — Capture current face from webcam/RTSP, enter name in console to enroll.
- `R` — Rescan `data/known_faces/` folder to reload database.
- `Q` / `ESC` — Quit viewer.

---

## 👨‍⚕️ Face Enrollment Workflow

You can enroll doctors and staff in two ways:

1. **Via Dashboard UI**:
   - Go to the **Doctor Enrollment & Settings** tab.
   - Enter the person's name, choose a photo, select *Register as Clinic Doctor*, and click **Enroll Face**.
2. **Via Folder**:
   - Add images directly into `data/known_faces/`:
     - `data/known_faces/bharat.jpg`
     - `data/known_faces/bharat_side.jpg` (additional poses encouraged)
     - `data/known_faces/dr_smith/front.jpg` (subfolders supported)
   - Click **Rescan** in the dashboard or press `R` in the desktop window.

---

## 🐳 Docker Deployment

The container runs the web dashboard with NVIDIA GPU acceleration. It uses an
RTSP camera stream and keeps face data and the SQLite database in the local
`data/` directory.

### Prerequisites

- Docker Desktop or Docker Engine
- NVIDIA GPU with the NVIDIA driver installed
- NVIDIA Container Toolkit, so Docker can access the GPU
- An RTSP camera reachable from the Docker host

### Configure and start

From the `portable_clinic` directory:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `RTSP_URL` to your camera URL and set `DOCTOR_NAMES` to the enrolled
doctor names. Keep camera credentials in `.env`; do not commit that file.

Start the container:

```bash
docker compose up --build
```

The first build downloads the Python packages and the InsightFace model, so it
can take several minutes. Open the dashboard at `http://localhost:8000`.

Useful commands:

```bash
docker compose logs -f              # Follow application logs
docker compose ps                   # Check container and health status
docker compose down                 # Stop and remove the container
docker compose up -d                # Start in the background
```

The host `data/` directory is mounted at `/app/data` inside the container.
Place enrolled face images under `data/known_faces/<name>/`; generated
embeddings and `clinic.db` remain there after the container is recreated.

The Docker compose setup is for RTSP cameras. For a laptop webcam, run the
application directly on Windows with `python run.py --webcam` instead.

### Shared PostgreSQL database

By default, the application uses local SQLite at `data/clinic.db`. To use a
Render PostgreSQL database, set `DATABASE_URL` in the Render service
environment. Do not commit the URL or put it in frontend code.

```text
DATABASE_URL=postgresql://user:password@host/database
```

The same setting can be used with Docker through the local `.env` file. When
`DATABASE_URL` is present, the application creates and uses these PostgreSQL
tables automatically:

- `clinic_events`
- `doctor_sessions`
- `visitor_sessions`
- `clinic_settings`
- `camera_sources`

The API remains the access boundary for other applications:

```text
GET /api/clinic/status
GET /api/clinic/events?date=YYYY-MM-DD
GET /api/clinic/sessions?date=YYYY-MM-DD
GET /api/clinic/daily-summary?date=YYYY-MM-DD
GET /api/clinic/analytics?date=YYYY-MM-DD
GET /api/export/csv?date=YYYY-MM-DD
```

A phone application should call these HTTPS API endpoints, using server-side
authentication when enabled. It should not connect directly to PostgreSQL or
contain database credentials. The live dashboard stream is available through
the existing `/ws/stream` WebSocket endpoint.

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `VIDEO_SOURCE` | `0` | `0` for webcam, or RTSP URL |
| `DOCTOR_NAMES` | `bharat,doctor,puli` | Comma-separated list of recognized doctor names |
| `DOCTOR_EXIT_GRACE_SEC` | `12.0` | Seconds without detection before marking doctor as left |
| `VISITOR_EXIT_GRACE_SEC` | `8.0` | Seconds without detection before closing visitor session |
| `SIMILARITY_THRESHOLD` | `0.38` | Minimum ArcFace cosine similarity score |
| `MATCH_MARGIN` | `0.05` | Margin required over the second-best match |
| `DET_WIDTH` / `DET_HEIGHT` | `640` | Face detector resolution (auto-drops to 384 on CPU) |
| `FASTAPI_PORT` | `8000` | Port for the web server and dashboard |

---

## 📜 License & Compliance

Designed for clinical operational monitoring and container security compliance.
