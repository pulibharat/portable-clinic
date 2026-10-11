"""
Main Entry Point for Portable Clinic System.

Usage:
    python run.py           # Prompts for camera, starts FastAPI + Dashboard (http://localhost:8000)
    python run.py --webcam  # Uses laptop webcam directly
    python run.py --rtsp    # Uses RTSP stream directly
    python run.py --desktop # Starts standalone OpenCV Desktop window
"""

import os
import sys
import argparse
from pathlib import Path

# Add directory to Python path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))


def choose_camera_source(args) -> str:
    """Determine camera source from CLI flags or interactive prompt."""
    if args.webcam:
        return "0"
    if args.rtsp:
        from clinic.config import RTSP_URL
        return RTSP_URL
    if args.source is not None:
        return str(args.source)

    # Interactive prompt (same as arcface_accurate.py)
    print("\nSelect video source:")
    print("  1 - Laptop webcam (0)")
    print("  2 - RTSP camera (NVR)")
    choice = input("Enter 1 or 2 (default 1): ").strip()

    if choice == "2":
        from clinic.config import RTSP_URL
        print("-> Selected RTSP camera")
        return RTSP_URL

    print("-> Selected Laptop webcam")
    return "0"


def main():
    parser = argparse.ArgumentParser(description="Portable Clinic — Vision Monitoring & Attendance System")
    parser.add_argument("--desktop", action="store_true", help="Launch OpenCV desktop GUI window instead of web server")
    parser.add_argument("--webcam", action="store_true", help="Use laptop/USB webcam (source 0) without asking")
    parser.add_argument("--rtsp", action="store_true", help="Use RTSP camera stream without asking")
    parser.add_argument("--source", type=str, default=None, help="Custom video source (0, 1, or rtsp://...)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="FastAPI host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    args = parser.parse_args()

    selected_source = choose_camera_source(args)
    os.environ["VIDEO_SOURCE"] = selected_source

    if args.desktop:
        from run_desktop import main as run_desktop_main
        run_desktop_main(source=selected_source)
    else:
        import uvicorn
        from clinic.server import app
        print("\n" + "=" * 65)
        print("🏥 PORTABLE CLINIC MONITOR SERVER STARTING")
        print(f"👉 Web Dashboard: http://localhost:{args.port}")
        print(f"👉 Camera Source: {'Webcam (0)' if selected_source == '0' else selected_source}")
        print("=" * 65 + "\n")
        uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
