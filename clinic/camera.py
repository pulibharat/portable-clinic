"""
Camera Manager for Portable Clinic.
RTSP streaming with latest-frame zero-latency grabber and auto-reconnect.
"""

import os
import threading
import time
import cv2
import numpy as np
from .config import (
    VIDEO_SOURCE,
    RTSP_URL,
    RTSP_TIMEOUT_MS,
    STREAM_STALE_SEC,
    CAM_WIDTH,
    CAM_HEIGHT,
)


class StreamGrabber:
    """Background reader thread continuously consuming frames to keep only the newest frame."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ok = False
        self._last_ok_at = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            try:
                ok, frame = self._cap.read()
            except cv2.error:
                ok, frame = False, None

            with self._lock:
                self._ok, self._frame = ok, frame
                if ok:
                    self._last_ok_at = time.monotonic()

            if not ok:
                time.sleep(0.05)

        try:
            self._cap.release()
        except cv2.error:
            pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._ok and time.monotonic() - self._last_ok_at > STREAM_STALE_SEC:
                return False, None
            return self._ok, self._frame

    def stop(self) -> None:
        self._stop = True


class CameraManager:
    """Handles camera connection lifecycle for webcam and RTSP camera feeds."""

    def __init__(self, source: str | None = None):
        self.source = str(source) if source is not None else VIDEO_SOURCE
        self.cap: cv2.VideoCapture | None = None
        self.grabber: StreamGrabber | None = None
        self.is_rtsp = False
        self._lock = threading.Lock()
        self._fail_reads = 0
        self._connected = False

    def connect(self, source: str | None = None) -> bool:
        with self._lock:
            if source is not None:
                self.source = str(source).strip()

            if self.cap is not None:
                self._disconnect_internal()

            src = str(self.source).strip()
            self.is_rtsp = not src.isdigit()

            try:
                if not self.is_rtsp:
                    self.cap = cv2.VideoCapture(int(src))
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                    print(f"[Camera] Connected to local webcam ({src})")
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                        "rtsp_transport;tcp|fflags;nobuffer|max_delay;0"
                    )
                    self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_TIMEOUT_MS)
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_TIMEOUT_MS)
                    print(f"[Camera] Connected to RTSP stream: {src}")

                if not self.cap.isOpened():
                    print(f"[Camera] Failed to open stream: {src}")
                    self._connected = False
                    return False

                if self.is_rtsp:
                    self.grabber = StreamGrabber(self.cap)

                self._connected = True
                self._fail_reads = 0
                return True
            except Exception as e:
                print(f"[Camera] Connection exception: {e}")
                self._connected = False
                return False

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if not self._connected or self.cap is None:
                return False, None

            try:
                if self.grabber is not None:
                    ret, frame = self.grabber.read()
                else:
                    ret, frame = self.cap.read()

                if not ret or frame is None:
                    self._fail_reads += 1
                    if self._fail_reads > 60:
                        print("[Camera] Stream stalled. Disconnecting for reconnect.")
                        self._connected = False
                    return False, None

                self._fail_reads = 0
                return True, frame
            except Exception as e:
                print(f"[Camera] Error reading frame: {e}")
                self._connected = False
                return False, None

    def _disconnect_internal(self) -> None:
        if self.grabber is not None:
            self.grabber.stop()
            self.grabber = None
        if self.cap is not None:
            try:
                self.cap.release()
            except cv2.error:
                pass
            self.cap = None
        self._connected = False

    def disconnect(self) -> None:
        with self._lock:
            self._disconnect_internal()

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected and self.cap is not None and self.cap.isOpened()

    def get_status(self) -> dict:
        return {
            "connected": self.is_connected(),
            "source": self.source,
            "is_rtsp": self.is_rtsp,
            "fail_reads": self._fail_reads,
        }
