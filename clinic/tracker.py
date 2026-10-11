"""
Tracker for Portable Clinic.
Per-face tracking with sticky identity, centroid fallback, and track relinking.
"""

from collections import deque
import numpy as np
from .config import (
    TRACK_IOU_THRESH,
    CENTROID_MATCH_FACTOR,
    RELINK_GRACE_FRAMES,
    RELINK_MAX_DIST_FACTOR,
    MAX_MISSES,
    SMOOTH_FRAMES,
)


def _iou(a: list[int], b: list[int]) -> float:
    """Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)


def _centroid(box: list[int]) -> tuple[float, float]:
    """Calculate center coordinate of a box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _centroid_close(a: list[int], b: list[int], factor: float) -> bool:
    """Check if box A center is within `factor` box widths of box B center."""
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    ref_width = max(a[2] - a[0], b[2] - b[0], 1)
    return dist <= ref_width * factor


class PersonRegistry:
    """Assigns each recognized NAME a persistent P0001-style ID for the session."""

    def __init__(self) -> None:
        self._name_to_id: dict[str, str] = {}
        self._counter = 0

    def id_for(self, name: str) -> str:
        if name not in self._name_to_id:
            self._counter += 1
            self._name_to_id[name] = f"P{self._counter:04d}"
        return self._name_to_id[name]


class FaceTracker:
    """Tracks faces across video frames, maintaining identity continuity."""

    def __init__(self, registry: PersonRegistry, max_misses: int = MAX_MISSES, sim_window: int = SMOOTH_FRAMES) -> None:
        self._registry = registry
        self._max_misses = max_misses
        self._sim_window = sim_window
        self._tracks: dict[int, dict] = {}
        self._next_id = 0
        self._recently_lost: list[dict] = []
        self._frame_i = 0

    def _find_track_match(self, box: list[int], used: set[int]) -> int | None:
        best_id, best_iou = None, TRACK_IOU_THRESH
        for tid, tr in self._tracks.items():
            if tid in used:
                continue
            iou = _iou(box, tr["box"])
            if iou > best_iou:
                best_iou, best_id = iou, tid
        if best_id is not None:
            return best_id

        # Centroid proximity fallback if IoU is lost due to quick movement or head turn
        best_dist = None
        for tid, tr in self._tracks.items():
            if tid in used:
                continue
            if not _centroid_close(box, tr["box"], CENTROID_MATCH_FACTOR):
                continue
            ax, ay = _centroid(box)
            bx, by = _centroid(tr["box"])
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist, best_id = dist, tid
        return best_id

    def _try_relink(self, box: list[int]) -> dict | None:
        """Attempt to re-link a newly spawned track with a recently lost identified person."""
        best_i, best_dist = None, None
        for i, e in enumerate(self._recently_lost):
            if self._frame_i - e["lost_at_frame"] > RELINK_GRACE_FRAMES:
                continue
            if not _centroid_close(box, e["box"], RELINK_MAX_DIST_FACTOR):
                continue
            ax, ay = _centroid(box)
            bx, by = _centroid(e["box"])
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_i, best_dist = i, dist
        if best_i is None:
            return None
        return self._recently_lost.pop(best_i)

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Update tracker with new frame detections.
        Returns list of track items.
        """
        self._frame_i += 1
        used: set[int] = set()

        # Clean expired tracks from memory
        self._recently_lost = [
            e for e in self._recently_lost
            if self._frame_i - e["lost_at_frame"] <= RELINK_GRACE_FRAMES
        ]

        for det in detections:
            box = det["box"]
            best_id = self._find_track_match(box, used)

            if best_id is None:
                relinked = self._try_relink(box)
                best_id = self._next_id
                self._next_id += 1
                self._tracks[best_id] = {
                    "box": box,
                    "name": relinked["name"] if relinked else "Unknown",
                    "person_id": relinked["person_id"] if relinked else None,
                    "sim_hist": deque(maxlen=self._sim_window),
                    "misses": 0,
                    "confident_this_frame": False,
                }

            tr = self._tracks[best_id]
            tr["box"] = box
            tr["misses"] = 0
            used.add(best_id)

            if det["name"] != "Unknown":
                tr["name"] = det["name"]
                tr["person_id"] = self._registry.id_for(det["name"])
                tr["sim_hist"].append(det["sim"])
                tr["confident_this_frame"] = True
            else:
                tr["confident_this_frame"] = False

        # De-duplicate tracks by person_id
        by_person: dict[str, list[int]] = {}
        for tid, tr in self._tracks.items():
            if tr["person_id"] is not None:
                by_person.setdefault(tr["person_id"], []).append(tid)

        for tids in by_person.values():
            if len(tids) <= 1:
                continue

            def _rank(tid: int) -> tuple[bool, float]:
                tr = self._tracks[tid]
                avg_sim = float(np.mean(tr["sim_hist"])) if tr["sim_hist"] else 0.0
                return (tid in used, avg_sim)

            keep = max(tids, key=_rank)
            for tid in tids:
                if tid != keep:
                    del self._tracks[tid]
                    used.discard(tid)

        # Age out tracks not matched in this frame
        for tid, tr in list(self._tracks.items()):
            if tid in used:
                continue
            if tr["person_id"] is None:
                del self._tracks[tid]
                continue
            tr["misses"] += 1
            if tr["misses"] > self._max_misses:
                self._recently_lost.append({
                    "person_id": tr["person_id"],
                    "name": tr["name"],
                    "box": tr["box"],
                    "lost_at_frame": self._frame_i,
                })
                del self._tracks[tid]

        out = []
        for tid, tr in self._tracks.items():
            sim = float(np.mean(tr["sim_hist"])) if tr["sim_hist"] else 0.0
            if tr["person_id"] is None:
                mode = "unknown"
            elif tr["misses"] == 0 and tr.get("confident_this_frame"):
                mode = "identified"
            else:
                mode = "tracking"

            out.append({
                "box": [int(b) for b in tr["box"]],
                "name": str(tr["name"]),
                "person_id": str(tr["person_id"]) if tr["person_id"] else f"V{tid:04d}",
                "sim": float(sim),
                "mode": str(mode),
                "track_id": int(tid),
            })
        return out


class AdaptiveFrameSkip:
    """
    Adapts how many frames are skipped between ArcFace inference calls
    based on actual inference latency, ensuring smooth 25-30 FPS playback.
    """

    def __init__(self, floor: int = 1, ceiling: int = 8, target_ms: float = 100.0) -> None:
        self.n = floor
        self._floor = floor
        self._ceiling = ceiling
        self._target_ms = target_ms
        self.infer_ms = 0.0

    def should_infer(self, frame_i: int) -> bool:
        return frame_i % self.n == 0

    def record(self, elapsed_ms: float) -> None:
        self.infer_ms = elapsed_ms if self.infer_ms == 0 else 0.8 * self.infer_ms + 0.2 * elapsed_ms
        if self.infer_ms > self._target_ms and self.n < self._ceiling:
            self.n += 1
        elif self.infer_ms < self._target_ms * 0.5 and self.n > self._floor:
            self.n -= 1


class BoxSmoother:
    """
    Interpolates bounding boxes between skipped frames so boxes move smoothly
    without jitter when frame skipping is active.
    """

    def __init__(self, max_extrapolate_frames: int = 15) -> None:
        self._tracks: dict[int, dict] = {}
        self._max_extrapolate = max_extrapolate_frames

    def on_result(self, items: list[dict], frame_i: int) -> None:
        updated: dict[int, dict] = {}
        for it in items:
            tid = it.get("track_id")
            if tid is None:
                continue
            box = it["box"]
            prev = self._tracks.get(tid)
            if prev is not None and frame_i > prev["frame_i"]:
                dt = frame_i - prev["frame_i"]
                vel = [(box[i] - prev["box"][i]) / dt for i in range(4)]
            else:
                vel = [0.0, 0.0, 0.0, 0.0]
            updated[tid] = {"box": box, "frame_i": frame_i, "vel": vel}
        self._tracks = updated

    def interpolate(self, items: list[dict], frame_i: int) -> list[dict]:
        out = []
        for it in items:
            tid = it.get("track_id")
            info = self._tracks.get(tid) if tid is not None else None
            if info is None:
                out.append(it)
                continue
            dt = min(frame_i - info["frame_i"], self._max_extrapolate)
            box, vel = info["box"], info["vel"]
            new_box = [int(box[i] + vel[i] * dt) for i in range(4)]
            out.append({**it, "box": new_box})
        return out
