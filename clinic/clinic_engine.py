"""
Clinic Engine for Portable Clinic.
State Machine & Analytics Engine for:
- Doctor Presence & Attendance (Entry timestamp, Exit timestamp, Shift duration).
- Grace-period debounce filter (prevents false exits and duplicate session entries).
- Visitor tracking & clinic occupancy calculation.
- Unattended visit detection (people entering while doctor is absent).
- Absence duration tracking.
"""

import time
from datetime import datetime
from typing import Optional
from .config import DOCTOR_NAMES, DOCTOR_EXIT_GRACE_SEC, VISITOR_EXIT_GRACE_SEC
from .database import ClinicDatabase


class ClinicEngine:
    """State machine and analytics engine for the Portable Clinic."""

    def __init__(self, db: ClinicDatabase):
        self.db = db
        self.db.compact_overlapping_sessions()
        self.db.recover_open_sessions()
        self.doctor_names: list[str] = [d.lower() for d in DOCTOR_NAMES]
        self.doctor_exit_grace_sec: float = DOCTOR_EXIT_GRACE_SEC
        self.visitor_exit_grace_sec: float = VISITOR_EXIT_GRACE_SEC

        self._load_settings()

        # Doctor State
        self.doctor_present: bool = False
        self.current_doctor_name: Optional[str] = None
        self.doctor_entry_time: Optional[datetime] = None
        self.doctor_entry_unix: Optional[float] = None
        self.doctor_entry_mono: Optional[float] = None
        self.doctor_last_seen_dt: Optional[datetime] = None
        self.doctor_last_seen_mono: float = 0.0
        self.doctor_unseen_started_dt: Optional[datetime] = None
        self.doctor_unseen_started_mono: Optional[float] = None
        self.doctor_unseen_total_sec: float = 0.0
        self.current_doctor_session_id: Optional[int] = None
        self.last_known_doctor_name: Optional[str] = None

        # Doctor Anti-Jitter Entry Filter (requires consecutive frames to confirm entry)
        self._doc_candidate_name: Optional[str] = None
        self._doc_candidate_frames: int = 0
        self._doc_candidate_first_seen_dt: Optional[datetime] = None

        # Doctor Absence Tracking
        self.absence_start_dt: Optional[datetime] = None
        self.absence_start_mono: Optional[float] = None

        # Active Visitors: track_id -> dict
        self.active_visitors: dict[int, dict] = {}
        # Candidate Visitors (anti-jitter)
        self._visitor_candidates: dict[int, dict] = {}

    def _load_settings(self) -> None:
        """Load settings from DB if configured."""
        saved_docs = self.db.get_setting("doctor_names")
        if saved_docs:
            self.doctor_names = [n.strip().lower()
                                 for n in saved_docs.split(",") if n.strip()]

        saved_doc_grace = self.db.get_setting("doctor_exit_grace_sec")
        if saved_doc_grace:
            try:
                self.doctor_exit_grace_sec = float(saved_doc_grace)
            except ValueError:
                pass

        saved_vis_grace = self.db.get_setting("visitor_exit_grace_sec")
        if saved_vis_grace:
            try:
                self.visitor_exit_grace_sec = float(saved_vis_grace)
            except ValueError:
                pass

    def set_doctor_names(self, names: list[str]) -> None:
        """Configure registered doctor names."""
        self.doctor_names = [n.strip().lower() for n in names if n.strip()]
        self.db.set_setting("doctor_names", ",".join(self.doctor_names))
        print(
            f"[ClinicEngine] Registered Doctors updated: {self.doctor_names}")

    def set_grace_periods(self, doc_grace: float, vis_grace: float) -> None:
        """Configure exit grace period thresholds in seconds."""
        self.doctor_exit_grace_sec = max(3.0, float(doc_grace))
        self.visitor_exit_grace_sec = max(2.0, float(vis_grace))
        self.db.set_setting("doctor_exit_grace_sec",
                            str(self.doctor_exit_grace_sec))
        self.db.set_setting("visitor_exit_grace_sec",
                            str(self.visitor_exit_grace_sec))
        print(
            f"[ClinicEngine] Grace periods updated: Doctor={self.doctor_exit_grace_sec}s, Visitor={self.visitor_exit_grace_sec}s")

    def is_doctor(self, name: str) -> bool:
        """Check if person name matches any registered doctor."""
        if not name or name.lower() == "unknown":
            return False
        return name.strip().lower() in self.doctor_names

    def process_frame(self, tracked_items: list[dict]) -> dict:
        """
        Process tracker output for a single frame.
        Applies hysteresis filters, logs entries/exits, updates session DB, and returns live status.
        """
        now_dt = datetime.now()
        now_mono = time.monotonic()

        # 1. Classify detections in this frame
        frame_doctors = []
        frame_visitors = []

        for item in tracked_items:
            name = item.get("name", "Unknown")
            if self.is_doctor(name):
                frame_doctors.append(item)
            else:
                frame_visitors.append(item)

        # -------------------------------------------------------------------
        # 2. Doctor Presence / Attendance State Machine
        # -------------------------------------------------------------------
        if frame_doctors:
            doc_item = frame_doctors[0]
            doc_name = doc_item["name"]
            self.last_known_doctor_name = doc_name

            if not self.doctor_present:
                # Anti-jitter check: require 2 consecutive frames to confirm entry
                if self._doc_candidate_name == doc_name:
                    self._doc_candidate_frames += 1
                else:
                    self._doc_candidate_name = doc_name
                    self._doc_candidate_frames = 1
                    self._doc_candidate_first_seen_dt = now_dt

                if self._doc_candidate_frames >= 2:
                    # Doctor entry confirmed!
                    self.doctor_present = True
                    self.current_doctor_name = doc_name
                    entry_time = self._doc_candidate_first_seen_dt or now_dt
                    self.doctor_entry_time = entry_time
                    self.doctor_entry_unix = entry_time.timestamp()
                    self.doctor_entry_mono = now_mono
                    self.doctor_unseen_started_dt = None
                    self.doctor_unseen_started_mono = None
                    self.doctor_unseen_total_sec = 0.0
                    self.current_doctor_session_id = self.db.start_doctor_session(
                        doc_name, entry_time)

                    absence_detail = ""
                    if self.absence_start_mono is not None:
                        absence_dur = now_mono - self.absence_start_mono
                        absence_detail = f" (Returned after {self.db.format_seconds(absence_dur)} absence)"
                        self.absence_start_mono = None
                        self.absence_start_dt = None

                    self.db.log_event(
                        event_type="DOCTOR_ENTRY",
                        person_name=doc_name,
                        person_id=doc_item.get("person_id", "DOC"),
                        details=f"Dr. {doc_name.capitalize()} entered clinic container{absence_detail}",
                    )
            else:
                # Doctor was already confirmed present, reset candidate counter
                self._doc_candidate_frames = 0
                self._doc_candidate_name = None
                if self.doctor_unseen_started_mono is not None:
                    self.doctor_unseen_total_sec += now_mono - self.doctor_unseen_started_mono
                    self.doctor_unseen_started_mono = None
                    self.doctor_unseen_started_dt = None

            # Continuously update last seen timestamp
            self.doctor_last_seen_dt = now_dt
            self.doctor_last_seen_mono = now_mono
        else:
            # Doctor not detected in this frame
            self._doc_candidate_frames = 0
            self._doc_candidate_name = None

            if self.doctor_present:
                if self.doctor_unseen_started_mono is None:
                    self.doctor_unseen_started_mono = now_mono
                    self.doctor_unseen_started_dt = now_dt
                # Check if doctor absence has exceeded the grace threshold
                time_since_seen = now_mono - self.doctor_unseen_started_mono
                if time_since_seen > self.doctor_exit_grace_sec:
                    # Doctor has exited!
                    self.doctor_present = False
                    exit_dt = self.doctor_unseen_started_dt or self.doctor_last_seen_dt or now_dt
                    duration_sec = max(
                        0.0,
                        exit_dt.timestamp() - (self.doctor_entry_unix or exit_dt.timestamp()) -
                        self.doctor_unseen_total_sec,
                    )

                    if self.current_doctor_session_id:
                        self.db.close_doctor_session(
                            self.current_doctor_session_id, exit_dt, duration_sec)

                    doc_name = self.current_doctor_name or self.last_known_doctor_name or "Doctor"

                    self.db.log_event(
                        event_type="DOCTOR_EXIT",
                        person_name=doc_name,
                        person_id="DOC",
                        details=f"Dr. {doc_name.capitalize()} exited clinic container (Shift duration: {self.db.format_seconds(duration_sec)})",
                    )

                    # Start absence tracking from exact moment doctor left
                    self.absence_start_mono = self.doctor_last_seen_mono
                    self.absence_start_dt = exit_dt
                    self.current_doctor_session_id = None
                    self.current_doctor_name = None
                    self.doctor_entry_time = None
                    self.doctor_entry_unix = None
                    self.doctor_entry_mono = None
                    self.doctor_unseen_started_dt = None
                    self.doctor_unseen_started_mono = None
                    self.doctor_unseen_total_sec = 0.0

        # -------------------------------------------------------------------
        # 3. Visitor Tracking & Unattended Visitors State Machine
        # -------------------------------------------------------------------
        current_frame_tids = set()

        for vis in frame_visitors:
            tid = vis["track_id"]
            pid = vis.get("person_id", f"V{tid:04d}")
            pname = vis.get("name", "Unknown")
            current_frame_tids.add(tid)

            if tid in self.active_visitors:
                # Existing active visitor, update last seen
                self.active_visitors[tid]["last_seen_mono"] = now_mono
                self.active_visitors[tid]["last_seen_dt"] = now_dt
                if pname != "Unknown":
                    self.active_visitors[tid]["person_name"] = pname
            else:
                # A known visitor may receive a new tracker ID after a brief occlusion.
                # Reuse the existing session instead of opening a duplicate visit.
                previous_tid = next(
                    (
                        active_tid
                        for active_tid, active in self.active_visitors.items()
                        if pid != f"V{tid:04d}"
                        and active["person_id"] == pid
                    ),
                    None,
                )
                if previous_tid is not None:
                    visitor = self.active_visitors.pop(previous_tid)
                    visitor["track_id"] = tid
                    visitor["last_seen_mono"] = now_mono
                    visitor["last_seen_dt"] = now_dt
                    if pname != "Unknown":
                        visitor["person_name"] = pname
                    self.active_visitors[tid] = visitor
                    continue

                # Candidate visitor anti-jitter
                if tid not in self._visitor_candidates:
                    self._visitor_candidates[tid] = {
                        "frames": 1,
                        "first_dt": now_dt,
                        "person_id": pid,
                        "person_name": pname,
                    }
                else:
                    self._visitor_candidates[tid]["frames"] += 1
                    if pname != "Unknown":
                        self._visitor_candidates[tid]["person_name"] = pname

                if self._visitor_candidates[tid]["frames"] >= 2:
                    # Visitor entry confirmed!
                    cand = self._visitor_candidates.pop(tid)
                    was_unattended = not self.doctor_present
                    entry_dt = cand["first_dt"]

                    session_id = self.db.start_visitor_session(
                        person_id=cand["person_id"],
                        person_name=cand["person_name"],
                        was_unattended=was_unattended,
                        start_time=entry_dt,
                    )

                    self.active_visitors[tid] = {
                        "track_id": tid,
                        "person_id": cand["person_id"],
                        "person_name": cand["person_name"],
                        "entry_time": entry_dt.strftime("%H:%M:%S"),
                        "entry_unix": entry_dt.timestamp(),
                        "last_seen_mono": now_mono,
                        "last_seen_dt": now_dt,
                        "session_id": session_id,
                        "was_unattended": was_unattended,
                    }

                    disp_name = cand["person_name"] if cand["person_name"] != "Unknown" else cand["person_id"]
                    self.db.log_event(
                        event_type="VISITOR_ENTRY",
                        person_name=disp_name,
                        person_id=cand["person_id"],
                        details=f"Visitor {disp_name} entered clinic" + (
                            " ⚠️ [UNATTENDED — Doctor is Absent]" if was_unattended else ""),
                    )

                    if was_unattended:
                        self.db.log_event(
                            event_type="UNATTENDED_VISITOR",
                            person_name=disp_name,
                            person_id=cand["person_id"],
                            details=f"⚠️ SECURITY ALERT: Visitor {cand['person_id']} entered while Doctor was absent",
                        )

        # Check for expired visitors
        expired_tids = []
        for tid, vis in list(self.active_visitors.items()):
            if tid not in current_frame_tids:
                if (now_mono - vis["last_seen_mono"]) > self.visitor_exit_grace_sec:
                    expired_tids.append(tid)

        for tid in expired_tids:
            vis = self.active_visitors.pop(tid)
            exit_dt = vis.get("last_seen_dt", now_dt)
            self.db.close_visitor_session(vis["session_id"], exit_dt)
            disp_name = vis["person_name"] if vis["person_name"] != "Unknown" else vis["person_id"]
            dwell_sec = max(0.0, exit_dt.timestamp() - vis["entry_unix"])

            self.db.log_event(
                event_type="VISITOR_EXIT",
                person_name=disp_name,
                person_id=vis["person_id"],
                details=f"Visitor {disp_name} exited clinic (Dwell time: {self.db.format_seconds(dwell_sec)})",
            )

        # -------------------------------------------------------------------
        # 4. Generate Live Status Snapshot
        # -------------------------------------------------------------------
        doctor_current_duration_sec = 0.0
        if self.doctor_present and self.doctor_entry_mono is not None:
            current_unseen_sec = 0.0
            if self.doctor_unseen_started_mono is not None:
                current_unseen_sec = now_mono - self.doctor_unseen_started_mono
            doctor_current_duration_sec = max(
                0.0,
                now_mono - self.doctor_entry_mono -
                self.doctor_unseen_total_sec - current_unseen_sec,
            )

        doctor_absence_duration_sec = 0.0
        if not self.doctor_present and self.absence_start_mono is not None:
            doctor_absence_duration_sec = max(
                0.0, now_mono - self.absence_start_mono)

        # Count each recognized person once even if tracker jitter briefly leaves
        # duplicate track IDs active for the same person.
        unique_visitors = {}
        for visitor in self.active_visitors.values():
            person_id = visitor["person_id"]
            current = unique_visitors.get(person_id)
            if current is None or visitor["last_seen_mono"] > current["last_seen_mono"]:
                unique_visitors[person_id] = visitor

        visitor_count = len(unique_visitors)
        occupancy_count = (1 if self.doctor_present else 0) + visitor_count
        unattended_active_count = visitor_count if not self.doctor_present else 0
        daily_stats = self.db.get_daily_summary()
        live_total_doc_sec = daily_stats["total_doctor_presence_sec"] + \
            doctor_current_duration_sec

        return {
            "doctor": {
                "present": self.doctor_present,
                "name": self.current_doctor_name or (self.last_known_doctor_name if self.last_known_doctor_name else "Doctor"),
                "status": "PRESENT" if self.doctor_present else "ABSENT",
                "entry_time": self.doctor_entry_time.strftime("%H:%M:%S") if self.doctor_entry_time else None,
                "last_seen_time": self.doctor_last_seen_dt.strftime("%H:%M:%S") if self.doctor_last_seen_dt else None,
                "current_duration_sec": doctor_current_duration_sec,
                "current_duration_formatted": self.db.format_seconds(doctor_current_duration_sec) if self.doctor_present else "0s",
                "absence_duration_sec": doctor_absence_duration_sec,
                "absence_duration_formatted": self.db.format_seconds(doctor_absence_duration_sec) if not self.doctor_present else "0s",
                "registered_doctors": self.doctor_names,
                "exit_grace_sec": self.doctor_exit_grace_sec,
            },
            "clinic": {
                "occupancy": occupancy_count,
                "active_visitors_count": visitor_count,
                "unattended_active_count": unattended_active_count,
                "active_visitors": [
                    {
                        "track_id": v["track_id"],
                        "person_id": v["person_id"],
                        "person_name": v["person_name"],
                        "entry_time": v["entry_time"],
                        "duration_formatted": self.db.format_seconds(time.time() - v["entry_unix"]),
                        "was_unattended": v["was_unattended"],
                    }
                    for v in unique_visitors.values()
                ],
                "visitor_exit_grace_sec": self.visitor_exit_grace_sec,
            },
            "summary_today": {
                "date": daily_stats["date"],
                "total_doctor_time_sec": live_total_doc_sec,
                "total_doctor_time_formatted": self.db.format_seconds(live_total_doc_sec),
                "doctor_sessions_count": daily_stats["doctor_sessions_count"] + (1 if self.doctor_present else 0),
                "total_visitors": daily_stats["total_visitors"],
                "unattended_visitors": daily_stats["unattended_visitors_count"],
            },
            "recent_events": self.db.get_recent_events(limit=20),
        }
