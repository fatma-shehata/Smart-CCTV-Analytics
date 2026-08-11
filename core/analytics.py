"""
analytics.py
The core analytical engine: tracks every object ID over time and
computes counts, dwell time, and security events (loitering,
restricted-zone entry, crowding).
"""

from core.geometry import point_in_polygon, bbox_bottom_center
from infrastructure.logger import setup_logger

logger = setup_logger(__name__)


class AnalyticsEngine:
    def __init__(self, config: dict, fps: float):
        self.events_cfg = config["events"]
        self.fps = fps

        self.first_seen_frame: dict[int, int] = {}
        self.last_seen_frame: dict[int, int] = {}
        self.class_of: dict[int, str] = {}

        self.max_people_seen = 0
        self.alerts: list[dict] = []
        self.restricted_violations: set[int] = set()

    def frame_to_seconds(self, frame_number: int) -> float:
        return frame_number / self.fps

    def update(self, frame_number: int, tracked_objects: list) -> list[dict]:
        """Called once per frame. Returns only the NEW alerts raised on
        this exact frame (not the full history) so app.py can display
        them live."""
        new_alerts = []
        current_people_ids = set()

        for obj in tracked_objects:
            tid = obj.track_id
            self.class_of[tid] = obj.class_name

            if tid not in self.first_seen_frame:
                self.first_seen_frame[tid] = frame_number
            self.last_seen_frame[tid] = frame_number

            if obj.class_name == "Person":
                current_people_ids.add(tid)
                new_alerts += self._check_loitering(tid, obj.box, frame_number)
                new_alerts += self._check_restricted_zone(tid, obj.box, frame_number)

        new_alerts += self._check_crowd(current_people_ids, frame_number)
        self.max_people_seen = max(self.max_people_seen, len(current_people_ids))
        return new_alerts

    def _check_loitering(self, tid: int, box, frame_number: int) -> list[dict]:
           duration = self.frame_to_seconds(frame_number - self.first_seen_frame[tid])
           if duration < self.events_cfg["loitering_seconds"]:
             return []
  
            # Skip the alert if the person is standing in a designated safe
             # waiting zone (e.g. a sidewalk, waiting to cross)
           if "safe_waiting_zone" in self.events_cfg:
                 point = bbox_bottom_center(box)
           if point_in_polygon(point, self.events_cfg["safe_waiting_zone"]):
                  return []

           already_alerted = any(
              a["type"] == "loitering" and a["track_id"] == tid for a in self.alerts
           )
           if already_alerted:
              return []

           alert = {
                  "type": "loitering",
                  "track_id": tid,
                  "frame": frame_number,
                   "time_sec": round(self.frame_to_seconds(frame_number), 1),
                   "message": f"Person {tid} standing for over "
                          f"{self.events_cfg['loitering_seconds']}s - Suspicious Activity",
               }
           self.alerts.append(alert)
           return [alert]

    def _check_restricted_zone(self, tid: int, box, frame_number: int) -> list[dict]:
        point = bbox_bottom_center(box)
        if not point_in_polygon(point, self.events_cfg["restricted_zone"]):
            return []
        if tid in self.restricted_violations:
            return []

        self.restricted_violations.add(tid)
        alert = {
            "type": "restricted_zone",
            "track_id": tid,
            "frame": frame_number,
            "time_sec": round(self.frame_to_seconds(frame_number), 1),
            "message": f"Person {tid} entered Restricted Area - Alert",
        }
        self.alerts.append(alert)
        return [alert]

    def _check_crowd(self, current_people_ids: set, frame_number: int) -> list[dict]:
        if len(current_people_ids) <= self.events_cfg["crowd_threshold"]:
            return []

        alert = {
            "type": "crowd",
            "track_id": None,
            "frame": frame_number,
            "time_sec": round(self.frame_to_seconds(frame_number), 1),
            "message": f"{len(current_people_ids)} people detected - Crowded Area",
        }
        self.alerts.append(alert)
        return [alert]

    def summary(self) -> dict:
        """Called once after the video finishes, to build the final report."""
        people_ids = [tid for tid, cls in self.class_of.items() if cls == "Person"]
        vehicle_ids = [tid for tid, cls in self.class_of.items() if cls in ("Car", "Truck", "Bus")]

        dwell_times = {
            tid: round(self.frame_to_seconds(self.last_seen_frame[tid] - self.first_seen_frame[tid]), 1)
            for tid in people_ids
        }

        return {
            "total_people": len(people_ids),
            "total_vehicles": len(vehicle_ids),
            "max_people_at_once": self.max_people_seen,
            "average_dwell_time_sec": round(sum(dwell_times.values()) / len(dwell_times), 1) if dwell_times else 0,
            "total_alerts": len(self.alerts),
            "restricted_zone_violations": len(self.restricted_violations),
            "loitering_events": len([a for a in self.alerts if a["type"] == "loitering"]),
            "crowd_events": len([a for a in self.alerts if a["type"] == "crowd"]),
            "alerts_detail": self.alerts,
        }