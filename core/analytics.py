"""
analytics.py
Tracks every object across frames and computes:
- People / vehicle counts
- Dwell time per person
- Events: loitering, restricted-zone entry, crowding
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from core.tracker import TrackedObject


@dataclass
class Event:
    """Represents an event detected by the analytics engine."""

    event_type: str
    frame_index: int
    timestamp: float
    track_id: int
    message: str


class AnalyticsEngine:

    def __init__(self, config: dict, fps: float):

        self.fps = fps

        events_config = config["events"]

        self.loitering_seconds = float(events_config["loitering_seconds"])
        self.crowd_threshold = int(events_config["crowd_threshold"])

        self.restricted_zone = np.array(
            events_config["restricted_zone"], dtype=np.int32
        )
        self.safe_waiting_zone = np.array(
            events_config["safe_waiting_zone"], dtype=np.int32
        )

        # ==================================================
        # Tracking history
        # ==================================================

        self.first_seen: Dict[int, int] = {}
        self.last_seen: Dict[int, int] = {}
        self.seen_classes: Dict[int, str] = {}

        self.last_position: Dict[int, Tuple[float, float]] = {}
        self.stationary_since: Dict[int, int] = {}
        self.loitering_triggered = set()
        self.previous_restricted_state: Dict[int, bool] = {}
        self.crowding_active = False

        # All events collected across the whole video, used by summary()
        self.all_events: List[Event] = []

    # ==================================================
    # Helper functions
    # ==================================================

    @staticmethod
    def is_person(obj: TrackedObject) -> bool:
        return obj.class_name.lower() == "person"

    @staticmethod
    def is_vehicle(obj: TrackedObject) -> bool:
        return obj.class_name.lower() in ("car", "truck", "bus", "motorcycle")

    @staticmethod
    def point_inside_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0

    @staticmethod
    def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

    # ==================================================
    # Main analysis
    # ==================================================

    def analyze(self, objects: List[TrackedObject], frame_index: int) -> List[Event]:

        events = []

        # ==================================================
        # Register objects (first_seen, last_seen, class)
        # ==================================================

        for obj in objects:

            if obj.track_id not in self.first_seen:
                self.first_seen[obj.track_id] = frame_index
                self.last_position[obj.track_id] = obj.center
                self.stationary_since[obj.track_id] = frame_index

            # Updated every frame the object appears in
            self.last_seen[obj.track_id] = frame_index
            self.seen_classes[obj.track_id] = obj.class_name

        # ==================================================
        # Loitering
        # ==================================================

        # Maximum movement allowed while considered stationary (pixels/frame)
        movement_threshold = 5.0

        for obj in objects:

            if not self.is_person(obj):
                continue

            # People inside the safe waiting zone are exempt from loitering
            # checks (e.g. a queue). Their stationary timer keeps resetting
            # so it never accumulates while they're there.
            if self.point_inside_polygon(obj.bottom_center, self.safe_waiting_zone):
                self.stationary_since[obj.track_id] = frame_index
                self.last_position[obj.track_id] = obj.center
                continue

            current_position = obj.center
            previous_position = self.last_position.get(obj.track_id, current_position)
            movement = self.distance(current_position, previous_position)

            if movement <= movement_threshold:
                if obj.track_id not in self.stationary_since:
                    self.stationary_since[obj.track_id] = frame_index
            else:
                # Restart stationary timer
                self.stationary_since[obj.track_id] = frame_index

            self.last_position[obj.track_id] = current_position

            stationary_frames = frame_index - self.stationary_since[obj.track_id]
            stationary_seconds = stationary_frames / self.fps

            if (
                stationary_seconds >= self.loitering_seconds
                and obj.track_id not in self.loitering_triggered
            ):
                events.append(
                    Event(
                        event_type="LOITERING",
                        frame_index=frame_index,
                        timestamp=frame_index / self.fps,
                        track_id=obj.track_id,
                        message=(
                            f"Person {obj.track_id} has remained almost "
                            f"stationary for {stationary_seconds:.1f} seconds"
                        ),
                    )
                )
                self.loitering_triggered.add(obj.track_id)

        # ==================================================
        # Crowding
        # ==================================================

        person_count = sum(1 for obj in objects if self.is_person(obj))

        if person_count >= self.crowd_threshold and not self.crowding_active:
            events.append(
                Event(
                    event_type="CROWDING",
                    frame_index=frame_index,
                    timestamp=frame_index / self.fps,
                    track_id=-1,
                    message=f"Crowding detected: {person_count} people",
                )
            )
            self.crowding_active = True

        elif person_count < self.crowd_threshold and self.crowding_active:
            self.crowding_active = False

        # ==================================================
        # Restricted Zone
        # ==================================================

        for obj in objects:

            if not self.is_person(obj):
                continue

            point = obj.bottom_center
            inside_zone = self.point_inside_polygon(point, self.restricted_zone)
            previous_state = self.previous_restricted_state.get(obj.track_id, False)

            if inside_zone and not previous_state:
                events.append(
                    Event(
                        event_type="RESTRICTED_ZONE",
                        frame_index=frame_index,
                        timestamp=frame_index / self.fps,
                        track_id=obj.track_id,
                        message=f"Person {obj.track_id} entered the restricted zone",
                    )
                )

            self.previous_restricted_state[obj.track_id] = inside_zone

        # ==================================================
        # Store + return this frame's events
        # ==================================================

        self.all_events.extend(events)
        return events

    # ==================================================
    # Final report data
    # ==================================================

    def summary(self) -> dict:
        """Called once after the video finishes to produce final report data."""

        people_ids = [tid for tid, cls in self.seen_classes.items() if cls.lower() == "person"]
        vehicle_ids = [
            tid for tid, cls in self.seen_classes.items()
            if cls.lower() in ("car", "truck", "bus", "motorcycle")
        ]

        dwell_times = [
            (self.last_seen[tid] - self.first_seen[tid]) / self.fps
            for tid in people_ids
        ]

        return {
            "total_people": len(people_ids),
            "total_vehicles": len(vehicle_ids),
            "average_dwell_time_sec": round(sum(dwell_times) / len(dwell_times), 1) if dwell_times else 0,
            "total_alerts": len(self.all_events),
            "loitering_events": len([e for e in self.all_events if e.event_type == "LOITERING"]),
            "restricted_zone_events": len([e for e in self.all_events if e.event_type == "RESTRICTED_ZONE"]),
            "crowd_events": len([e for e in self.all_events if e.event_type == "CROWDING"]),
            "events_detail": self.all_events,
        }

    # ==================================================
    # Draw zones
    # ==================================================

    def draw_zones(self, frame):

        cv2.polylines(frame, [self.restricted_zone], True, (0, 0, 255), 2)
        cv2.putText(
            frame, "Restricted Zone", tuple(self.restricted_zone[0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

        cv2.polylines(frame, [self.safe_waiting_zone], True, (0, 255, 0), 2)
        cv2.putText(
            frame, "Safe Waiting Zone", tuple(self.safe_waiting_zone[0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

        return frame