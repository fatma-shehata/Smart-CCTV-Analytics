"""
tracker.py
Converts the raw result object returned by a detector's predict() into
a clean list of TrackedObject instances that the rest of the project
(analytics.py, app.py) can work with — without knowing anything about
the underlying YOLO/Ultralytics result format.
"""

from dataclasses import dataclass

from infrastructure.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    class_name: str
    box: tuple  # (x1, y1, x2, y2)
    confidence: float


def extract_tracks(result, class_names: dict) -> list[TrackedObject]:
    """
    Safely returns an empty list if the frame has no detections yet
    (e.g. the very first frame, before ByteTrack assigns any IDs).
    """
    tracked = []

    if result.boxes is None or result.boxes.id is None:
        return tracked

    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    for box, track_id, cls_id, conf in zip(boxes, ids, classes, confidences):
        tracked.append(
            TrackedObject(
                track_id=int(track_id),
                class_id=int(cls_id),
                class_name=class_names.get(int(cls_id), f"class_{cls_id}"),
                box=tuple(box),
                confidence=float(conf),
            )
        )
    return tracked