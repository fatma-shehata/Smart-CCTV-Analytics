from dataclasses import dataclass
from typing import List


@dataclass
class TrackedObject:
    """
    Represents one object tracked across video frames.
    """

    track_id: int
    class_id: int
    class_name: str
    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    frame_index: int

    @property
    def center(self):
        """Center of the bounding box."""

        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2

        return cx, cy

    @property
    def bottom_center(self):
        """
        Bottom-center of the bounding box.

        Useful for checking zone membership.
        """

        cx = (self.x1 + self.x2) / 2
        cy = self.y2

        return cx, cy


class Tracker:
    """
    Converts raw YOLO + ByteTrack results
    into TrackedObject instances.
    """

    def __init__(self, class_names: dict):
        self.class_names = class_names

    def update(
        self,
        result,
        frame_index: int
    ) -> List[TrackedObject]:

        tracked_objects = []

        if result is None:
            return tracked_objects

        if result.boxes is None:
            return tracked_objects

        boxes = result.boxes

        # No tracking IDs available
        if boxes.id is None:
            return tracked_objects

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        for box, track_id, class_id, confidence in zip(
            xyxy,
            track_ids,
            class_ids,
            confidences
        ):

            class_id = int(class_id)

            class_name = self.class_names.get(
                class_id,
                f"Class_{class_id}"
            )

            x1, y1, x2, y2 = map(float, box)

            tracked_object = TrackedObject(
                track_id=int(track_id),
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidence),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                frame_index=frame_index
            )

            tracked_objects.append(tracked_object)

        return tracked_objects