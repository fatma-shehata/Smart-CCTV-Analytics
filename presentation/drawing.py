"""
drawing.py
Draws bounding boxes, track IDs, the restricted zone, and live alerts
onto a video frame. Purely visual — has no effect on analytics logic.
"""

import cv2
import numpy as np


def draw_polygon_zone(frame, polygon: list, color=(0, 0, 255), alpha=0.25):
    """Draws a semi-transparent restricted zone on the frame."""
    overlay = frame.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    return frame


def draw_tracked_objects(frame, tracked_objects):
    """Draws a box and label for each tracked object."""
    for obj in tracked_objects:
        x1, y1, x2, y2 = map(int, obj.box)
        color = (0, 255, 0) if obj.class_name == "Person" else (255, 165, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{obj.class_name} {obj.track_id}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def draw_alerts(frame, alerts: list):
    """Draws active alert messages at the top of the frame."""
    for i, alert in enumerate(alerts):
        cv2.putText(frame, f"ALERT: {alert['message']}", (20, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame