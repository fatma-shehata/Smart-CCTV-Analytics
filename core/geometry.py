"""
geometry.py
Pure geometric helpers used by analytics.py to determine object
positions relative to zones (e.g. the restricted area polygon).
"""

import cv2
import numpy as np


def bbox_bottom_center(box) -> tuple:
    """Returns the bottom-center point of a bounding box — i.e. where
    the person's feet touch the ground. Used instead of the box's full
    center because it better represents the object's actual position
    on the floor plane."""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int(y2))


def point_in_polygon(point: tuple, polygon: list) -> bool:
    """Checks whether a point lies inside a polygon (e.g. the
    restricted zone) using OpenCV's point-in-polygon test."""
    contour = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, point, False)
    return result >= 0