"""
video_io.py
Handles all direct interaction with video files: opening, reading FPS,
and writing the annotated output video.
"""

import cv2


def get_video_fps(video_path: str) -> float:
    """
    Returns the video's real FPS instead of assuming a fixed 30.
    This matters because dwell-time calculations depend directly on it.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    # Some downloaded videos report fps=0 or an unrealistic value
    if not fps or fps <= 1 or fps > 120:
        return 25.0  # safe fallback
    return fps


def open_video(video_path: str) -> cv2.VideoCapture:
    """Opens a video file and raises a clear error if it fails."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    return cap


def create_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Creates a video writer for saving the annotated output."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))