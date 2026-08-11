"""
app.py
Main Streamlit application. Wires together config, the detector,
tracker, analytics engine, and report generator into one pipeline,
and exposes it through a web UI.

Run with:  streamlit run presentation/app.py
"""

import os
import sys
import tempfile
import time
import subprocess
import cv2
import streamlit as st

# Allow running this file directly while still importing from the
# project root (core/, infrastructure/, report.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.config_loader import load_config
from infrastructure.logger import setup_logger
from infrastructure.video_io import get_video_fps, open_video, create_video_writer
from core.detection_model import DetectionModel, CLASS_NAMES
from core.tracker import extract_tracks
from core.analytics import AnalyticsEngine
from presentation.drawing import draw_polygon_zone, draw_tracked_objects, draw_alerts
from report import generate_report

logger = setup_logger(__name__)


@st.cache_resource
def get_detector(config: dict) -> DetectionModel:
    """Loads the model once and reuses it across Streamlit reruns,
    instead of reloading it on every user interaction."""
    detector = DetectionModel(config)
    detector.load()
    return detector


def process_video(video_path: str, config: dict, progress_bar) -> tuple[str, dict, float, float]:
    """Runs the full pipeline on a video file. Returns the output video
    path, the analytics summary, video duration, and processing FPS."""
    detector = get_detector(config)
    video_fps = get_video_fps(video_path)
    analytics = AnalyticsEngine(config, fps=video_fps)

    cap = open_video(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # The writer always outputs at the ORIGINAL resolution. Resizing
    # happens only on a copy of the frame used for detection, so video
    # quality is preserved while detection still runs faster on CPU.
    output_path = os.path.join("outputs", "processed_video.mp4")
    writer = create_video_writer(output_path, video_fps, width, height)

    frame_number = 0
    start_time = time.time()
    resize_width = config["performance"]["frame_resize_width"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # Resize a COPY of the frame for detection only (keeps the
        # written output at the original resolution)
        scale = resize_width / frame.shape[1]
        if scale < 1:
            detection_frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
        else:
            detection_frame = frame
            scale = 1.0

        result = detector.predict(detection_frame)
        tracked_objects = extract_tracks(result, CLASS_NAMES)

        # Scale bounding boxes back up to the original frame size
        # before drawing, since detection ran on a smaller copy.
        if scale < 1:
            for obj in tracked_objects:
                x1, y1, x2, y2 = obj.box
                obj.box = (x1 / scale, y1 / scale, x2 / scale, y2 / scale)

        new_alerts = analytics.update(frame_number, tracked_objects)

        frame = draw_polygon_zone(frame, config["events"]["restricted_zone"])
        frame = draw_tracked_objects(frame, tracked_objects)
        frame = draw_alerts(frame, new_alerts)
        writer.write(frame)

        frame_number += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_number / total_frames, 1.0))

    cap.release()
    writer.release()

    # Re-encode to H.264 so browsers can actually play the video.
    # OpenCV's VideoWriter output is often not browser-compatible even
    # when the file extension is .mp4.
    final_output_path = os.path.join("outputs", "processed_video_web.mp4")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", output_path,
            "-vcodec", "libx264", "-acodec", "aac",
            final_output_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"ffmpeg failed:\n{result.stderr}")
        raise RuntimeError(f"Video re-encoding failed. ffmpeg output:\n{result.stderr}")

    output_path = final_output_path

    processing_duration = time.time() - start_time
    processing_fps = frame_number / processing_duration if processing_duration > 0 else 0
    video_duration_sec = frame_number / video_fps

    summary = analytics.summary()
    return output_path, summary, video_duration_sec, processing_fps


def main():
    st.set_page_config(page_title="Smart CCTV Analytics", layout="wide")
    st.title("Smart CCTV Video Analytics for Real-Time Security Monitoring")
    st.write("Upload a surveillance video to run detection, tracking, and event analysis.")

    config = load_config("config.yaml")

    uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Streamlit gives us an in-memory file; OpenCV needs a real path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            temp_video_path = tmp.name

        if st.button("Run Analysis", type="primary"):
            progress_bar = st.progress(0)
            with st.spinner("Processing video..."):
                output_path, summary, duration, proc_fps = process_video(
                    temp_video_path, config, progress_bar
                )

            json_path, md_path = generate_report(summary, duration, proc_fps)

            st.success("Analysis complete.")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Processed Video")
                st.video(output_path)

            with col2:
                st.subheader("Summary")
                st.metric("People", summary["total_people"])
                st.metric("Vehicles", summary["total_vehicles"])
                st.metric("Max People At Once", summary["max_people_at_once"])
                st.metric("Total Alerts", summary["total_alerts"])

            with open(md_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())

            with open(json_path, "rb") as f:
                st.download_button("Download JSON report", f, file_name="report.json")

            os.remove(temp_video_path)


if __name__ == "__main__":
    main()