import os
import sys
import time
import tempfile
import cv2
import pandas as pd
import streamlit as st

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.detection_model import DetectionModel, CLASS_NAMES
from core.tracker import Tracker
from core.analytics import AnalyticsEngine
from infrastructure.config_loader import load_config
from infrastructure.video_io import open_video, get_video_fps, create_video_writer

# Page setup
st.set_page_config(page_title="Smart CCTV Analytics", layout="wide")
st.markdown("<style>.stMetric {background-color: #1E1E2E; padding: 15px; border-radius: 10px;}</style>", unsafe_allow_html=True)


def main():
    st.title("Smart CCTV Video Surveillance Dashboard")
    st.caption("Real-time AI Object Detection, Tracking & Event Analytics")

    config = load_config()

    # Sidebar Controls
    st.sidebar.header("Control Panel")
    video_source_type = st.sidebar.radio("Choose Input Source", ["Select Sample Video", "Upload Video File"])

    video_path, temp_file_path = None, None

    if video_source_type == "Select Sample Video":
        video_files = [f for f in os.listdir("videos") if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))] if os.path.exists("videos") else []
        if video_files:
            video_path = os.path.join("videos", st.sidebar.selectbox("Sample Videos", video_files))
        else:
            st.sidebar.warning("No sample videos found in 'videos/' directory.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            video_path = temp_file_path

    # Model & Thresholds
    selected_model = st.sidebar.selectbox("Select YOLO Model", ["yolo11s.pt", "yolo11n.pt", "yolo11m.pt"])
    confidence_val = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, float(config["model"].get("confidence", 0.40)), 0.05)

    config["model"]["weights"] = selected_model
    config["model"]["confidence"] = confidence_val

    start_button = st.sidebar.button("Start Analytics", type="primary", use_container_width=True)

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    metric_people = col1.metric("Total People", "0")
    metric_vehicles = col2.metric("Total Vehicles", "0")
    metric_dwell = col3.metric("Avg Dwell Time", "0s")
    metric_alerts = col4.metric("Security Alerts", "0")

    st.divider()

    # Video & Events Layout
    video_col, log_col = st.columns([2, 1])
    with video_col:
        st.subheader("Live Annotated Video Feed")
        video_placeholder = st.empty()
    with log_col:
        st.subheader("Real-Time Security Events Log")
        log_placeholder = st.empty()

    # Processing Loop
    if start_button:
        if not video_path or not os.path.exists(video_path):
            st.error("Please select or upload a valid video file first!")
            return

        fps = get_video_fps(video_path)
        cap = open_video(video_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detector = DetectionModel(config)
        detector.load()
        tracker = Tracker(CLASS_NAMES)
        analytics = AnalyticsEngine(config=config, fps=fps)

        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", "streamlit_output.mp4")
        writer = create_video_writer(output_path, fps, width, height)

        progress_bar = st.progress(0)
        frame_index = 0
        events_history = []
        last_result = None
        last_annotated = None
        DETECT_EVERY = 5  # Run YOLO every 5 frames only

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Run YOLO detection every N frames to keep video smooth
            if frame_index % DETECT_EVERY == 0:
                last_result = detector.predict(frame)
                tracked_objects = tracker.update(last_result, frame_index)
                events = analytics.analyze(tracked_objects, frame_index)

                for evt in events:
                    events_history.append({
                        "Time (s)": f"{evt.timestamp:.1f}s",
                        "Event Type": evt.event_type,
                        "Track ID": evt.track_id if evt.track_id != -1 else "N/A",
                        "Message": evt.message
                    })

                last_annotated = analytics.draw_zones(last_result.plot())
                writer.write(last_annotated)

                # Update KPIs and log only on detection frames
                summary_stats = analytics.summary()
                metric_people.metric("Total People", str(summary_stats["total_people"]))
                metric_vehicles.metric("Total Vehicles", str(summary_stats["total_vehicles"]))
                metric_dwell.metric("Avg Dwell Time", f"{summary_stats['average_dwell_time_sec']}s")
                metric_alerts.metric("Security Alerts", str(summary_stats["total_alerts"]))

                if events_history:
                    log_placeholder.dataframe(
                        pd.DataFrame(events_history[::-1]),
                        height=350,
                        use_container_width=True
                    )
                else:
                    log_placeholder.info("No security events detected yet.")
            else:
                # Non-detection frame: write raw frame to video but KEEP
                # last_annotated unchanged so displayed boxes don't flicker
                writer.write(analytics.draw_zones(frame.copy()))

            # Always display the last YOLO-annotated frame (no flickering)
            if last_annotated is not None:
                rgb_frame = cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Slow down playback a bit so IDs are readable (1.5x normal speed)
            time.sleep(1 / max(fps, 1))

            frame_index += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_index / total_frames, 1.0))

        cap.release()
        writer.release()

        st.success("Video processing completed successfully!")

        # Summary & Exports
        st.subheader("Analytics Summary & Exports")
        final_summary = analytics.summary()
        col_a, col_b = st.columns(2)

        with col_a:
            st.json({
                "Total People": final_summary["total_people"],
                "Total Vehicles": final_summary["total_vehicles"],
                "Average Dwell Time (sec)": final_summary["average_dwell_time_sec"],
                "Total Alerts": final_summary["total_alerts"],
                "Loitering Alerts": final_summary["loitering_events"],
                "Restricted Zone Entries": final_summary["restricted_zone_events"],
                "Crowding Events": final_summary["crowd_events"],
            })

        with col_b:
            if events_history:
                st.download_button(
                    "Download Security Events (CSV)",
                    data=pd.DataFrame(events_history).to_csv(index=False),
                    file_name="security_events_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            if os.path.exists(output_path):
                with open(output_path, "rb") as vid_file:
                    st.download_button(
                        "Download Processed Video (MP4)",
                        data=vid_file.read(),
                        file_name="annotated_cctv_output.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )

        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    main()