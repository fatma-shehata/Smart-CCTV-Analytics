import os
import sys
import tempfile
from datetime import datetime

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

# --------------------------------------------------------------------------
# Cached, expensive resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading detection model…")
def get_detector(weights: str, confidence: float):
    """Load the YOLO model once per (weights, confidence) combo and reuse it
    across reruns instead of reloading it from disk every single time."""
    cfg = load_config()
    cfg["model"]["weights"] = weights
    cfg["model"]["confidence"] = confidence
    model = DetectionModel(cfg)
    model.load()
    return model

# --------------------------------------------------------------------------
# Page configuration & theming
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel | Smart CCTV Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* ---- Global ---- */
    .stApp {
        background: #FAFBFC;
    }
    html, body, [class*="css"] {
        color: #2B2F38;
    }
    #MainMenu, footer {visibility: hidden;}

    /* ---- Header ---- */
    .sentinel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.1rem 1.6rem;
        margin-bottom: 1.2rem;
        border-radius: 16px;
        background: #FFFFFF;
        border: 1px solid #EAECF0;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
    }
    .sentinel-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #1F2430;
        margin: 0;
        letter-spacing: 0.2px;
    }
    .sentinel-subtitle {
        font-size: 0.88rem;
        color: #8A93A3;
        margin-top: 2px;
    }
    .status-pill {
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .status-idle { background: #F1F3F6; color: #7C8494; border: 1px solid #E2E5EA;}
    .status-live { background: #E7F8EF; color: #17A566; border: 1px solid #C7EFDA;}
    .status-done { background: #EAF2FE; color: #3373E0; border: 1px solid #D2E3FC;}

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #EAECF0;
        padding: 18px 20px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.03);
    }
    div[data-testid="stMetricLabel"] {
        color: #8A93A3 !important;
        font-size: 0.82rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    div[data-testid="stMetricValue"] {
        color: #1F2430 !important;
        font-weight: 700;
    }

    /* ---- Section headers ---- */
    .section-label {
        font-size: 0.95rem;
        font-weight: 700;
        color: #3A4150;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.4rem;
        border-left: 3px solid #6FA8FF;
        padding-left: 10px;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #F5F7FA;
        border-right: 1px solid #EAECF0;
    }

    /* ---- Containers / dataframes / expanders ---- */
    div[data-testid="stDataFrame"], div[data-testid="stExpander"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #EAECF0;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    .stButton > button[kind="primary"] {
        background: #6FA8FF;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background: #5C97F5;
    }

    /* ---- Divider ---- */
    hr {
        border-color: #EAECF0 !important;
    }
</style>
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def render_header(status: str = "idle"):
    """Render the branded header with a live status pill."""
    status_map = {
        "idle": ("status-idle", "● IDLE"),
        "live": ("status-live", "● PROCESSING"),
        "done": ("status-done", "● COMPLETE"),
    }
    css_class, label = status_map.get(status, status_map["idle"])
    st.markdown(
        f"""
        <div class="sentinel-header">
            <div>
                <p class="sentinel-title">🛡️ Sentinel — Smart CCTV Analytics</p>
                <p class="sentinel-subtitle">Real-time AI object detection, tracking &amp; security event analytics</p>
            </div>
            <div>
                <span class="status-pill {css_class}">{label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_video_source():
    """Render sidebar input controls and return (video_path, temp_file_path)."""
    st.sidebar.markdown('<p class="section-label">Video Source</p>', unsafe_allow_html=True)
    source_type = st.sidebar.radio(
        "Input source", ["Select Sample Video", "Upload Video File"], label_visibility="collapsed"
    )

    video_path, temp_file_path = None, None

    if source_type == "Select Sample Video":
        video_dir = "videos"
        video_files = (
            [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
            if os.path.exists(video_dir)
            else []
        )
        if video_files:
            selected = st.sidebar.selectbox("Sample library", sorted(video_files))
            video_path = os.path.join(video_dir, selected)
        else:
            st.sidebar.warning("No sample videos found in the `videos/` directory.")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload footage", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed"
        )
        if uploaded_file:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(uploaded_file.read())
            temp_file.close()
            temp_file_path = temp_file.name
            video_path = temp_file_path
            st.sidebar.success(f"Loaded: {uploaded_file.name}")

    return video_path, temp_file_path


def resolve_model_settings(config: dict) -> dict:
    """Render sidebar model controls and mutate config in place."""
    st.sidebar.markdown('<p class="section-label">Detection Settings</p>', unsafe_allow_html=True)

    selected_model = st.sidebar.selectbox(
        "YOLO model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
        index=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"].index(
            config["model"].get("weights", "yolo11s.pt")
        ) if config["model"].get("weights", "yolo11s.pt") in ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"] else 1,
        help="Nano is fastest, Small is balanced, Medium is most accurate.",
    )
    confidence_val = st.sidebar.slider(
        "Confidence threshold",
        0.10, 1.00, float(config["model"].get("confidence", 0.40)), 0.05,
        help="Higher values reduce false positives but may miss faint detections.",
    )
    detect_every = st.sidebar.slider(
        "Run detection every N frames",
        1, 15, 6,
        help="Higher values improve playback speed at the cost of detection freshness.",
    )

    st.sidebar.markdown('<p class="section-label">Performance</p>', unsafe_allow_html=True)
    display_every = st.sidebar.slider(
        "Refresh preview every N frames",
        1, 10, 3,
        help="Rendering an image in Streamlit is expensive — updating it less often speeds up the whole run.",
    )
    display_scale = st.sidebar.select_slider(
        "Preview resolution",
        options=["25%", "50%", "75%", "100%"],
        value="50%",
        help="Downscaling the preview (not the saved video) cuts rendering cost significantly.",
    )
    save_video = st.sidebar.checkbox(
        "Save annotated output video", value=True,
        help="Turn off if you only need the live preview and event log — disk writes cost time.",
    )

    config["model"]["weights"] = selected_model
    config["model"]["confidence"] = confidence_val
    return {
        "detect_every": detect_every,
        "display_every": display_every,
        "display_scale": int(display_scale.strip("%")) / 100,
        "save_video": save_video,
    }


def render_kpi_row(container):
    """Render (and return handles to) the top KPI metric cards."""
    c1, c2, c3, c4 = container.columns(4)
    return {
        "people": c1.metric("Total People", "0"),
        "vehicles": c2.metric("Total Vehicles", "0"),
        "dwell": c3.metric("Avg Dwell Time", "0s"),
        "alerts": c4.metric("Security Alerts", "0"),
    }


def update_kpi_row(placeholders, stats: dict):
    placeholders["people"].metric("Total People", str(stats["total_people"]))
    placeholders["vehicles"].metric("Total Vehicles", str(stats["total_vehicles"]))
    placeholders["dwell"].metric("Avg Dwell Time", f"{stats['average_dwell_time_sec']}s")
    placeholders["alerts"].metric("Security Alerts", str(stats["total_alerts"]))


def render_event_log(placeholder, events_history: list):
    if events_history:
        placeholder.dataframe(
            pd.DataFrame(events_history[::-1]),
            height=380,
            use_container_width=True,
            hide_index=True,
        )
    else:
        placeholder.info("No security events detected yet.")


# --------------------------------------------------------------------------
# Main application
# --------------------------------------------------------------------------
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "run_status" not in st.session_state:
        st.session_state.run_status = "idle"

    header_placeholder = st.empty()
    with header_placeholder.container():
        render_header(st.session_state.run_status)

    config = load_config()

    video_path, temp_file_path = resolve_video_source()
    detect_opts = resolve_model_settings(config)

    st.sidebar.markdown("---")
    start_button = st.sidebar.button("▶  Start Analytics", type="primary", use_container_width=True)
    st.sidebar.caption(f"Session started · {datetime.now().strftime('%H:%M:%S')}")

    st.markdown('<p class="section-label">Overview</p>', unsafe_allow_html=True)
    kpi_container = st.container()
    kpi_placeholders = render_kpi_row(kpi_container)

    st.divider()

    video_col, log_col = st.columns([2, 1])
    with video_col:
        st.markdown('<p class="section-label">Live Annotated Feed</p>', unsafe_allow_html=True)
        video_placeholder = st.empty()
        video_placeholder.info("Feed will appear here once analytics starts.")
    with log_col:
        st.markdown('<p class="section-label">Security Event Log</p>', unsafe_allow_html=True)
        log_placeholder = st.empty()
        log_placeholder.info("No security events detected yet.")

    if not start_button:
        return

    if not video_path or not os.path.exists(video_path):
        st.error("Please select or upload a valid video file first.")
        return

    st.session_state.run_status = "live"
    with header_placeholder.container():
        render_header("live")

    try:
        fps = get_video_fps(video_path)
        cap = open_video(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detector = get_detector(config["model"]["weights"], config["model"]["confidence"])
        tracker = Tracker(CLASS_NAMES)
        analytics = AnalyticsEngine(config=config, fps=fps)

        detect_every = detect_opts["detect_every"]
        display_every = detect_opts["display_every"]
        display_scale = detect_opts["display_scale"]
        save_video = detect_opts["save_video"]

        writer = None
        output_path = None
        if save_video:
            os.makedirs("outputs", exist_ok=True)
            output_path = os.path.join("outputs", "streamlit_output.mp4")
            writer = create_video_writer(output_path, fps, width, height)

        progress_bar = st.progress(0, text="Initializing…")
        frame_index = 0
        events_history = []
        last_annotated = None
        last_kpi_update = -1
        KPI_UPDATE_EVERY = max(detect_every * 3, 1)  # metrics don't need per-detection refresh

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            is_detect_frame = frame_index % detect_every == 0

            if is_detect_frame:
                result = detector.predict(frame)
                tracked_objects = tracker.update(result, frame_index)
                events = analytics.analyze(tracked_objects, frame_index)

                for evt in events:
                    events_history.append({
                        "Time (s)": f"{evt.timestamp:.1f}s",
                        "Event Type": evt.event_type,
                        "Track ID": evt.track_id if evt.track_id != -1 else "N/A",
                        "Message": evt.message,
                    })
                    render_event_log(log_placeholder, events_history)  # only touches UI on new events

                last_annotated = analytics.draw_zones(result.plot())

                if frame_index - last_kpi_update >= KPI_UPDATE_EVERY:
                    update_kpi_row(kpi_placeholders, analytics.summary())
                    last_kpi_update = frame_index

            if writer is not None:
                writer.write(last_annotated if is_detect_frame else analytics.draw_zones(frame.copy()))

            # Only push a new image to the browser every `display_every` frames —
            # this is the single biggest cost in the loop.
            if last_annotated is not None and frame_index % display_every == 0:
                display_frame = last_annotated
                if display_scale != 1.0:
                    display_frame = cv2.resize(
                        display_frame, None, fx=display_scale, fy=display_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            frame_index += 1
            if total_frames > 0 and frame_index % 5 == 0:  # throttle progress-bar UI updates too
                pct = min(frame_index / total_frames, 1.0)
                progress_bar.progress(pct, text=f"Processing frame {frame_index}/{total_frames}")

        if total_frames > 0:
            progress_bar.progress(1.0, text=f"Processing frame {total_frames}/{total_frames}")

        cap.release()
        if writer is not None:
            writer.release()

        st.session_state.run_status = "done"
        with header_placeholder.container():
            render_header("done")

        st.success("✅ Video processing completed successfully.")

        # ------------------------------------------------------------
        # Summary & exports
        # ------------------------------------------------------------
        st.markdown('<p class="section-label">Analytics Summary &amp; Exports</p>', unsafe_allow_html=True)
        final_summary = analytics.summary()

        s1, s2, s3 = st.columns(3)
        s1.metric("Loitering Alerts", final_summary["loitering_events"])
        s2.metric("Restricted Zone Entries", final_summary["restricted_zone_events"])
        s3.metric("Crowding Events", final_summary["crowd_events"])

        col_a, col_b = st.columns(2)
        with col_a:
            with st.expander("Full summary (JSON)", expanded=False):
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
                    "⬇ Download Security Events (CSV)",
                    data=pd.DataFrame(events_history).to_csv(index=False),
                    file_name="security_events_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            if output_path and os.path.exists(output_path):
                with open(output_path, "rb") as vid_file:
                    st.download_button(
                        "⬇ Download Processed Video (MP4)",
                        data=vid_file.read(),
                        file_name="annotated_cctv_output.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
            elif not save_video:
                st.caption("Video export was disabled for this run — enable it in the sidebar to download.")

    except Exception as exc:
        st.session_state.run_status = "idle"
        st.error(f"An error occurred while processing the video: {exc}")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    main()