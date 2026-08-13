# Smart CCTV Video Analytics for Real-Time Security Monitoring

An AI-powered video analytics system that turns a plain CCTV feed into a system
that detects people and vehicles, tracks them across frames, and raises
real-time alerts for suspicious loitering, restricted-zone entry, and
overcrowding — all through a live Streamlit dashboard.

**Status:** Live and deployed. Built for CPU-only environments (no GPU
required).

---

## Features

- **Detection** — People and vehicles detected in every frame with YOLOv11 (nano).
- **Tracking** — Each object keeps a persistent ID across frames via ByteTrack.
- **Loitering detection** — Flags sustained *stillness*, not just presence,
  so people walking normally are never falsely flagged.
- **Safe Waiting Zone** — A configurable area (e.g. a queue or reception
  desk) where people are exempt from the loitering check.
- **Restricted Zone** — Alerts once when someone enters a defined area, not
  on every frame they remain inside it.
- **Crowd detection** — Alerts once when the person count crosses a
  threshold, and resets automatically when it clears.
- **Live dashboard** — Upload a video and see the processed footage, running
  stats, and the full alert log together in one screen.
- **Segmentation-ready** — An optional YOLO-seg mode is built in for
  pixel-level masks, with frame-skipping and frame-resizing so it stays
  usable without a GPU.

---

## Tech Stack

| Tool | Role |
|---|---|
| [Ultralytics YOLOv11](https://docs.ultralytics.com/) | Object detection (and optional segmentation) |
| ByteTrack (via Ultralytics) | Multi-object tracking with persistent IDs |
| OpenCV | Video I/O, frame processing, zone drawing |
| PyYAML | Config-driven settings — nothing is hardcoded |
| Streamlit | Live web dashboard |
| Python 3 | Core language |

---

## Project Structure

```
Smart-CCTV/
├── app.py                       # Streamlit entry point (single-page dashboard)
├── config.yaml                  # Every tunable setting lives here
├── requirements.txt
│
├── core/                        # Business logic — framework-agnostic
│   ├── base_detector.py           # Shared interface for all detector models
│   ├── detection_model.py          # YOLO detection (bounding boxes)
│   ├── segmentation_model.py        # YOLO-seg (pixel masks), CPU-optimized
│   ├── tracker.py                    # Converts raw results to TrackedObject
│   └── analytics.py                   # Events (loitering/zone/crowd) + summary()
│
├── infrastructure/               # Technical plumbing
│   ├── logger.py                   # Unified logging
│   ├── config_loader.py             # Reads config.yaml
│   └── video_io.py                   # Open video, read FPS, write output
│
├── models/                       # YOLO weights (auto-downloaded, gitignored)
├── videos/                       # Test videos (gitignored)
└── outputs/                      # Processed video output (gitignored)
```

Each layer only knows about the one below it — swapping the model or the UI
never requires touching the other layers.

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/fatma-shehata/Smart-CCTV-Analytics.git
cd Smart-CCTV-Analytics

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

The YOLO weights (`yolo11n.pt`) download automatically the first time the
app runs. To pre-download them manually:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

## Getting test videos

No video files are tracked in this repo. Add your own `.mp4` files to
`videos/`, sourced from:

- [Pexels Videos](https://www.pexels.com/videos/) — search "CCTV footage", "parking lot camera"
- [Pixabay Videos](https://pixabay.com/videos/)
- [VIRAT Dataset](https://viratdata.org/) — real surveillance footage
- [MOT17 Benchmark](https://motchallenge.net/data/MOT17/) — for tracking accuracy testing

## Configuring zones

The `restricted_zone` and `safe_waiting_zone` coordinates in `config.yaml`
are pixel points specific to a video's resolution and camera angle. To find
the right coordinates for your own video, extract a frame and mark the
corners visually before editing the config.

## Running the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`),
upload a video, and run the analysis.

---

## Configuration

Every tunable value lives in `config.yaml` — detection confidence, IoU
threshold, loitering duration, crowd threshold, zone coordinates, and CPU
performance settings (frame resize width, segmentation frame-skip rate).
No thresholds are hardcoded in the source.

---

## Known Limitations

- **Class coverage** — Built on COCO classes; no fine-grained distinction
  between vehicle types (e.g. sedan vs. SUV).
- **ID switching under occlusion** — Heavy crowding can cause a tracked ID
  to change, which affects dwell-time precision for that person.
- **CPU-bound throughput** — Processing runs below true real-time speed on
  CPU-only hardware; a GPU removes this ceiling.
- **Segmentation frame-skipping** — When segmentation mode is enabled, masks
  are only recomputed every N frames (not every frame) to stay usable
  without a GPU. This is a deliberate, documented accuracy/speed trade-off.

---

## Team

Built by Fatma and team — AI Program, Kafr El-Sheikh University.
