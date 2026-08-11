"""
report.py
Generates the final analytics report in two formats after a video
finishes processing: JSON (raw data, machine-readable) and Markdown
(human-readable, renders nicely on GitHub or when shared with others).
"""

import json
import os
from datetime import datetime

from infrastructure.logger import setup_logger

logger = setup_logger(__name__)


def _format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def _build_markdown(summary: dict, video_duration_sec: float, processing_fps: float) -> str:
    duration_str = _format_duration(video_duration_sec)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Smart CCTV Video Analytics Report",
        "",
        f"**Generated:** {generated_at}  ",
        f"**Video Duration:** {duration_str}  ",
        f"**System Processing Speed:** {processing_fps:.1f} FPS",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| People (unique) | {summary['total_people']} |",
        f"| Vehicles (unique) | {summary['total_vehicles']} |",
        f"| Max People At Once | {summary['max_people_at_once']} |",
        f"| Average Dwell Time | {summary['average_dwell_time_sec']}s |",
        f"| Total Alerts | {summary['total_alerts']} |",
        f"| Restricted Zone Violations | {summary['restricted_zone_violations']} |",
        f"| Loitering Events | {summary['loitering_events']} |",
        f"| Crowd Events | {summary['crowd_events']} |",
        "",
    ]

    if summary["alerts_detail"]:
        lines += ["## Alerts", "", "| Time | Type | Message |", "|---|---|---|"]
        for alert in summary["alerts_detail"]:
            lines.append(f"| {alert['time_sec']}s | {alert['type']} | {alert['message']} |")
        lines.append("")

    return "\n".join(lines)


def generate_report(summary: dict, video_duration_sec: float, processing_fps: float,
                     output_dir: str = "outputs") -> tuple[str, str]:
    """Writes report.json and report.md to output_dir. Returns their paths."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"report_{timestamp}.json")
    md_path = os.path.join(output_dir, f"report_{timestamp}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": timestamp,
                "video_duration_sec": video_duration_sec,
                "processing_fps": processing_fps,
                **summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    markdown_content = _build_markdown(summary, video_duration_sec, processing_fps)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info(f"Report saved: {json_path}, {md_path}")
    return json_path, md_path