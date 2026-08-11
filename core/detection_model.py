"""
detection_model.py
Concrete implementation of BaseDetector for plain object detection
(bounding boxes) using YOLO + ByteTrack.
"""

from ultralytics import YOLO

from core.base_detector import BaseDetector
from infrastructure.logger import setup_logger

logger = setup_logger(__name__)

# COCO class ids we care about, mapped to readable names
CLASS_NAMES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


class DetectionModel(BaseDetector):
    """Runs YOLO detection + ByteTrack tracking on each frame."""

    def __init__(self, config: dict):
        self.model_config = config["model"]
        self.model = None  # not loaded yet — load() does that explicitly

    def load(self) -> None:
        weights_path = self.model_config["weights"]
        logger.info(f"Loading detection model: {weights_path}")
        try:
            self.model = YOLO(weights_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO weights from '{weights_path}'. "
                f"Check the path in config.yaml, or use a standard name "
                f"like 'yolo11n.pt' to auto-download it. "
                f"Original error: {e}"
            )

    def predict(self, frame):
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. Call load() before predict()."
            )

        results = self.model.track(
            frame,
            persist=True,
            conf=self.model_config["confidence"],
            iou=self.model_config["iou"],
            classes=self.model_config["classes"],
            device=self.model_config["device"],
            tracker="bytetrack_custom.yaml",
            verbose=False,
        )
        return results[0]