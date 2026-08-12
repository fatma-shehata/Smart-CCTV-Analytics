"""
segmentation_model.py
Concrete implementation of BaseDetector for instance segmentation
(pixel-level masks) using YOLO-seg + ByteTrack.

Segmentation is significantly more expensive than plain detection on CPU,
so this class applies two optimizations controlled from config.yaml:
  1. Frames are resized down before inference.
  2. Segmentation only runs every N frames (see performance.segmentation_every_n_frames).
    On skipped frames, the last known result is reused.
"""

import cv2

from ultralytics import YOLO

from core.base_detector import BaseDetector, CLASS_NAMES
from infrastructure.logger import setup_logger

logger = setup_logger(__name__)




class SegmentationModel(BaseDetector):
    """Runs YOLO-seg detection + ByteTrack tracking on each frame,
    with frame-skipping to stay usable on CPU-only machines."""

    def __init__(self, config: dict):
        self.model_config = config["model"]
        self.perf_config = config["performance"]
        self.model = None

        self._frame_counter = 0
        self._last_result = None  

    def load(self) -> None:
        weights_path = self.model_config.get("segmentation_weights", "yolo11n-seg.pt")
        logger.info(f"Loading segmentation model: {weights_path}")
        try:
            self.model = YOLO(weights_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO segmentation weights from '{weights_path}'. "
                f"Use a standard name like 'yolo11n-seg.pt' to auto-download it. "
                f"Original error: {e}"
            )

    def _resize_for_inference(self, frame):
        """Shrinks the frame before running the (expensive) segmentation model."""
        target_width = self.perf_config["frame_resize_width"]
        h, w = frame.shape[:2]
        if w <= target_width:
            return frame, 1.0  # no resize needed, scale factor is 1
        scale = target_width / w
        resized = cv2.resize(frame, (target_width, int(h * scale)))
        return resized, scale

    def predict(self, frame):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")

        every_n = self.perf_config["segmentation_every_n_frames"]
        self._frame_counter += 1

        if self._last_result is not None and self._frame_counter % every_n != 0:
            return self._last_result

        original_h, original_w = frame.shape[:2]
        small_frame, scale = self._resize_for_inference(frame)

        results = self.model.track(
            small_frame,
            persist=True,
            conf=self.model_config["confidence"],
            iou=self.model_config["iou"],
            classes=self.model_config["classes"],
            device=self.model_config["device"],
            tracker=self.model_config.get("tracker_type", "bytetrack.yaml"),
            verbose=False,
        )

        result = results[0]

        if scale != 1.0:
            # Boxes were computed on the resized frame — scale coordinates
            # back up so they match the original frame.
            if result.boxes is not None:
                result.boxes.xyxy[:] = result.boxes.xyxy / scale

            # Masks need the same correction, but as a full image resize
            # (they're per-pixel, not just coordinates).
            if result.masks is not None:
                result.masks.data = self._resize_masks(
                    result.masks.data, original_w, original_h
                )

        self._last_result = result
        return result

    def _resize_masks(self, masks_tensor, target_w: int, target_h: int):
        """Resizes segmentation masks from the (smaller) inference size
        back up to the original frame dimensions."""
        import torch
        import torch.nn.functional as F

        # masks_tensor shape: (num_objects, mask_h, mask_w)
        masks = masks_tensor.unsqueeze(1)  # add channel dim for interpolate
        resized = F.interpolate(
            masks, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return resized.squeeze(1)