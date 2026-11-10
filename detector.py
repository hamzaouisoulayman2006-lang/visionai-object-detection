"""detector.py - Core object detection logic for VisionAI.

This module deliberately contains no Streamlit code, so it can be tested
and reused independently of the web interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

# Resolve the default weights path relative to this file, so the code works
# no matter which folder the program is started from.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolo11n.pt"


class ModelLoadError(RuntimeError):
    """Raised when the YOLO model cannot be loaded or downloaded."""


class DetectionError(RuntimeError):
    """Raised when inference fails on a given image."""


@dataclass(frozen=True)
class Detection:
    """A single detected object."""

    class_id: int
    class_name: str
    confidence: float  # 0.0 - 1.0
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels


@dataclass
class DetectionResult:
    """Everything produced by one detection run."""

    detections: list[Detection]
    annotated_image: Image.Image  # RGB image with boxes and labels drawn
    inference_ms: float  # time spent in the model call
    image_size: tuple[int, int]  # (width, height) of the input image

    @property
    def count(self) -> int:
        """Total number of detected objects."""
        return len(self.detections)


class ObjectDetector:
    """Wraps a pretrained YOLO model for object detection on single images."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """Load the model; Ultralytics downloads known weights if missing."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            return YOLO(str(self.model_path))
        except Exception as exc:  # broad on purpose: wrap any load/download failure
            raise ModelLoadError(
                f"Could not load the model '{self.model_path.name}'. "
                "Check your internet connection (needed for the first download) "
                f"or place the weights file at: {self.model_path}"
            ) from exc

    @property
    def class_names(self) -> dict[int, str]:
        """Mapping from class id to class name (80 classes for COCO)."""
        return dict(self.model.names)

    def detect(
        self, image: Image.Image, conf_threshold: float = 0.25
    ) -> DetectionResult:
        """Run object detection on one image.

        Args:
            image: A PIL image (any mode; it is converted to RGB).
            conf_threshold: Detections with a lower confidence are discarded.

        Returns:
            A DetectionResult with detections sorted by confidence (highest first).
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0.0 and 1.0")

        rgb = image.convert("RGB")
        # Ultralytics expects NumPy arrays in BGR order, Pillow uses RGB.
        bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])

        start = time.perf_counter()
        try:
            results = self.model.predict(source=bgr, conf=conf_threshold, verbose=False)
        except Exception as exc:
            raise DetectionError("The model failed to process this image.") from exc
        inference_ms = (time.perf_counter() - start) * 1000

        result = results[0]
        detections = self._parse_detections(result)

        # result.plot() draws boxes and labels and returns a BGR array.
        annotated_bgr = result.plot()
        annotated = Image.fromarray(np.ascontiguousarray(annotated_bgr[:, :, ::-1]))

        return DetectionResult(
            detections=detections,
            annotated_image=annotated,
            inference_ms=inference_ms,
            image_size=rgb.size,
        )

    def _parse_detections(self, result) -> list[Detection]:
        """Convert Ultralytics output into our own simple Detection objects."""
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.model.names[class_id],
                    confidence=float(box.conf[0]),
                    box=(x1, y1, x2, y2),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections