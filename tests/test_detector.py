"""Tests for detector.py. They load the real YOLO11n model once and reuse it."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from detector import ModelLoadError, ObjectDetector

SAMPLE_IMAGE = Path(__file__).resolve().parent.parent / "assets" / "samples" / "bus.jpg"


@pytest.fixture(scope="session")
def detector() -> ObjectDetector:
    """Load the model once for the whole test session (loading is slow)."""
    return ObjectDetector()


@pytest.fixture(scope="session")
def bus_image() -> Image.Image:
    if not SAMPLE_IMAGE.exists():
        pytest.skip(f"Sample image not found: {SAMPLE_IMAGE}")
    return Image.open(SAMPLE_IMAGE).convert("RGB")


# ---------- model initialization ----------


def test_model_loads_with_coco_classes(detector):
    names = detector.class_names
    assert len(names) == 80  # the COCO dataset has 80 classes
    assert "person" in names.values()
    assert "bus" in names.values()


def test_invalid_model_file_raises_model_load_error(tmp_path):
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"this is not a real model file")
    with pytest.raises(ModelLoadError):
        ObjectDetector(model_path=broken)


# ---------- detection ----------


def test_detects_bus_and_people(detector, bus_image):
    result = detector.detect(bus_image, conf_threshold=0.25)
    names = {d.class_name for d in result.detections}
    assert result.count >= 1
    assert "bus" in names
    assert "person" in names
    assert result.inference_ms > 0


def test_detections_are_valid_and_sorted(detector, bus_image):
    result = detector.detect(bus_image, conf_threshold=0.25)
    width, height = bus_image.size
    assert result.image_size == (width, height)

    confidences = [d.confidence for d in result.detections]
    assert confidences == sorted(confidences, reverse=True)

    for detection in result.detections:
        assert 0.0 <= detection.confidence <= 1.0
        x1, y1, x2, y2 = detection.box
        assert x1 < x2 and y1 < y2
        # Small tolerance: boxes may touch the image border.
        assert x1 >= -1 and y1 >= -1 and x2 <= width + 1 and y2 <= height + 1


# ---------- confidence threshold ----------


def test_higher_threshold_never_finds_more_objects(detector, bus_image):
    low = detector.detect(bus_image, conf_threshold=0.1)
    high = detector.detect(bus_image, conf_threshold=0.8)
    assert high.count <= low.count


def test_all_detections_respect_threshold(detector, bus_image):
    result = detector.detect(bus_image, conf_threshold=0.7)
    assert all(d.confidence >= 0.7 for d in result.detections)


# ---------- empty results and output image ----------


def test_blank_image_gives_empty_result(detector):
    blank = Image.new("RGB", (640, 640), (128, 128, 128))
    result = detector.detect(blank, conf_threshold=0.9)
    assert result.count == 0
    assert result.detections == []
    # Even with nothing detected, we must still get a valid output image.
    assert result.annotated_image.size == (640, 640)


def test_annotated_image_is_valid_output(detector, bus_image):
    result = detector.detect(bus_image, conf_threshold=0.25)
    annotated = result.annotated_image
    assert isinstance(annotated, Image.Image)
    assert annotated.mode == "RGB"
    assert annotated.size == bus_image.size
    # Boxes and labels were drawn, so the pixels must differ from the original.
    assert not np.array_equal(np.asarray(annotated), np.asarray(bus_image))


# ---------- input handling ----------


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_non_rgb_inputs_are_accepted(detector, mode):
    image = Image.new(mode, (320, 320))
    result = detector.detect(image, conf_threshold=0.5)
    assert result.annotated_image.size == (320, 320)


def test_non_image_input_raises_type_error(detector):
    with pytest.raises(TypeError):
        detector.detect("bus.jpg")  # a file name, not a PIL image


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.5])
def test_invalid_threshold_raises_value_error(detector, bus_image, bad_threshold):
    with pytest.raises(ValueError):
        detector.detect(bus_image, conf_threshold=bad_threshold)