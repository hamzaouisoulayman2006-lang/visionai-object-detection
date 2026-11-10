"""Tests for utils.py. They need no YOLO model, so they run in seconds."""

import io

import pytest
from PIL import Image

import utils
from detector import Detection
from utils import (
    InvalidImageError,
    class_statistics,
    detections_to_rows,
    image_to_png_bytes,
    load_image_from_bytes,
)


def make_bytes(fmt: str, size=(40, 20), **save_kwargs) -> bytes:
    """Create a small image in memory and return its encoded bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def make_detection(name: str, confidence: float) -> Detection:
    return Detection(
        class_id=0,
        class_name=name,
        confidence=confidence,
        box=(0.0, 0.0, 10.0, 10.0),
    )


# ---------- image loading and validation ----------


def test_valid_png_loads_as_rgb():
    image = load_image_from_bytes(make_bytes("PNG"), "photo.png")
    assert image.mode == "RGB"
    assert image.size == (40, 20)


def test_valid_jpeg_loads_as_rgb():
    image = load_image_from_bytes(make_bytes("JPEG"), "photo.JPG")  # uppercase ok
    assert image.mode == "RGB"


def test_unsupported_extension_is_rejected():
    with pytest.raises(InvalidImageError, match="Unsupported file type"):
        load_image_from_bytes(make_bytes("GIF"), "animation.gif")


def test_empty_file_is_rejected():
    with pytest.raises(InvalidImageError, match="empty"):
        load_image_from_bytes(b"", "photo.jpg")


def test_oversized_file_is_rejected():
    too_big = b"x" * (utils.MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
    with pytest.raises(InvalidImageError, match="too large"):
        load_image_from_bytes(too_big, "photo.jpg")


def test_corrupted_content_is_rejected():
    with pytest.raises(InvalidImageError, match="could not be read"):
        load_image_from_bytes(b"this is not an image", "fake.jpg")


def test_wrong_format_with_jpg_name_is_rejected():
    # A real GIF renamed to .jpg must not get through.
    with pytest.raises(InvalidImageError, match="not a JPEG or PNG"):
        load_image_from_bytes(make_bytes("GIF"), "renamed.jpg")


def test_too_many_pixels_is_rejected(monkeypatch):
    monkeypatch.setattr(utils, "MAX_PIXELS", 100)  # the test image has 800 pixels
    with pytest.raises(InvalidImageError, match="resolution is too high"):
        load_image_from_bytes(make_bytes("PNG"), "big.png")


def test_exif_rotation_is_applied():
    exif = Image.Exif()
    exif[0x0112] = 6  # EXIF orientation 6: the photo must be rotated by 90 degrees
    data = make_bytes("JPEG", size=(40, 20), exif=exif)
    image = load_image_from_bytes(data, "phone.jpg")
    assert image.size == (20, 40)  # width and height swapped


# ---------- statistics and export ----------


def test_class_statistics_groups_and_sorts():
    detections = [
        make_detection("person", 0.9),
        make_detection("person", 0.7),
        make_detection("bus", 0.8),
        make_detection("car", 0.5),
        make_detection("car", 0.6),
    ]
    stats = class_statistics(detections)
    # Sorted by count (highest first), then alphabetically: car, person, bus
    assert [s.class_name for s in stats] == ["car", "person", "bus"]
    assert [s.count for s in stats] == [2, 2, 1]
    person = next(s for s in stats if s.class_name == "person")
    assert person.mean_confidence == pytest.approx(0.8)


def test_class_statistics_empty():
    assert class_statistics([]) == []


def test_detections_to_rows_format():
    detection = Detection(
        class_id=5,
        class_name="bus",
        confidence=0.94321,
        box=(4.2, 229.7, 796.0, 728.4),
    )
    rows = detections_to_rows([detection])
    assert rows[0]["#"] == 1
    assert rows[0]["Class"] == "bus"
    assert rows[0]["Confidence"] == 0.943
    assert rows[0]["Box (x1, y1, x2, y2)"] == "(4, 230, 796, 728)"


def test_image_to_png_bytes_is_a_png():
    data = image_to_png_bytes(Image.new("RGB", (10, 10)))
    assert data.startswith(b"\x89PNG")