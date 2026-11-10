"""utils.py - Helpers for image validation, statistics and export.

Like detector.py, this module contains no Streamlit code.
"""

from __future__ import annotations

import io
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from detector import Detection

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_FILE_SIZE_MB = 10
MAX_PIXELS = 40_000_000  # about 40 megapixels; larger images are rejected


class InvalidImageError(ValueError):
    """Raised when an uploaded file is not a usable image.

    The message is written for end users, so the interface can display it as is.
    """


@dataclass(frozen=True)
class ClassStat:
    """Summary of one detected class."""

    class_name: str
    count: int
    mean_confidence: float


def load_image_from_bytes(data: bytes, filename: str) -> Image.Image:
    """Validate raw file bytes and return an upright RGB PIL image.

    Raises:
        InvalidImageError: if the file is empty, too large, of an unsupported
            type, corrupted, or has too many pixels.
    """
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            "Please upload a JPG, JPEG or PNG image."
        )
    if len(data) == 0:
        raise InvalidImageError("The uploaded file is empty.")
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise InvalidImageError(
            f"The file is too large. The maximum size is {MAX_FILE_SIZE_MB} MB."
        )

    try:
        # First pass: cheap integrity check. verify() makes the image object
        # unusable afterwards, so we open the data a second time to really use it.
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in ALLOWED_FORMATS:
                raise InvalidImageError(
                    "The file content is not a JPEG or PNG image, "
                    "even though its name says otherwise."
                )
            if probe.width * probe.height > MAX_PIXELS:
                raise InvalidImageError(
                    "The image resolution is too high. Please upload a smaller image."
                )
            probe.verify()

        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)  # apply the camera's rotation flag
        return image.convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError(
            "The file could not be read as an image. It may be corrupted."
        ) from exc


def class_statistics(detections: list[Detection]) -> list[ClassStat]:
    """Group detections by class: count and mean confidence per class.

    The result is sorted by count (highest first), then alphabetically.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for detection in detections:
        grouped[detection.class_name].append(detection.confidence)

    stats = [
        ClassStat(name, len(scores), sum(scores) / len(scores))
        for name, scores in grouped.items()
    ]
    stats.sort(key=lambda s: (-s.count, s.class_name))
    return stats


def detections_to_rows(detections: list[Detection]) -> list[dict]:
    """Turn detections into plain dictionaries, ready for a table in the UI."""
    rows = []
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.box
        rows.append(
            {
                "#": index,
                "Class": detection.class_name,
                "Confidence": round(detection.confidence, 3),
                "Box (x1, y1, x2, y2)": f"({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})",
            }
        )
    return rows


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes (used for the download button)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()