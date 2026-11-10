# VisionAI: Project Documentation

This document explains how VisionAI is built and why, and what has and has not been verified.

## 1. What the application does

A user uploads a JPG or PNG image. The app validates it, runs a pretrained YOLO11n detector, and shows the annotated image, summary metrics, per-class statistics, and a table of every detection. The annotated image can be downloaded as a PNG.

## 2. Data flow

```
Upload (Streamlit)
   -> utils.load_image_from_bytes   validate file, fix EXIF rotation, convert to RGB
   -> detector.ObjectDetector.detect   RGB -> BGR, YOLO inference, parse boxes, draw annotations
   -> utils.class_statistics / detections_to_rows   prepare data for charts and tables
   -> Streamlit renders images, metrics, charts, tables, download button
```

## 3. Modules

| File | Responsibility |
|---|---|
| `app.py` | Layout and interaction only. Contains no model logic. |
| `detector.py` | `ObjectDetector` loads the model and runs inference. Returns plain dataclasses (`Detection`, `DetectionResult`), not Ultralytics objects. Raises `ModelLoadError` and `DetectionError`. |
| `utils.py` | Upload validation (`InvalidImageError`), statistics, table rows, PNG encoding. |
| `tests/` | pytest suites for `detector.py` and `utils.py`. |

## 4. Design decisions

- **UI separated from logic.** `detector.py` and `utils.py` import no Streamlit, so they can be tested and reused without a browser.
- **Own data types.** The interface depends on small dataclasses defined in this project, not on Ultralytics internals, so the model could be swapped with limited changes.
- **Model cached once.** `@st.cache_resource` loads the model a single time per server process instead of on every interaction.
- **Colour order.** Pillow uses RGB while the OpenCV-based Ultralytics pipeline uses BGR. The detector converts in both directions; skipping this would silently reduce accuracy.
- **Layered upload validation.** File extension, size (10 MB), real image format (JPEG or PNG, regardless of file name), pixel count, and corruption are checked in turn, each with a user-readable message.
- **EXIF orientation.** Phone photos often store rotation as metadata. The image is rotated before inference so the model sees it upright.
- **Weights not committed.** `*.pt` files are git-ignored. Ultralytics downloads `yolo11n.pt` into `models/` on first use; this was verified by deleting the local file and loading the detector again.

## 5. Testing

Run with `python -m pytest -v`. The suite has 26 test cases:

- **Detector (13):** model loads with 80 COCO classes; a corrupt weights file raises `ModelLoadError`; the sample image yields a bus and people; detections are sorted and inside the image; a higher threshold never returns more objects; every result respects the threshold; a blank image gives an empty result with a valid output image; greyscale and RGBA inputs work; non-image input and out-of-range thresholds are rejected.
- **Utilities (13):** valid PNG and JPEG; wrong extension, empty file, oversized file, corrupted content, a GIF renamed to `.jpg`, too many pixels; EXIF rotation; class statistics (grouping, sorting, empty list); table rows; PNG encoding.

The Streamlit interface has no automated tests. The author checked it manually: uploading the sample image, changing the confidence threshold, downloading the annotated image, and uploading an empty file (rejected with a message).

## 6. Environment

Developed and tested on Windows with Python 3.14.7, CPU only (PyTorch CPU build). Other operating systems and GPUs have not been tested.

## 7. Known limitations

- Only the 80 COCO classes are recognised.
- The detector is used as pretrained; no fine-tuning or evaluation on custom data has been done, so no precision, recall, or mAP figures are claimed.
- Speed figures are informal observations, not benchmarks.
- With a single detected class, Streamlit's bar chart draws one very wide bar, which looks odd but is correct.