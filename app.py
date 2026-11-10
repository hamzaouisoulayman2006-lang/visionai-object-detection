"""app.py - Streamlit interface for VisionAI.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from detector import DetectionError, ModelLoadError, ObjectDetector
from utils import (
    MAX_FILE_SIZE_MB,
    InvalidImageError,
    class_statistics,
    detections_to_rows,
    image_to_png_bytes,
    load_image_from_bytes,
)

st.set_page_config(page_title="VisionAI", page_icon="👁", layout="wide")

CUSTOM_CSS = """
<style>
.block-container, [data-testid="stMainBlockContainer"] {padding-top: 5rem !important; max-width: 1200px;}
[data-testid="stMetric"] {
    background: #111a2e;
    border: 1px solid #1e2a44;
    border-radius: 10px;
    padding: 14px 16px;
}
.visionai-title {font-size: 4.5rem; font-weight: 700; margin-bottom: 0;}
.visionai-title span {color: #3b82f6;}
.visionai-sub {color: #94a3b8; margin-top: 0.2rem; font-size: 1.05rem;}
</style>
"""


@st.cache_resource(show_spinner="Loading the YOLO model...")
def load_detector() -> ObjectDetector:
    """Load the model once and reuse it across reruns and sessions."""
    return ObjectDetector()


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<p class="visionai-title">Vision<span>AI</span></p>'
        '<p class="visionai-sub">AI-powered object detection and image analysis</p>',
        unsafe_allow_html=True,
    )
    st.write(
        "Upload an image and a pretrained YOLO model will locate and label the "
        "objects it recognizes, with a confidence score for each detection."
    )


def render_sidebar() -> float:
    """Draw the detection settings and return the chosen confidence threshold."""
    st.sidebar.header("Detection settings")
    threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Detections with a confidence below this value are hidden. "
        "Lower values find more objects but include more mistakes.",
    )
    st.sidebar.caption(
        f"Accepted files: JPG, JPEG, PNG (max {MAX_FILE_SIZE_MB} MB)."
    )
    return threshold


def render_results(uploaded_file, detector: ObjectDetector, threshold: float) -> None:
    """Validate the upload, run detection, and show the results dashboard."""
    try:
        image = load_image_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
    except InvalidImageError as exc:
        st.error(str(exc))
        return

    try:
        with st.spinner("Detecting objects..."):
            result = detector.detect(image, conf_threshold=threshold)
    except DetectionError as exc:
        st.error(str(exc))
        return

    st.subheader("Results")
    left, right = st.columns(2)
    with left:
        st.caption("Original image")
        st.image(image, width="stretch")
    with right:
        st.caption("Processed image")
        st.image(result.annotated_image, width="stretch")

    stats = class_statistics(result.detections)
    mean_conf = (
        sum(d.confidence for d in result.detections) / result.count
        if result.count
        else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objects detected", result.count)
    c2.metric("Unique classes", len(stats))
    c3.metric("Mean confidence", f"{mean_conf:.0%}")
    c4.metric("Inference time", f"{result.inference_ms:.0f} ms")

    if result.count == 0:
        st.warning(
            "No objects were detected at this threshold. "
            "Try lowering the confidence threshold in the sidebar."
        )
        return

    st.download_button(
        "Download annotated image",
        data=image_to_png_bytes(result.annotated_image),
        file_name=f"{Path(uploaded_file.name).stem}_visionai.png",
        mime="image/png",
    )

    st.subheader("Detection statistics")
    chart_col, table_col = st.columns(2)
    with chart_col:
        st.caption("Number of detections per class")
        counts = pd.DataFrame(
            {"Class": [s.class_name for s in stats], "Count": [s.count for s in stats]}
        ).set_index("Class")
        st.bar_chart(counts)
    with table_col:
        st.caption("Summary by class")
        summary = pd.DataFrame(
            {
                "Class": [s.class_name for s in stats],
                "Count": [s.count for s in stats],
                "Mean confidence": [round(s.mean_confidence, 3) for s in stats],
            }
        )
        st.dataframe(summary, hide_index=True, width="stretch")

    st.subheader("Individual detections")
    st.dataframe(
        pd.DataFrame(detections_to_rows(result.detections)),
        hide_index=True,
        width="stretch",
    )


def render_about(detector: ObjectDetector) -> None:
    st.divider()
    st.subheader("About the model")
    st.write(
        "VisionAI uses **YOLO11n**, the smallest and fastest model of the "
        "Ultralytics YOLO11 family. It is pretrained on the COCO dataset, so it "
        "recognizes 80 everyday object classes and nothing outside them. "
        "Detection runs on your own machine."
    )
    with st.expander("Show all recognizable classes"):
        st.write(", ".join(sorted(detector.class_names.values())))


def main() -> None:
    render_header()
    threshold = render_sidebar()

    try:
        detector = load_detector()
    except ModelLoadError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("Upload an image")
    uploaded_file = st.file_uploader(
        "Drag and drop an image here, or click to browse",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload a JPG or PNG image to start. Try the sample in assets/samples.")
    else:
        render_results(uploaded_file, detector, threshold)

    render_about(detector)


main()