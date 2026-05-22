"""
Page 5: Annotation Review
Visualize and manually edit annotations after Adala validation.
"""

import streamlit as st
import requests
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os

API = "http://localhost:8000/api"
UPLOAD_DIR = "data/uploads"

STATUS_COLORS = {
    "approved": (0, 200, 0),
    "rejected": (0, 0, 220),
    "relabeled": (220, 130, 0),
    "flagged": (0, 165, 255),
    "pending": (180, 180, 180),
}

STATUS_ICONS = {
    "approved": "✅", "rejected": "❌",
    "relabeled": "🔄", "flagged": "🚩", "pending": "⚪"
}


def draw_detections_on_image(img_path: str, detections: list) -> np.ndarray:
    """Draw bounding boxes on image."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max(0.4, min(w, h) / 800)
    thickness = max(1, int(scale * 2))

    for det in detections:
        if det.get("adala_status") == "rejected":
            continue
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        status = det.get("adala_status", "pending")
        color = STATUS_COLORS.get(status, STATUS_COLORS["pending"])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.get('class', '?')} {det.get('confidence', 0):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.5, 1)
        label_y = max(y1 - 5, th + 5)
        cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(img, label, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.5, (255, 255, 255), 1)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


st.set_page_config(page_title="Annotation Review", page_icon="👁️", layout="wide")
st.markdown("# 👁️ Annotation Review")
st.markdown("Inspect, edit, and approve annotations before export.")
st.divider()

# ── Dataset Selection ──────────────────────────────────────────────────────────
datasets = []
try:
    datasets = requests.get(f"{API}/datasets/list", timeout=5).json()
except Exception:
    st.error("Backend not reachable")

if not datasets:
    st.warning("No datasets found.")
    st.stop()

dataset_name = st.selectbox("Dataset", [d["name"] for d in datasets])

# Load annotations
try:
    resp = requests.get(f"{API}/adala/{dataset_name}/results", timeout=10)
    annotations = resp.json() if resp.status_code == 200 else []
except Exception as e:
    st.error(f"Could not load annotations: {e}")
    annotations = []

if not annotations:
    st.info("No annotations yet. Run inference and Adala validation first.")
    st.stop()

# ── Summary Stats ──────────────────────────────────────────────────────────────
all_dets = [d for a in annotations for d in a.get("detections", [])]
counts = {}
for d in all_dets:
    s = d.get("adala_status", "pending")
    counts[s] = counts.get(s, 0) + 1

cols = st.columns(len(counts) + 1)
cols[0].metric("Total Detections", len(all_dets))
for i, (status, count) in enumerate(counts.items()):
    icon = STATUS_ICONS.get(status, "⚪")
    cols[i + 1].metric(f"{icon} {status}", count)

st.divider()

# ── Status Filter ───────────────────────────────────────────────────────────────
status_filter = st.selectbox(
    "Filter by status",
    ["all", "approved", "rejected", "relabeled", "flagged", "pending"],
    index=0,
    help="Show only images containing detections with the selected status. Useful for reviewing flagged detections.",
)

filtered_annotations = annotations
if status_filter != "all":
    filtered_annotations = [
        a for a in annotations
        if any(d.get("adala_status") == status_filter for d in a.get("detections", []))
    ]

if not filtered_annotations:
    st.info(f"No images with '{status_filter}' detections.")
    st.stop()

st.divider()

# ── Image-by-Image Review ──────────────────────────────────────────────────────
image_names = [a["image_name"] for a in filtered_annotations]
selected_img = st.selectbox("Select Image to Review", image_names)

ann = next((a for a in filtered_annotations if a["image_name"] == selected_img), None)
if not ann:
    st.warning("Annotation not found.")
    st.stop()

detections = ann.get("detections", [])
img_path = os.path.join(UPLOAD_DIR, dataset_name, selected_img)

# Draw image
col_img, col_edit = st.columns([3, 2])

with col_img:
    st.markdown("#### Annotated Image")
    annotated = draw_detections_on_image(img_path, detections)
    if annotated is not None:
        st.image(annotated, use_container_width=True)
    else:
        st.error("Could not load image file.")

with col_edit:
    st.markdown("#### Edit Detections")
    st.caption(f"{len(detections)} detection(s)")

    updated_detections = []
    for i, det in enumerate(detections):
        status = det.get("adala_status", "pending")
        icon = STATUS_ICONS.get(status, "⚪")

        with st.expander(f"{icon} [{i}] {det.get('class', '?')} — conf: {det.get('confidence', 0):.3f}"):
            # Show Adala reasoning
            if det.get("adala_reasoning"):
                st.info(f"🤖 Adala: *{det['adala_reasoning']}*")

            new_class = st.text_input(f"Class", value=det.get("class", ""), key=f"cls_{i}")
            new_status = st.selectbox(
                "Status",
                ["approved", "rejected", "relabeled", "flagged", "pending"],
                index=["approved", "rejected", "relabeled", "flagged", "pending"].index(status),
                key=f"status_{i}",
            )

            bbox = det.get("bbox", [0, 0, 0, 0])
            st.caption(f"BBox: x1={bbox[0]:.0f}, y1={bbox[1]:.0f}, x2={bbox[2]:.0f}, y2={bbox[3]:.0f}")

            updated_det = dict(det)
            updated_det["class"] = new_class
            updated_det["adala_status"] = new_status
            updated_detections.append(updated_det)

    if st.button("💾 Save Changes", type="primary"):
        try:
            resp = requests.put(
                f"{API}/adala/{dataset_name}/{selected_img}",
                json={"detections": updated_detections},
                timeout=10,
            )
            if resp.status_code == 200:
                st.success("✅ Annotation saved!")
                st.rerun()
            else:
                st.error(f"Save failed: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.info("➡️ When done reviewing, proceed to the **Export** page to download YOLO labels.")
