"""
Page 6: Export
Export annotations as YOLO dataset.
"""

import streamlit as st
import requests

API = "http://localhost:8000/api"

st.set_page_config(page_title="Export", page_icon="📦", layout="wide")

st.markdown("# 📦 YOLO Export")
st.markdown("Export your annotated dataset in YOLO format, ready for training.")
st.divider()

# ── Load data ──────────────────────────────────────────────────────────────────
datasets, models = [], []
try:
    datasets = requests.get(f"{API}/datasets/list", timeout=5).json()
    models = requests.get(f"{API}/models/list", timeout=5).json()
except Exception as e:
    st.error(f"Backend not reachable: {e}")

if not datasets:
    st.warning("No datasets available.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    dataset_name = st.selectbox("Dataset", [d["name"] for d in datasets])
with col2:
    model_name = st.selectbox("Model (for class definitions)", [m["name"] for m in models])

# Show class list
selected_model = next((m for m in models if m["name"] == model_name), None)
if selected_model and selected_model.get("classes"):
    with st.expander("📋 Class List"):
        for i, cls in enumerate(selected_model["classes"]):
            st.text(f"  [{i}] {cls}")

# Export structure preview
st.markdown("### Output Structure")
st.code("""
dataset_name/
├── images/
│   ├── frame_00001.jpg
│   ├── frame_00002.jpg
│   └── ...
├── labels/
│   ├── frame_00001.txt
│   ├── frame_00002.txt
│   └── ...
└── dataset.yaml
""")

st.markdown("### Label Format (YOLO)")
st.code("class_id  cx  cy  width  height  (all normalized 0-1)")

st.divider()

# Stats preview
try:
    resp = requests.get(f"{API}/adala/{dataset_name}/results", timeout=5)
    if resp.status_code == 200:
        results = resp.json()
        all_dets = [d for r in results for d in r.get("detections", [])]
        approved = [d for d in all_dets if d.get("adala_status") != "rejected"]
        rejected = [d for d in all_dets if d.get("adala_status") == "rejected"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Images to export", len(results))
        col2.metric("Labels to export", len(approved))
        col3.metric("Rejected (excluded)", len(rejected))
except Exception:
    pass

# ── Export Button ──────────────────────────────────────────────────────────────
if st.button("📦 Export YOLO Dataset", type="primary", use_container_width=True):
    with st.spinner("Packaging YOLO dataset..."):
        try:
            resp = requests.post(
                f"{API}/export/yolo",
                json={"dataset_name": dataset_name, "model_name": model_name},
                timeout=120,
            )
            if resp.status_code == 200:
                zip_bytes = resp.content
                st.success("✅ Export ready!")
                st.download_button(
                    label="⬇️ Download YOLO Dataset ZIP",
                    data=zip_bytes,
                    file_name=f"{dataset_name}_yolo.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            else:
                st.error(f"Export failed: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")
