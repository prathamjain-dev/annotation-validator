"""
Page 3: Auto Annotation
Run ONNX inference on dataset images.
"""

import streamlit as st
import requests
import time

API = "http://localhost:8000/api"

st.set_page_config(page_title="Auto Annotation", page_icon="🔍", layout="wide")

st.markdown("# 🔍 Auto Annotation")
st.markdown("Run your ONNX detector to generate bounding box annotations automatically.")
st.divider()

# ── Load Datasets & Models ─────────────────────────────────────────────────────
datasets, models = [], []
try:
    datasets = requests.get(f"{API}/datasets/list", timeout=5).json()
    models = requests.get(f"{API}/models/list", timeout=5).json()
except Exception as e:
    st.error(f"Could not reach backend: {e}")

if not datasets:
    st.warning("⚠️ No datasets found. Upload a dataset first.")
    st.stop()
if not models:
    st.warning("⚠️ No ONNX models found. Upload a model first.")
    st.stop()

# ── Configuration ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    dataset_name = st.selectbox("Select Dataset", [d["name"] for d in datasets])
with col2:
    model_name = st.selectbox("Select ONNX Model", [m["name"] for m in models])

col3, col4, col5 = st.columns(3)
with col3:
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.99, 0.25, 0.01)
with col4:
    iou_threshold = st.slider("IoU Threshold (NMS)", 0.1, 0.99, 0.45, 0.01)
with col5:
    postprocessor = st.selectbox("Postprocessor", ["yolo"], help="Custom postprocessors can be registered")

# Show selected model details
selected_model = next((m for m in models if m["name"] == model_name), None)
if selected_model:
    with st.expander("ℹ️ Model Details"):
        st.json({
            "input_shape": selected_model.get("input_shape"),
            "output_shape": selected_model.get("output_shape"),
            "classes": selected_model.get("classes"),
        })

st.divider()

# ── Run Inference ──────────────────────────────────────────────────────────────
if st.button("🚀 Run Auto Detection", type="primary", use_container_width=True):
    with st.spinner("Running ONNX inference..."):
        progress = st.progress(0, text="Initializing detector...")
        log_area = st.empty()

        try:
            resp = requests.post(
                f"{API}/inference/run",
                json={
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "conf_threshold": conf_threshold,
                    "iou_threshold": iou_threshold,
                    "postprocessor": postprocessor,
                },
                timeout=600,
            )

            if resp.status_code == 200:
                data = resp.json()
                progress.progress(100, text="Done!")

                st.success(f"✅ Inference complete!")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Images", data["total_images"])
                col2.metric("Processed", data["processed"])
                col3.metric("Errors", data["errors"])
                total_dets = sum(r["detections"] for r in data["results"])
                col4.metric("Total Detections", total_dets)

                # Per-image log
                st.markdown("### Detection Log")
                for r in data["results"]:
                    icon = "✅" if r["detections"] > 0 else "⭕"
                    st.text(f"{icon}  {r['image']:<40}  {r['detections']} detections")

                if data["error_details"]:
                    with st.expander(f"⚠️ {data['errors']} Errors"):
                        for err in data["error_details"]:
                            st.error(f"{err['image']}: {err['error']}")

                st.info("➡️ Proceed to **Adala Validation** to run LLM reasoning on these detections.")
            else:
                st.error(f"Inference failed: {resp.text}")

        except requests.Timeout:
            st.error("Request timed out. Try a smaller dataset or increase timeout.")
        except Exception as e:
            st.error(f"Error: {e}")

# ── Preview Existing Detections ────────────────────────────────────────────────
st.divider()
st.markdown("### Current Detections")
try:
    resp = requests.get(f"{API}/inference/{dataset_name}/detections", timeout=5)
    if resp.status_code == 200:
        annotations = resp.json()
        if annotations:
            total = sum(len(a["detections"]) for a in annotations)
            st.metric("Total saved detections", total)
            show_n = st.slider("Preview images", 1, min(len(annotations), 20), 5)
            for ann in annotations[:show_n]:
                with st.expander(f"📷 {ann['image_name']}  [{len(ann['detections'])} detections]"):
                    for det in ann["detections"]:
                        st.json(det)
        else:
            st.info("No detections yet. Run inference above.")
    else:
        st.info("No inference results yet.")
except Exception as e:
    st.warning(f"Could not load detections: {e}")
