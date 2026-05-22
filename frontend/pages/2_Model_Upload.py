"""
Page 2: ONNX Model Upload
Upload detection models and inspect their metadata.
"""

import streamlit as st
import requests

API = "http://localhost:8000/api"

st.set_page_config(page_title="Model Upload", page_icon="🧠", layout="wide")

st.markdown("# 🧠 ONNX Model Upload")
st.markdown("Upload a custom ONNX object detection model and its class definitions.")
st.divider()

# ── Upload Form ────────────────────────────────────────────────────────────────
with st.form("model_upload_form"):
    model_name = st.text_input("Model Name", placeholder="e.g. yolov8_helmet_detector")
    model_file = st.file_uploader("ONNX Model File (.onnx)", type=["onnx"])
    classes_file = st.file_uploader(
        "Classes File (optional)",
        type=["yaml", "yml", "txt"],
        help="classes.yaml (YOLO format) or classes.txt (one class per line)",
    )
    submitted = st.form_submit_button("⬆️ Upload Model", type="primary")

if submitted:
    if not model_name:
        st.error("Please enter a model name.")
    elif not model_file:
        st.error("Please upload an ONNX model file.")
    else:
        with st.spinner("Uploading and inspecting model..."):
            files = [("model_file", (model_file.name, model_file.getvalue(), "application/octet-stream"))]
            if classes_file:
                files.append(("classes_file", (classes_file.name, classes_file.getvalue(), "text/plain")))
            try:
                resp = requests.post(
                    f"{API}/models/upload",
                    data={"name": model_name},
                    files=files,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ Model **{data['name']}** uploaded!")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Input Shape", str(data.get("input_shape", "?")))
                    col2.metric("Num Classes", data.get("num_classes", 0))
                    col3.metric("Output Shape", str(data.get("output_shape", "?")))

                    if data.get("classes"):
                        with st.expander("📋 Detected Classes"):
                            for i, cls in enumerate(data["classes"]):
                                st.text(f"  [{i}] {cls}")
                else:
                    st.error(f"Upload failed: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

st.divider()

# ── Existing Models ────────────────────────────────────────────────────────────
st.markdown("### Uploaded Models")

try:
    resp = requests.get(f"{API}/models/list", timeout=5)
    if resp.status_code == 200:
        models = resp.json()
        if not models:
            st.info("No models uploaded yet.")
        else:
            for m in models:
                with st.expander(f"🧠 {m['name']}  —  {m['num_classes']} classes"):
                    col1, col2 = st.columns(2)
                    col1.markdown(f"**Input:** `{m.get('input_shape', 'unknown')}`")
                    col2.markdown(f"**Output:** `{m.get('output_shape', 'unknown')}`")
                    if m.get("classes"):
                        st.markdown("**Classes:** " + ", ".join(f"`{c}`" for c in m["classes"][:20]))
    else:
        st.warning("Could not load models list")
except Exception as e:
    st.error(f"Backend not reachable: {e}")
