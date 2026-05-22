"""
Page 1: Dataset Upload
Upload images, videos, or zip files as datasets.
"""

import streamlit as st
import requests
from pathlib import Path

API = "http://localhost:8000/api"

st.set_page_config(page_title="Dataset Upload", page_icon="📁", layout="wide")

st.markdown("# 📁 Dataset Upload")
st.markdown("Upload images, videos, or a zip archive to create an annotatable dataset.")
st.divider()

# ── Upload Form ────────────────────────────────────────────────────────────────
with st.form("dataset_upload_form"):
    dataset_name = st.text_input("Dataset Name", placeholder="e.g. construction_site_v1")
    files = st.file_uploader(
        "Upload Files",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "zip"],
        help="Images, videos (frames will be extracted), or zip archives",
    )
    submitted = st.form_submit_button("⬆️ Upload Dataset", type="primary")

if submitted:
    if not dataset_name:
        st.error("Please enter a dataset name.")
    elif not files:
        st.error("Please select at least one file.")
    else:
        with st.spinner("Uploading and processing files..."):
            multipart = [("files", (f.name, f.getvalue(), f.type)) for f in files]
            try:
                resp = requests.post(
                    f"{API}/datasets/upload",
                    data={"name": dataset_name},
                    files=multipart,
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"✅ Dataset **{data['dataset']}** uploaded successfully!")
                    st.metric("Files processed", data["count"])
                    with st.expander("View file list"):
                        for f in data["files"][:50]:
                            st.text(f"• {f}")
                        if len(data["files"]) > 50:
                            st.text(f"... and {len(data['files']) - 50} more")
                else:
                    st.error(f"Upload failed: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

st.divider()

# ── Existing Datasets ──────────────────────────────────────────────────────────
st.markdown("### Existing Datasets")

try:
    resp = requests.get(f"{API}/datasets/list", timeout=5)
    if resp.status_code == 200:
        datasets = resp.json()
        if not datasets:
            st.info("No datasets yet. Upload your first dataset above.")
        else:
            for ds in datasets:
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    st.markdown(f"**{ds['name']}**")
                with col2:
                    st.markdown(f"`{ds['file_count']} files`")
                with col3:
                    st.markdown(f"<small>{ds['created_at'][:19]}</small>", unsafe_allow_html=True)
    else:
        st.warning("Could not load datasets list")
except Exception as e:
    st.error(f"Backend not reachable: {e}")
