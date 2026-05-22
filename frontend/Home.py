"""
Adala Auto Annotator - Streamlit Frontend
Main entry point that shows navigation and landing page.
"""

import streamlit as st

st.set_page_config(
    page_title="Adala Auto Annotator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #00FF88;
    text-shadow: 0 0 20px rgba(0,255,136,0.3);
    letter-spacing: -1px;
}

.subtitle {
    color: #888;
    font-size: 1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.pipeline-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #00FF88;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #00FF88;
}

.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}

[data-testid="stSidebar"] {
    background: #0d0d1a !important;
}
</style>
""", unsafe_allow_html=True)

# ── Landing Page ───────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">🎯 ADALA AUTO ANNOTATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered annotation with LLM reasoning agents</div>', unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### Pipeline Overview")
    pipeline_steps = [
        ("📁", "Upload Dataset", "Images, videos, or zip files"),
        ("🧠", "Upload ONNX Model", "Custom detection model + classes"),
        ("🔍", "Auto Detection", "Run ONNX inference on all images"),
        ("🤖", "Adala Validation", "LLM agents validate & reason about detections"),
        ("👁️", "Human Review", "Edit, approve, or reject annotations"),
        ("📦", "YOLO Export", "Download labels in YOLO format"),
    ]
    for icon, step, desc in pipeline_steps:
        st.markdown(f"""
        <div class="pipeline-box">
            {icon} <strong>{step}</strong><br>
            <span style="color:#aaa; font-size:0.78rem;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### Quick Start")
    st.info("""
    1. Start the backend:
    ```
    uvicorn main:app --reload
    ```
    
    2. Navigate using the **sidebar** to access each workflow step.
    
    3. Configure your LLM provider (OpenAI / Claude / vLLM) on the Adala page.
    """)

    st.markdown("### Backend Status")
    import requests
    try:
        r = requests.get("http://localhost:8000/", timeout=2)
        if r.status_code == 200:
            st.success("✅ Backend API is running")
        else:
            st.warning("⚠️ Backend returned unexpected status")
    except Exception:
        st.error("❌ Backend not reachable. Start with: `uvicorn main:app --reload`")

    st.markdown("### Adala Agent Statuses")
    statuses = [
        ("✅", "approved", "#00C875"),
        ("❌", "rejected", "#E2445C"),
        ("🔄", "relabeled", "#FDAB3D"),
        ("🚩", "flagged", "#579BFC"),
    ]
    for icon, label, color in statuses:
        st.markdown(f'<span style="color:{color}; font-weight:600;">{icon} {label.upper()}</span>', unsafe_allow_html=True)
