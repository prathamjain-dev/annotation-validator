"""
Page 4: Adala Agent Validation
Run LLM-powered annotation reasoning and validation.
"""

import streamlit as st
import requests

API = "http://localhost:8000/api"

st.set_page_config(page_title="Adala Validation", page_icon="🤖", layout="wide")

st.markdown("# 🤖 Adala Agent Validation")
st.markdown("""
The Adala agent uses an LLM to **validate, reason about, and correct** each detection:
- **Approve** ✅ — detection looks correct
- **Reject** ❌ — likely false positive, will be excluded from export
- **Relabel** 🔄 — correct object but wrong class
- **Flag** 🚩 — uncertain, needs human review
""")
st.divider()

# ── Load Datasets & Models ─────────────────────────────────────────────────────
datasets, models = [], []
try:
    datasets = requests.get(f"{API}/datasets/list", timeout=5).json()
    models = requests.get(f"{API}/models/list", timeout=5).json()
except Exception as e:
    st.error(f"Backend not reachable: {e}")

if not datasets or not models:
    st.warning("Upload a dataset and model first, then run inference.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    dataset_name = st.selectbox("Dataset", [d["name"] for d in datasets])
with col2:
    model_name = st.selectbox("Model (for class list)", [m["name"] for m in models])

# ── LLM Provider Config ────────────────────────────────────────────────────────
st.markdown("### LLM Provider Configuration")
provider = st.selectbox("Provider", ["openrouter", "openai", "anthropic", "vllm"])

api_key, model_id, base_url = None, None, None

# Popular OpenRouter models (free ones first)
OPENROUTER_MODELS = [
    # Free
    "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    # Paid — strong
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "mistralai/mistral-large",
    "google/gemini-pro-1.5",
    "google/gemini-flash-2.5",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "cohere/command-r-plus",
    "perplexity/llama-3.1-sonar-large-128k-online",
    "google/gemini-2.0-flash-lite-001"
]

if provider == "openrouter":
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-...",
        help="Get your key at https://openrouter.ai/keys",
    )
    col_or1, col_or2 = st.columns([3, 1])
    with col_or1:
        model_preset = st.selectbox("Select Model", OPENROUTER_MODELS,
                                    help="Models marked :free have no token cost")
    with col_or2:
        custom_model = st.text_input("Or type custom model ID", placeholder="org/model-name")
    model_id = custom_model.strip() if custom_model.strip() else model_preset
    st.caption(f"🔗 Browse all models: https://openrouter.ai/models  |  Selected: `{model_id}`")

elif provider == "openai":
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model_id = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
elif provider == "anthropic":
    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    model_id = st.selectbox("Model", ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"])
elif provider == "vllm":
    base_url = st.text_input("vLLM Base URL", value="http://localhost:8001")
    model_id = st.text_input("Model Name", value="llama3")

st.divider()

# ── Validation Stage ────────────────────────────────────────────────────────────
st.markdown("### Validation Stage")
val_stage = st.selectbox(
    "Stage",
    ["text_llm", "vlm"],
    index=0,
    help="text_llm = metadata-only (cheap), vlm = vision-based (uses image content, more accurate but expensive)",
)
use_json_mode = st.checkbox(
    "Structured JSON output (recommended)",
    value=True,
    help="Enables response_format=json_object for guaranteed valid JSON. Disable if model doesn't support it.",
)
max_retries = st.slider("Max retries on failure", 0, 5, 2, help="Number of retries if LLM returns invalid output")

image_context = st.text_input(
    "Optional Image Context",
    placeholder="e.g. Construction site with workers wearing safety equipment",
    help="Providing scene context improves agent accuracy"
)

st.divider()

# ── Run Validation ─────────────────────────────────────────────────────────────
if st.button("🚀 Run Adala Validation", type="primary", use_container_width=True):
    if provider in ("openai", "anthropic", "openrouter") and not api_key:
        st.error("Please enter your API key.")
    elif provider == "vllm" and not base_url:
        st.error("Please enter the vLLM base URL.")
    else:
        with st.spinner("Running Adala agents... (this may take a while for large datasets)"):
            payload = {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "llm_provider": provider,
                "api_key": api_key,
                "model": model_id,
                "base_url": base_url,
                "image_context": image_context or None,
                "validation_stage": val_stage,
                "json_mode": use_json_mode,
                "max_retries": max_retries,
            }
            try:
                resp = requests.post(f"{API}/adala/validate", json=payload, timeout=900)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success("✅ Adala validation complete!")

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Processed", data.get("processed", 0))
                    col2.metric("✅ Approved", data.get("approved", 0))
                    col3.metric("❌ Rejected", data.get("rejected", 0))
                    col4.metric("🔄 Relabeled", data.get("relabeled", 0))
                    col5.metric("🚩 Flagged", data.get("flagged", 0))

                    if data.get("errors"):
                        with st.expander(f"⚠️ {len(data['errors'])} errors"):
                            for err in data["errors"]:
                                st.error(f"{err['image']}: {err['error']}")

                    st.info("➡️ Review annotations in the **Annotation Review** page.")
                else:
                    st.error(f"Validation failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Preview Results ────────────────────────────────────────────────────────────
st.divider()
st.markdown("### Validation Results Preview")

try:
    resp = requests.get(f"{API}/adala/{dataset_name}/results", timeout=5)
    if resp.status_code == 200:
        results = resp.json()
        validated = [r for r in results if r["status"] == "validated"]
        if not validated:
            st.info("No validated annotations yet. Run Adala validation above.")
        else:
            STATUS_COLORS = {
                "approved": "🟢", "rejected": "🔴",
                "relabeled": "🟠", "flagged": "🔵", "pending": "⚪"
            }
            for ann in validated[:10]:
                with st.expander(f"📷 {ann['image_name']}"):
                    for det in ann["detections"]:
                        status = det.get("adala_status", "pending")
                        icon = STATUS_COLORS.get(status, "⚪")
                        st.markdown(f"""
                        {icon} **{det.get('class', '?')}**  conf: `{det.get('confidence', 0):.3f}`
                        > *{det.get('adala_reasoning', 'No reasoning')}*
                        """)
except Exception as e:
    st.warning(f"Could not load results: {e}")
