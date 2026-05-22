# 🎯 Adala Auto Annotator

AI-powered automatic annotation platform integrating Adala-style LLM agents for intelligent bounding box validation and correction.

---

## Architecture

```
Images/Videos
      ↓
Custom ONNX Detector
      ↓
Raw Bounding Boxes
      ↓
Adala Annotation Agent (LLM)
      ↓
Annotation Validation / Correction
      ↓
Human Review UI (Streamlit)
      ↓
YOLO Export
```

---

## Project Structure

```
adala-annotator/
├── main.py                          # FastAPI entry point
├── requirements.txt
├── setup.sh                         # Quick setup script
├── README.md
│
├── backend/
│   ├── api/
│   │   ├── datasets.py              # Dataset upload/management
│   │   ├── models.py                # ONNX model upload/inspection
│   │   ├── inference.py             # ONNX inference runner
│   │   ├── adala.py                 # Adala agent validation
│   │   └── export.py                # YOLO export
│   │
│   └── core/
│       ├── database.py              # SQLite + SQLAlchemy models
│       ├── onnx_engine.py           # Generic ONNX inference wrapper
│       ├── adala_agent.py           # LLM agent integration
│       ├── visualizer.py            # Annotation drawing utilities
│       └── exporter.py              # YOLO format exporter
│
├── frontend/
│   ├── Home.py                      # Landing page
│   └── pages/
│       ├── 1_Dataset_Upload.py
│       ├── 2_Model_Upload.py
│       ├── 3_Auto_Annotation.py
│       ├── 4_Adala_Validation.py
│       ├── 5_Annotation_Review.py
│       └── 6_Export.py
│
└── data/
    ├── uploads/                     # Uploaded images
    ├── models/                      # ONNX models
    ├── annotations/                 # (reserved)
    └── exports/                     # YOLO exports
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the setup script:

```bash
chmod +x setup.sh && ./setup.sh
```

### 2. Start the Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### 3. Start the Frontend

```bash
streamlit run frontend/Home.py
```

Opens at: http://localhost:8501

---

## Workflow

### Step 1: Upload Dataset
- Upload images (jpg, png, bmp, webp), videos (mp4, avi, mov), or zip files
- Videos are automatically split into frames using OpenCV

### Step 2: Upload ONNX Model
- Upload `model.onnx` + optional `classes.yaml` or `classes.txt`
- Model is inspected for input/output shapes automatically
- Supports YOLO-style ONNX exports

### Step 3: Run Auto Detection
- Select dataset + model
- Configure confidence and IoU thresholds
- Run ONNX inference — bounding boxes saved to SQLite

### Step 4: Adala Validation
Configure your LLM provider:

**OpenRouter (recommended — 200+ models, free tier available):**
```
Provider: openrouter
API Key: sk-or-...   (get at https://openrouter.ai/keys)
Model: meta-llama/llama-3.1-8b-instruct:free   (free)
       mistralai/mistral-large                   (paid)
       anthropic/claude-3.5-sonnet               (paid)
       google/gemini-flash-1.5                   (paid)
       openai/gpt-4o                             (paid)
```

**OpenAI:**
```
Provider: openai
API Key: sk-...
Model: gpt-4o-mini
```

**Anthropic Claude:**
```
Provider: anthropic
API Key: sk-ant-...
Model: claude-sonnet-4-20250514
```

**Local vLLM / Ollama:**
```
Provider: vllm
Base URL: http://localhost:8001
Model: llama3
```

The agent reviews each detection and returns:
- `approved` — detection is correct
- `rejected` — false positive, excluded from export
- `relabeled` — wrong class, corrected automatically
- `flagged` — uncertain, needs human review

### Step 5: Review Annotations
- View annotated images with color-coded status boxes
- Edit class labels and statuses manually
- Save changes back to the database

### Step 6: Export YOLO
- Downloads a zip containing:
  - `images/` — all dataset images
  - `labels/` — YOLO format `.txt` files (rejected detections excluded)
  - `dataset.yaml` — class definitions for training

---

## ONNX Model Support

The ONNX inference engine supports:
- **YOLO-style outputs**: `[1, N, 5+num_classes]` or `[1, 5+nc, N]`
- Dynamic input shapes (defaults to 640×640)
- CPU and CUDA providers (auto-detected)
- Custom postprocessors via `ONNXDetector.register_postprocessor(name, fn)`

### Export YOLO model for compatibility

```bash
# YOLOv8 export
yolo export model=yolov8n.pt format=onnx imgsz=640

# YOLOv5 export
python export.py --weights yolov5s.pt --include onnx
```

---

## Adala Agent Prompt

The agent system prompt instructs the LLM to validate detections using the following schema:

```json
{
  "status": "approved | rejected | relabeled | flagged",
  "reasoning": "Brief explanation of the decision",
  "suggested_class": "corrected class name or null",
  "confidence_adjustment": 0.0
}
```

You can customize the prompt in `backend/core/adala_agent.py` → `SYSTEM_PROMPT`.

---

## Internal Annotation Format

```json
{
  "image_name": "frame_00001.jpg",
  "detections": [
    {
      "class": "helmet",
      "class_id": 0,
      "confidence": 0.91,
      "bbox": [100, 50, 200, 180],
      "adala_status": "approved",
      "adala_reasoning": "Helmet clearly visible on worker's head",
      "adala_suggested_class": null,
      "adala_confidence_adjustment": 0.0
    }
  ]
}
```

---

## Extending the System

### Add a custom postprocessor

```python
from backend.core.onnx_engine import ONNXDetector

def my_postprocessor(outputs, ratio, pad, orig_size, **kwargs):
    # Parse your custom model output
    return detections  # List of detection dicts

ONNXDetector.register_postprocessor("my_model", my_postprocessor)
```

### Add a new LLM provider

```python
from backend.core.adala_agent import LLMProvider

class MyProvider(LLMProvider):
    async def complete(self, system: str, user: str) -> str:
        # Call your LLM API
        return response_text
```

---

## Future Extensions (Planned)

- [ ] CVAT integration for collaborative annotation
- [ ] Active learning loop (sample uncertain images)
- [ ] FiftyOne visualization
- [ ] VLM models (GPT-4V, Claude Vision) for image-aware reasoning
- [ ] Multi-user workflows with auth
- [ ] Distributed inference with Celery + Redis

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| Inference | ONNX Runtime + OpenCV |
| AI Agent | Custom Adala-style LLM agent |
| LLM Providers | OpenRouter, OpenAI, Anthropic, vLLM |
| Database | SQLite (aiosqlite + SQLAlchemy) |
| Storage | Local filesystem |
