"""
ONNX Model API endpoints
Handles upload and inspection of ONNX detection models.
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db, OnnxModel
from backend.core.onnx_engine import inspect_onnx_model

router = APIRouter()

MODELS_DIR = "data/models"
os.makedirs(MODELS_DIR, exist_ok=True)


def parse_classes_file(path: str) -> List[str]:
    """Parse classes from .yaml or .txt file."""
    ext = Path(path).suffix.lower()
    if ext in (".yaml", ".yml"):
        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            names = data.get("names", [])
            if isinstance(names, dict):
                return [names[k] for k in sorted(names.keys())]
            return list(names)
        elif isinstance(data, list):
            return data
    elif ext == ".txt":
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    return []


@router.post("/upload")
async def upload_model(
    name: str = Form(...),
    model_file: UploadFile = File(...),
    classes_file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    """Upload ONNX model and optional classes file."""
    model_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    # Save ONNX model
    model_path = os.path.join(model_dir, "model.onnx")
    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)

    # Save classes file
    classes_path = None
    classes = []
    if classes_file and classes_file.filename:
        ext = Path(classes_file.filename).suffix.lower()
        classes_path = os.path.join(model_dir, f"classes{ext}")
        with open(classes_path, "wb") as f:
            shutil.copyfileobj(classes_file.file, f)
        classes = parse_classes_file(classes_path)

    # Inspect ONNX
    try:
        meta = inspect_onnx_model(model_path)
        input_shape = meta["inputs"][0]["shape"] if meta["inputs"] else []
        output_shape = meta["outputs"][0]["shape"] if meta["outputs"] else []
    except Exception as e:
        input_shape = []
        output_shape = []

    # Persist to DB
    existing = await db.execute(select(OnnxModel).where(OnnxModel.name == name))
    existing = existing.scalar_one_or_none()

    if existing:
        existing.path = model_path
        existing.classes_path = classes_path
        existing.input_shape = input_shape
        existing.output_shape = output_shape
        existing.classes = classes
    else:
        db.add(OnnxModel(
            name=name,
            path=model_path,
            classes_path=classes_path,
            input_shape=input_shape,
            output_shape=output_shape,
            classes=classes,
        ))
    await db.commit()

    return {
        "name": name,
        "model_path": model_path,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "classes": classes,
        "num_classes": len(classes),
    }


@router.get("/list")
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all uploaded ONNX models."""
    result = await db.execute(select(OnnxModel))
    models = result.scalars().all()
    return [
        {
            "id": m.id, "name": m.name, "path": m.path,
            "input_shape": m.input_shape, "output_shape": m.output_shape,
            "classes": m.classes, "num_classes": len(m.classes or []),
            "created_at": str(m.created_at),
        }
        for m in models
    ]


@router.get("/{name}/inspect")
async def inspect_model(name: str, db: AsyncSession = Depends(get_db)):
    """Detailed ONNX model inspection."""
    result = await db.execute(select(OnnxModel).where(OnnxModel.name == name))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    meta = inspect_onnx_model(model.path)
    return {
        "name": name,
        "metadata": meta,
        "classes": model.classes,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
    }
