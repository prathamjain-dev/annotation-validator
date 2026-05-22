"""
Inference API endpoints
Runs ONNX object detection on dataset images.
"""

import os
import json
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db, OnnxModel, Dataset, Annotation
from backend.core.onnx_engine import ONNXDetector

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "data/uploads"
ALLOWED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class InferenceRequest(BaseModel):
    dataset_name: str
    model_name: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    postprocessor: str = "yolo"  # "yolo" or custom


@router.post("/run")
async def run_inference(req: InferenceRequest, db: AsyncSession = Depends(get_db)):
    """Run ONNX inference on all images in a dataset."""

    # Load model from DB
    model_res = await db.execute(select(OnnxModel).where(OnnxModel.name == req.model_name))
    model_rec = model_res.scalar_one_or_none()
    if not model_rec:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found")

    # Load dataset
    dataset_dir = os.path.join(UPLOAD_DIR, req.dataset_name)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_name}' not found")

    # Load detector
    try:
        detector = ONNXDetector(model_rec.path, classes=model_rec.classes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load ONNX model: {e}")

    image_files = [
        f for f in sorted(os.listdir(dataset_dir))
        if Path(f).suffix.lower() in ALLOWED_IMAGES
    ]

    results = []
    errors = []

    for img_name in image_files:
        img_path = os.path.join(dataset_dir, img_name)
        try:
            detections = detector.run(
                img_path,
                conf_threshold=req.conf_threshold,
                iou_threshold=req.iou_threshold,
                postprocessor=req.postprocessor,
            )

            # Upsert annotation record in DB
            ann_res = await db.execute(
                select(Annotation).where(
                    Annotation.dataset_name == req.dataset_name,
                    Annotation.image_name == img_name,
                )
            )
            ann = ann_res.scalar_one_or_none()

            if ann:
                ann.detections = detections
                ann.status = "detected"
            else:
                ann = Annotation(
                    dataset_name=req.dataset_name,
                    image_name=img_name,
                    detections=detections,
                    status="detected",
                )
                db.add(ann)

            results.append({
                "image": img_name,
                "detections": len(detections),
                "status": "ok",
            })
        except Exception as e:
            logger.error(f"Inference error on {img_name}: {e}")
            errors.append({"image": img_name, "error": str(e)})

    await db.commit()

    return {
        "dataset": req.dataset_name,
        "model": req.model_name,
        "total_images": len(image_files),
        "processed": len(results),
        "errors": len(errors),
        "results": results,
        "error_details": errors,
    }


@router.get("/{dataset_name}/detections")
async def get_detections(dataset_name: str, db: AsyncSession = Depends(get_db)):
    """Retrieve all detections for a dataset."""
    result = await db.execute(
        select(Annotation).where(Annotation.dataset_name == dataset_name)
    )
    annotations = result.scalars().all()
    return [
        {
            "id": a.id,
            "image_name": a.image_name,
            "detections": a.detections,
            "status": a.status,
            "updated_at": str(a.updated_at),
        }
        for a in annotations
    ]


@router.get("/{dataset_name}/{image_name}/detections")
async def get_image_detections(
    dataset_name: str, image_name: str, db: AsyncSession = Depends(get_db)
):
    """Get detections for a specific image."""
    result = await db.execute(
        select(Annotation).where(
            Annotation.dataset_name == dataset_name,
            Annotation.image_name == image_name,
        )
    )
    ann = result.scalar_one_or_none()
    if not ann:
        raise HTTPException(status_code=404, detail="No annotations found for this image")
    return {
        "image_name": ann.image_name,
        "detections": ann.detections,
        "status": ann.status,
    }
