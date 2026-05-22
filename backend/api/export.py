"""
Export API endpoints
Export annotations in YOLO format.
"""

import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db, Annotation, OnnxModel, Dataset
from backend.core.exporter import export_yolo

router = APIRouter()

EXPORT_DIR = "data/exports"
UPLOAD_DIR = "data/uploads"
os.makedirs(EXPORT_DIR, exist_ok=True)


class ExportRequest(BaseModel):
    dataset_name: str
    model_name: str


@router.post("/yolo")
async def export_yolo_dataset(req: ExportRequest, db: AsyncSession = Depends(get_db)):
    """Export annotations as YOLO dataset zip."""

    # Get classes from model
    model_res = await db.execute(select(OnnxModel).where(OnnxModel.name == req.model_name))
    model_rec = model_res.scalar_one_or_none()
    if not model_rec:
        raise HTTPException(status_code=404, detail="Model not found")

    classes = model_rec.classes or []

    # Get annotations
    result = await db.execute(
        select(Annotation).where(Annotation.dataset_name == req.dataset_name)
    )
    annotations = result.scalars().all()

    if not annotations:
        raise HTTPException(status_code=404, detail="No annotations found")

    # Build export list with image paths
    dataset_dir = os.path.join(UPLOAD_DIR, req.dataset_name)
    export_annotations = [
        {
            "image_name": a.image_name,
            "image_path": os.path.join(dataset_dir, a.image_name),
            "detections": a.detections or [],
        }
        for a in annotations
    ]

    zip_path = export_yolo(
        annotations=export_annotations,
        classes=classes,
        export_dir=EXPORT_DIR,
        dataset_name=req.dataset_name,
    )

    return FileResponse(
        path=zip_path,
        filename=os.path.basename(zip_path),
        media_type="application/zip",
    )
