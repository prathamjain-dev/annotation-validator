"""
Dataset API endpoints
Handles upload and management of image/video datasets.
"""

import os
import zipfile
import shutil
import cv2
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db, Dataset

router = APIRouter()

UPLOAD_DIR = "data/uploads"
ALLOWED_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEOS = {".mp4", ".avi", ".mov", ".mkv"}


def extract_video_frames(video_path: str, output_dir: str, fps: int = 1) -> List[str]:
    """Extract frames from video at given fps using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(video_fps / fps))
    frames = []
    frame_idx = 0
    saved = 0

    os.makedirs(output_dir, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            fname = f"frame_{saved:05d}.jpg"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, frame)
            frames.append(fname)
            saved += 1
        frame_idx += 1

    cap.release()
    return frames


@router.post("/upload")
async def upload_dataset(
    name: str = Form(...),
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload images, videos, or a zip file as a dataset."""
    dataset_dir = os.path.join(UPLOAD_DIR, name)
    os.makedirs(dataset_dir, exist_ok=True)

    image_files = []

    for upload in files:
        ext = Path(upload.filename).suffix.lower()
        dest = os.path.join(dataset_dir, upload.filename)

        # Save file
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)

        if ext in ALLOWED_IMAGES:
            image_files.append(upload.filename)

        elif ext in ALLOWED_VIDEOS:
            frames_dir = os.path.join(dataset_dir, "frames_" + Path(upload.filename).stem)
            frames = extract_video_frames(dest, frames_dir)
            # Move frames up to dataset_dir
            for fname in frames:
                src = os.path.join(frames_dir, fname)
                dst = os.path.join(dataset_dir, fname)
                shutil.move(src, dst)
                image_files.append(fname)
            shutil.rmtree(frames_dir, ignore_errors=True)
            os.remove(dest)  # Remove original video after frame extraction

        elif ext == ".zip":
            with zipfile.ZipFile(dest, "r") as zf:
                for member in zf.namelist():
                    m_ext = Path(member).suffix.lower()
                    if m_ext in ALLOWED_IMAGES:
                        data = zf.read(member)
                        dest_path = os.path.join(dataset_dir, os.path.basename(member))
                        with open(dest_path, "wb") as f:
                            f.write(data)
                        image_files.append(os.path.basename(member))
            os.remove(dest)

    # Persist to DB (upsert by name)
    existing = await db.execute(select(Dataset).where(Dataset.name == name))
    existing = existing.scalar_one_or_none()

    if existing:
        existing.file_count = len(image_files)
        existing.path = dataset_dir
    else:
        db.add(Dataset(name=name, path=dataset_dir, file_count=len(image_files)))

    await db.commit()

    return {"dataset": name, "path": dataset_dir, "files": image_files, "count": len(image_files)}


@router.get("/list")
async def list_datasets(db: AsyncSession = Depends(get_db)):
    """List all datasets."""
    result = await db.execute(select(Dataset))
    datasets = result.scalars().all()
    return [
        {"id": d.id, "name": d.name, "path": d.path,
         "file_count": d.file_count, "created_at": str(d.created_at)}
        for d in datasets
    ]


@router.get("/{name}/files")
async def list_dataset_files(name: str):
    """List image files in a dataset."""
    dataset_dir = os.path.join(UPLOAD_DIR, name)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=404, detail="Dataset not found")

    files = [
        f for f in os.listdir(dataset_dir)
        if Path(f).suffix.lower() in ALLOWED_IMAGES
    ]
    return {"dataset": name, "files": sorted(files), "count": len(files)}
