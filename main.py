"""
Adala Annotator - FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from backend.api import datasets, models, inference, adala, export
from backend.core.database import init_db

app = FastAPI(
    title="Adala Auto Annotator API",
    description="AI-powered automatic annotation platform with Adala agents",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving uploaded images
os.makedirs("data/uploads", exist_ok=True)
app.mount("/files", StaticFiles(directory="data/uploads"), name="files")

# Include routers
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(adala.router, prefix="/api/adala", tags=["adala"])
app.include_router(export.router, prefix="/api/export", tags=["export"])


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/")
def root():
    return {"status": "Adala Annotator API running", "version": "0.1.0"}
