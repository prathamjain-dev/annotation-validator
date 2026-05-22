"""
Adala Agent API endpoints
Runs LLM-powered annotation validation and reasoning.
"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.core.database import get_db, Annotation, OnnxModel
from backend.core.adala_agent import AdalaAnnotationAgent
from backend.core.schemas import VERDICT_SCHEMAS

router = APIRouter()
logger = logging.getLogger(__name__)


class AdalaRunRequest(BaseModel):
    dataset_name: str
    model_name: str                        # To fetch class list
    llm_provider: str = "openai"           # "openai" | "anthropic" | "vllm" | "openrouter"
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None         # For vLLM
    site_url: Optional[str] = None         # For OpenRouter HTTP-Referer
    image_context: Optional[str] = None
    validation_stage: str = "text_llm"     # "text_llm" | "vlm"
    json_mode: bool = True                 # Use structured JSON output
    max_retries: int = 2                   # Retries on malformed output


class AnnotationUpdateRequest(BaseModel):
    detections: list


@router.post("/validate")
async def run_adala_validation(req: AdalaRunRequest, db: AsyncSession = Depends(get_db)):
    """Run Adala agent validation on all detected annotations in a dataset."""

    # Get class list from model
    model_res = await db.execute(select(OnnxModel).where(OnnxModel.name == req.model_name))
    model_rec = model_res.scalar_one_or_none()
    if not model_rec:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found")

    classes = model_rec.classes or []

    # Resolve schema + prompt for requested stage
    if req.validation_stage not in VERDICT_SCHEMAS:
        raise HTTPException(status_code=400, detail=f"Unknown validation stage: {req.validation_stage}. Choose from: {list(VERDICT_SCHEMAS.keys())}")
    schema, system_prompt = VERDICT_SCHEMAS[req.validation_stage]

    # Build LLM config
    llm_config = {
        "provider": req.llm_provider,
        "json_mode": req.json_mode,
    }
    if req.api_key:
        llm_config["api_key"] = req.api_key
    if req.model:
        llm_config["model"] = req.model
    if req.base_url:
        llm_config["base_url"] = req.base_url
    if req.site_url:
        llm_config["site_url"] = req.site_url

    # Build agent with schema-driven validation
    try:
        agent = AdalaAnnotationAgent(
            llm_config,
            schema=schema,
            system_prompt=system_prompt,
            max_retries=req.max_retries,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build LLM agent: {e}")

    # Get all annotations for dataset
    result = await db.execute(
        select(Annotation).where(Annotation.dataset_name == req.dataset_name)
    )
    annotations = result.scalars().all()

    if not annotations:
        raise HTTPException(status_code=404, detail="No annotations found. Run inference first.")

    summary = {"total": len(annotations), "processed": 0, "approved": 0,
               "rejected": 0, "relabeled": 0, "flagged": 0, "errors": []}

    for ann in annotations:
        if not ann.detections:
            continue
        try:
            validated = await agent.validate_all(
                image_name=ann.image_name,
                detections=ann.detections,
                all_classes=classes,
                image_context=req.image_context,
            )
            ann.detections = validated
            ann.status = "validated"
            summary["processed"] += 1

            for det in validated:
                s = det.get("adala_status", "pending")
                if s in summary:
                    summary[s] += 1

        except Exception as e:
            logger.error(f"Adala failed on {ann.image_name}: {e}")
            summary["errors"].append({"image": ann.image_name, "error": str(e)})

    await db.commit()
    return summary


@router.get("/{dataset_name}/results")
async def get_adala_results(dataset_name: str, db: AsyncSession = Depends(get_db)):
    """Retrieve Adala validation results for a dataset."""
    result = await db.execute(
        select(Annotation).where(Annotation.dataset_name == dataset_name)
    )
    annotations = result.scalars().all()
    return [
        {
            "image_name": a.image_name,
            "detections": a.detections,
            "status": a.status,
        }
        for a in annotations
    ]


@router.put("/{dataset_name}/{image_name}")
async def update_annotation(
    dataset_name: str,
    image_name: str,
    body: AnnotationUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update detections for a single image (human review edits)."""
    result = await db.execute(
        select(Annotation).where(
            Annotation.dataset_name == dataset_name,
            Annotation.image_name == image_name,
        )
    )
    ann = result.scalar_one_or_none()
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")

    ann.detections = body.detections
    ann.status = "reviewed"
    await db.commit()
    return {"status": "updated", "image_name": image_name}
