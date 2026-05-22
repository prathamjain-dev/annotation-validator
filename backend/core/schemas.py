"""
Schema definitions for all validation stages.
Each schema is a JSON Schema dict + bundled system prompt.
Single source of truth for LLM contracts.
"""

import json
from typing import Tuple

# ── Text-only LLM verdict ──────────────────────────────────────────────────────

TEXT_LLM_VERDICT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["approved", "rejected", "flagged"],
            "description": "approved=correct, rejected=false positive, flagged=uncertain",
        },
        "reasoning": {
            "type": "string",
            "maxLength": 500,
            "description": "Brief explanation for the decision",
        },
        "confidence_adjustment": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "Suggested confidence delta (-1 to +1), 0 if no change",
        },
    },
    "required": ["status", "reasoning", "confidence_adjustment"],
    "additionalProperties": False,
}


TEXT_LLM_SYSTEM_PROMPT = """You are an expert computer vision annotation validator (Adala Agent).
Your job is to review bounding box detections based on their metadata and determine if they are correct.

For each detection you receive, assess:
1. Is the label semantically plausible for this type of scene?
2. Is the confidence score reasonable given the context?
3. Is the bounding box geometry (size, position, aspect ratio) reasonable for this class?
4. Are there any contradictory detections nearby?

You MUST respond with a JSON object matching this schema exactly:
{schema}

Status meanings:
- "approved": detection looks correct
- "rejected": clear false positive or wrong label
- "flagged": uncertain, needs human review

Rules:
- Respond with ONLY the JSON object. No markdown, no backticks, no code fences, no extra text.
- Begin with {{ and end with }}."""


# ── VLM Vision verdict ─────────────────────────────────────────────────────────

VLM_VERDICT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["approved", "rejected", "relabeled", "flagged"],
            "description": "approved=correct, rejected=false positive, relabeled=wrong class, flagged=uncertain",
        },
        "reasoning": {
            "type": "string",
            "maxLength": 500,
            "description": "Brief explanation for the decision",
        },
        "suggested_class": {
            "type": "string",
            "description": "If relabeled, the corrected class name. Null otherwise.",
        },
        "confidence_adjustment": {
            "type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
            "description": "Suggested confidence delta (-1 to +1), 0 if no change",
        },
    },
    "required": ["status", "reasoning", "confidence_adjustment"],
    "additionalProperties": False,
}


VLM_SYSTEM_PROMPT = """You are an expert computer vision annotation validator with vision capabilities (Adala Agent).
Your job is to examine the image inside a bounding box and determine if the detection is correct.

For each detection:
1. Look at the image content inside the bounding box
2. Does the box actually contain the predicted object?
3. Is the bounding box accurately placed (not too loose, not too tight)?
4. If the wrong class, what should it be?

You MUST respond with a JSON object matching this schema exactly:
{schema}

Status meanings:
- "approved": bbox correctly contains the predicted object
- "rejected": bbox does not contain the predicted object (false positive)
- "relabeled": object is real but the class label is wrong — provide suggested_class
- "flagged": uncertain, needs human review

Rules:
- Respond with ONLY the JSON object. No markdown, no backticks, no code fences, no extra text.
- Begin with {{ and end with }}."""


# ── CV result ──────────────────────────────────────────────────────────────────

CV_RESULT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "rejected": {
            "type": "boolean",
            "description": "True if CV checks confidently reject this detection",
        },
        "flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of warning flags (e.g. 'blurry', 'low_edges', 'bad_aspect_ratio')",
        },
        "blur_score": {
            "type": "number",
            "description": "Laplacian variance of the bbox region",
        },
        "edge_ratio": {
            "type": "number",
            "description": "Fraction of edge pixels inside the bbox",
        },
        "color_variance": {
            "type": "number",
            "description": "Standard deviation of pixel values in the bbox",
        },
    },
    "required": ["rejected", "flags"],
    "additionalProperties": False,
}


# ── Pipeline config ────────────────────────────────────────────────────────────

STAGE_STATUSES = ["pending", "approved", "rejected", "relabeled", "flagged", "cv_rejected", "text_rejected"]

PIPELINE_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "cv": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "blur_threshold": {"type": "number", "minimum": 0},
                "min_edge_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                "min_color_stddev": {"type": "number", "minimum": 0},
            },
            "required": ["enabled"],
        },
        "text_llm": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "provider": {"type": "string"},
                "api_key": {"type": "string"},
                "model": {"type": "string"},
            },
            "required": ["enabled"],
        },
        "vlm": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "provider": {"type": "string"},
                "api_key": {"type": "string"},
                "model": {"type": "string"},
            },
            "required": ["enabled"],
        },
    },
    "required": ["cv", "text_llm", "vlm"],
}


# ── Lookup helpers ─────────────────────────────────────────────────────────────

VERDICT_SCHEMAS = {
    "text_llm": (TEXT_LLM_VERDICT_SCHEMA, TEXT_LLM_SYSTEM_PROMPT),
    "vlm": (VLM_VERDICT_SCHEMA, VLM_SYSTEM_PROMPT),
}


def get_verdict_schema(stage: str) -> Tuple[dict, str]:
    """Returns (schema_dict, system_prompt) for a validation stage."""
    if stage not in VERDICT_SCHEMAS:
        raise ValueError(f"Unknown stage: {stage}. Choose from: {list(VERDICT_SCHEMAS.keys())}")
    return VERDICT_SCHEMAS[stage]


def format_system_prompt(stage: str) -> str:
    """Returns system prompt with schema embedded."""
    schema, prompt_template = get_verdict_schema(stage)
    return prompt_template.format(schema=json.dumps(schema, indent=2))
