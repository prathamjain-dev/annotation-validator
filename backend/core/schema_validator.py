"""
Schema validation utilities for structured LLM output.
Handles stripping markdown fences, JSON Schema validation, and retry logic.
"""

import json
import logging
from typing import Optional, Tuple, List

import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def strip_markdown_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def validate_against_schema(data: dict, schema: dict) -> List[str]:
    try:
        validate(instance=data, schema=schema)
        return []
    except ValidationError as e:
        path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
        return [f"At '{path}': {e.message}"]
    except Exception as e:
        return [str(e)]


def _build_error_feedback(errors: List[str], attempt: int, max_retries: int) -> str:
    retries_left = max_retries - attempt
    return (
        "\n\n"
        "⚠️ Your previous response failed validation with these errors:\n"
        + "\n".join(f"  - {e}" for e in errors)
        + f"\n\nFix your response and try again. ({retries_left} retries remaining)"
        "\nRespond with ONLY a valid JSON object matching the schema. "
        "No markdown, no backticks, no code fences."
    )


async def validate_with_retry(
    schema: dict,
    provider_complete,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = MAX_RETRIES,
) -> Tuple[Optional[dict], Optional[List[str]]]:
    """
    Calls the provider, validates output against schema, retries on failure.

    Returns:
        (parsed_dict, None) on success
        (None, error_list) if all retries exhausted
    """
    current_user_prompt = user_prompt

    for attempt in range(max_retries + 1):
        try:
            raw = await provider_complete(system_prompt, current_user_prompt)
        except Exception as e:
            logger.error(f"Provider call failed (attempt {attempt+1}): {e}")
            if attempt < max_retries:
                current_user_prompt += (
                    f"\n\n⚠️ API call failed: {e}. Please retry."
                )
                continue
            return None, [f"Provider call failed after {max_retries+1} attempts: {e}"]

        cleaned = strip_markdown_fences(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON (attempt {attempt+1}): {cleaned[:200]}")
            if attempt < max_retries:
                current_user_prompt += (
                    "\n\n⚠️ Your response was not valid JSON. "
                    "Return ONLY a raw JSON object. No markdown, no backticks, no code fences."
                )
                continue
            return None, [f"Invalid JSON after {max_retries+1} attempts: {e}"]

        errors = validate_against_schema(parsed, schema)
        if not errors:
            return parsed, None

        logger.warning(f"Schema validation failed (attempt {attempt+1}): {errors}")
        if attempt < max_retries:
            current_user_prompt += _build_error_feedback(errors, attempt, max_retries)
            continue

        return None, errors

    return None, ["Unexpected error in validation loop"]
