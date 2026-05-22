"""
Adala Agent Integration Layer
Handles annotation validation, reasoning, and correction via LLMs.
Supports OpenAI, Anthropic Claude, OpenRouter, and local vLLM endpoints.

Uses JSON Schema-driven validation with retry logic for reliable structured output.
"""

import json
import logging
from typing import List, Dict, Any, Optional

import httpx

from backend.core.schemas import TEXT_LLM_VERDICT_SCHEMA, TEXT_LLM_SYSTEM_PROMPT, format_system_prompt
from backend.core.schema_validator import validate_with_retry

logger = logging.getLogger(__name__)


# ── LLM Provider Abstraction ───────────────────────────────────────────────────

class LLMProvider:
    """Base class for LLM providers."""

    async def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    async def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 1200,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model

    async def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {
            "model": self.model,
            "max_tokens": 1200,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]


class VLLMProvider(LLMProvider):
    """Local vLLM / Ollama OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 1200,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter — unified API gateway to 200+ models.
    Uses the OpenAI-compatible /chat/completions endpoint.
    Docs: https://openrouter.ai/docs
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/llama-3.1-8b-instruct:free",
        site_url: str = "http://localhost:8501",
        site_name: str = "Adala Auto Annotator",
    ):
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.site_name = site_name

    async def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": 1200,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.site_name,
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"OpenRouter error: {data['error']}")
            return data["choices"][0]["message"]["content"]


# Curated list of popular OpenRouter models for the UI dropdown
OPENROUTER_POPULAR_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "mistralai/mistral-large",
    "google/gemini-pro-1.5",
    "google/gemini-flash-1.5",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "cohere/command-r-plus",
    "perplexity/llama-3.1-sonar-large-128k-online",
]


def get_provider(config: Dict[str, Any]) -> LLMProvider:
    """Factory: build LLM provider from config dict."""
    provider = config.get("provider", "openai").lower()
    json_mode = config.get("json_mode", False)

    if provider == "openai":
        return OpenAIProvider(
            api_key=config["api_key"],
            model=config.get("model", "gpt-4o-mini"),
        )
    elif provider in ("anthropic", "claude"):
        return AnthropicProvider(
            api_key=config["api_key"],
            model=config.get("model", "claude-sonnet-4-20250514"),
        )
    elif provider == "vllm":
        return VLLMProvider(
            base_url=config["base_url"],
            model=config.get("model", "llama3"),
        )
    elif provider == "openrouter":
        return OpenRouterProvider(
            api_key=config["api_key"],
            model=config.get("model", "meta-llama/llama-3.1-8b-instruct:free"),
            site_url=config.get("site_url", "http://localhost:8501"),
            site_name=config.get("site_name", "Adala Auto Annotator"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Adala Agent ────────────────────────────────────────────────────────────────

def _build_user_prompt(
    image_name: str,
    detection: Dict[str, Any],
    all_classes: List[str],
    image_context: Optional[str] = None,
) -> str:
    bbox = detection.get("bbox", [])
    bbox_str = (
        f"[x1={bbox[0]:.0f}, y1={bbox[1]:.0f}, x2={bbox[2]:.0f}, y2={bbox[3]:.0f}]"
        if len(bbox) == 4 else str(bbox)
    )

    prompt = f"""Image: {image_name}
Available classes: {all_classes}
{"Context: " + image_context if image_context else ""}

Detection to validate:
- Predicted class: {detection.get('class', 'unknown')}
- Confidence: {detection.get('confidence', 0):.3f}
- Bounding box: {bbox_str}

Validate this detection and respond with JSON matching the schema described in the system prompt."""
    return prompt


class AdalaAnnotationAgent:
    """
    Schema-driven annotation validation agent.
    Validates detections against a JSON Schema using an LLM backend.
    Retries on malformed or invalid output.
    """

    def __init__(
        self,
        llm_config: Dict[str, Any],
        schema: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        max_retries: int = 2,
    ):
        self.provider = get_provider(llm_config)
        self.json_mode = llm_config.get("json_mode", True)
        self.schema = schema or TEXT_LLM_VERDICT_SCHEMA
        self.system_prompt_template = system_prompt or TEXT_LLM_SYSTEM_PROMPT
        self.max_retries = max_retries

    def _build_system_prompt(self) -> str:
        return self.system_prompt_template.format(
            schema=json.dumps(self.schema, indent=2)
        )

    def _merge_result(self, detection: Dict[str, Any], verdict: dict) -> Dict[str, Any]:
        enriched = dict(detection)
        enriched["adala_status"] = verdict.get("status", "flagged")
        enriched["adala_reasoning"] = verdict.get("reasoning", "")
        enriched["adala_suggested_class"] = verdict.get("suggested_class")
        enriched["adala_confidence_adjustment"] = verdict.get("confidence_adjustment", 0.0)

        if enriched["adala_status"] == "relabeled" and enriched["adala_suggested_class"]:
            enriched["class"] = enriched["adala_suggested_class"]

        return enriched

    def _flagged_result(self, detection: Dict[str, Any], reason: str) -> Dict[str, Any]:
        enriched = dict(detection)
        enriched["adala_status"] = "flagged"
        enriched["adala_reasoning"] = reason
        enriched["adala_suggested_class"] = None
        enriched["adala_confidence_adjustment"] = 0.0
        return enriched

    async def validate_detection(
        self,
        image_name: str,
        detection: Dict[str, Any],
        all_classes: List[str],
        image_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate a single detection. Returns enriched detection dict."""
        user_prompt = _build_user_prompt(image_name, detection, all_classes, image_context)
        system_prompt = self._build_system_prompt()

        async def _complete(system: str, user: str) -> str:
            return await self.provider.complete(system, user, json_mode=self.json_mode)

        result, errors = await validate_with_retry(
            schema=self.schema,
            provider_complete=_complete,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_retries=self.max_retries,
        )

        if result is None:
            return self._flagged_result(
                detection,
                f"Validation failed after retries: {'; '.join(errors) if errors else 'unknown error'}"
            )

        return self._merge_result(detection, result)

    async def validate_all(
        self,
        image_name: str,
        detections: List[Dict[str, Any]],
        all_classes: List[str],
        image_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Validate all detections for an image."""
        results = []
        for det in detections:
            enriched = await self.validate_detection(image_name, det, all_classes, image_context)
            results.append(enriched)
        return results
