"""LLM provider implementations."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseLLMProvider, LLMMessage, LLMResponse

log = logging.getLogger(__name__)


class GitHubModelsProvider(BaseLLMProvider):
    """GitHub Models inference API (OpenAI-compatible).

    Uses GITHUB_TOKEN which is automatically available in all GitHub Actions
    runs (with `models: read` requested in the workflow permissions) — no
    extra secret required.

    NOTE: GitHub migrated the inference endpoint from the legacy
    `models.inference.ai.azure.com` host to `models.github.ai/inference`.
    The old host stopped honoring the Actions-issued GITHUB_TOKEN and returns
    401 Unauthorized even when `models: read` is correctly granted. Model IDs
    on the new endpoint also require a publisher namespace prefix, e.g.
    "openai/gpt-4o" instead of the old bare "gpt-4o".
    See: https://github.blog/ai-and-ml/llms/solving-the-inference-problem-for-open-source-ai-projects-with-github-models/
    """

    BASE_URL = "https://models.github.ai/inference"

    def __init__(self, token: str, model: str = "openai/gpt-4.1") -> None:
        self.token = token
        # Defensive normalization: if a bare model id (no publisher namespace)
        # is passed — e.g. via a stale GITHUB_MODELS_MODEL env override — assume
        # the OpenAI publisher, since that's what this project always used.
        self.model = model if "/" in model else f"openai/{model}"

    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        url = f"{self.BASE_URL}/chat/completions"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            content=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.api_key = api_key
        self.model = model

    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        system_parts = [m.content for m in messages if m.role == "system"]
        user_parts = [m.content for m in messages if m.role != "system"]

        system_instruction: dict[str, Any] | None = None
        if system_parts:
            system_instruction = {"parts": [{"text": "\n\n".join(system_parts)}]}

        contents = [{"role": "user", "parts": [{"text": "\n\n".join(user_parts)}]}]

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.3,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        return LLMResponse(
            content=text,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
        )


class FallbackLLMProvider(BaseLLMProvider):
    """Wraps a primary LLM provider with an automatic runtime fallback.

    Previously, the pipeline picked ONE provider at startup based purely on
    which credential was present (GITHUB_TOKEN is always set in Actions, so
    Gemini was never actually tried even when GEMINI_API_KEY was configured).
    Any runtime failure of the primary (e.g. a 401 from GitHub Models) was
    caught deep inside stage2_batch_rank and silently downgraded to
    metadata-only scoring for that batch — never retried with Gemini.

    This wrapper closes that gap: complete() tries the primary provider first
    and, on any exception, transparently retries the same request with the
    secondary provider before giving up. Call sites (stage2_batch_rank,
    generate_synthesis, etc.) are unchanged — they just see one BaseLLMProvider.
    """

    def __init__(
        self,
        primary: BaseLLMProvider,
        secondary: BaseLLMProvider,
        primary_name: str = "primary",
        secondary_name: str = "secondary",
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.primary_name = primary_name
        self.secondary_name = secondary_name

    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        try:
            return await self.primary.complete(messages, max_tokens=max_tokens)
        except Exception as exc:
            log.warning(
                "%s LLM call failed (%s) — retrying with %s",
                self.primary_name, exc, self.secondary_name,
            )
            return await self.secondary.complete(messages, max_tokens=max_tokens)
