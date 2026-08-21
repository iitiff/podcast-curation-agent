"""LLM provider implementations."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseLLMProvider, LLMMessage, LLMResponse

log = logging.getLogger(__name__)


class GitHubModelsProvider(BaseLLMProvider):
    """!! RETIRED — DO NOT USE. Kept for historical reference only. !!

    GitHub permanently retired the entire GitHub Models product on
    2026-07-30. The playground, model catalog, inference API, and BYOK are
    all gone for every customer, including accounts with active usage.
    Requests to the inference endpoint now return **410 Gone**.
    See: https://github.blog/changelog/2026-07-30-github-models-is-now-retired/

    Endpoint history (all now dead):
      - https://models.inference.ai.azure.com  (deprecated 2025-07-17,
        support removed 2025-10-17)
      - https://models.github.ai/inference     (retired 2026-07-30, 410 Gone)

    There is no replacement endpoint. cli.py._make_llm() intentionally never
    constructs this class. If you are here because episodes aren't getting
    real LLM scores, the fix is a working GEMINI_API_KEY (or adding a
    different provider), NOT another endpoint change.
    """

    BASE_URL = "https://models.github.ai/inference"

    def __init__(self, token: str, model: str = "openai/gpt-4.1") -> None:
        log.warning(
            "GitHubModelsProvider was instantiated, but GitHub Models was "
            "permanently retired on 2026-07-30 and returns 410 Gone. "
            "All requests through this provider will fail."
        )
        self.token = token
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



def _extract_gemini_text(data: dict[str, Any], max_tokens: int) -> str:
    """Pull the generated text out of a Gemini generateContent response.

    Deliberately strict. The previous implementation did
    `data["candidates"][0]["content"]["parts"][0]["text"]`, which had two
    failure modes that both surfaced as silent quality loss rather than errors:

      1. When thinking consumes the whole budget the API returns
         finishReason=MAX_TOKENS with NO parts at all. That raised a bare
         `IndexError: list index out of range`, which the batch ranker caught
         and turned into metadata-only scoring for every episode -- no
         indication that thinking tokens were the cause.
      2. Gemini may split output across MULTIPLE parts. Reading only parts[0]
         silently truncated the JSON array mid-way.

    This joins all parts and raises a message naming finishReason and the
    thoughts/prompt token counts, so the next failure is diagnosable from the
    log alone.
    """
    usage = data.get("usageMetadata", {})
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(
            f"Gemini returned no candidates "
            f"(promptFeedback={data.get('promptFeedback')})"
        )

    candidate = candidates[0]
    finish_reason = candidate.get("finishReason")
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)

    if not text:
        raise RuntimeError(
            f"Gemini returned empty text (finishReason={finish_reason}, "
            f"maxOutputTokens={max_tokens}, "
            f"thoughtsTokens={usage.get('thoughtsTokenCount')}, "
            f"promptTokens={usage.get('promptTokenCount')}). "
            f"If finishReason is MAX_TOKENS the budget was consumed before any "
            f"output was produced -- check thinkingConfig."
        )
    if finish_reason == "MAX_TOKENS":
        log.warning(
            "Gemini hit MAX_TOKENS (maxOutputTokens=%s, thoughtsTokens=%s) — the "
            "response is TRUNCATED; downstream parsing will recover only the "
            "entries that made it.",
            max_tokens, usage.get("thoughtsTokenCount"),
        )
    return text



class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider — the primary (and currently only) LLM."""

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
                # CRITICAL: disable thinking.
                #
                # gemini-2.5-flash defaults to *dynamic thinking*, and thinking
                # tokens are billed against maxOutputTokens. When
                # thoughts + output exceed that cap the API returns
                # finishReason=MAX_TOKENS with an EMPTY text part -- the JSON
                # array we asked for never arrives at all.
                #
                # Observed in production: a 5-episode batch came back with only
                # 2 objects (truncated) or none (empty), and every unmatched
                # episode silently fell through to metadata-only scoring at the
                # 50.0 floor with no summary and no key ideas.
                # https://ai.google.dev/gemini-api/docs/thinking  (2.5 Flash:
                # thinkingBudget 0 disables thinking)
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        url = f"{self.BASE_URL}/{self.model}:generateContent?key={self.api_key}"
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        usage = data.get("usageMetadata", {})
        text = _extract_gemini_text(data, max_tokens)

        return LLMResponse(
            content=text,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
        )


class OpenAICompatibleProvider(BaseLLMProvider):
    """Generic provider for ANY OpenAI-compatible /chat/completions endpoint.

    Deliberately provider-agnostic: NVIDIA NIM (build.nvidia.com), OpenRouter,
    Groq, Together, Fireworks, vLLM, and a self-hosted NIM container all speak
    this same wire format. Point it at a base URL + key + model and it works.

    This exists instead of an NvidiaProvider class specifically so that the next
    provider swap is a config change, not a code change. The GitHub Models
    retirement (2026-07-30) demonstrated how fast a hosted inference dependency
    can vanish; keeping the transport generic means the blast radius next time
    is one environment variable.

    Note on JSON adherence: stage2_batch_rank parses a raw JSON array from the
    response, and _parse_llm_json_array already strips markdown fences and
    attempts json_repair. Smaller / more chat-tuned models sometimes prepend
    prose despite instructions -- that is handled downstream, not here.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        provider_name: str = "openai-compatible",
    ) -> None:
        self.api_key = api_key
        # Tolerate a trailing slash in the configured base URL.
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider_name = provider_name

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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
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


class FallbackLLMProvider(BaseLLMProvider):
    """Wraps a primary LLM provider with an automatic runtime fallback.

    Currently unused (Gemini is the only live provider after GitHub Models was
    retired), but retained because it's the correct shape for adding a second
    provider later: complete() tries the primary and, on ANY exception,
    transparently retries the same request with the secondary before giving up.

    Why per-call and not per-startup: choosing one provider once at startup
    means a mid-run failure (401, 410, rate limit, timeout) silently degrades
    every subsequent episode to metadata-only scoring with no retry. Doing the
    fallback inside complete() means each batch gets its own second chance.
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
