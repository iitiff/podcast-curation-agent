"""OpenAI LLM provider (default). Swap out by implementing BaseLLMProvider."""
from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=60.0,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def complete(self, messages: list[dict[str, str]], model: str, response_format: Any = None, max_tokens: int = 2000) -> str:
        payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if response_format:
            payload["response_format"] = response_format
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def count_tokens(self, text: str) -> int:
        # Rough estimate: 1 token ≈ 4 chars
        return len(text) // 4

    async def aclose(self) -> None:
        await self._client.aclose()
