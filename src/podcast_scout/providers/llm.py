"""OpenAI LLM provider with token tracking and cost estimation."""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

# Approximate costs per 1K tokens (update as pricing changes)
_COST_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4-turbo": {"input": 0.010, "output": 0.030},
}


class TokenBudget:
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.used_tokens = 0
        self.estimated_cost_usd = 0.0

    def record(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        self.used_tokens += prompt_tokens + completion_tokens
        costs = _COST_PER_1K.get(model, {"input": 0.005, "output": 0.015})
        self.estimated_cost_usd += (
            prompt_tokens / 1000 * costs["input"]
            + completion_tokens / 1000 * costs["output"]
        )

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def exhausted(self) -> bool:
        return self.used_tokens >= self.max_tokens


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        stage1_model: str = "gpt-4o-mini",
        stage2_model: str = "gpt-4o",
        budget: TokenBudget | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.budget = budget or TokenBudget(500_000)
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=60.0,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def complete(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None = None,
        model: str | None = None,
        max_tokens: int = 2000,
    ) -> str:
        if self.budget.exhausted:
            raise RuntimeError("LLM token budget exhausted for this run.")

        resolved_model = model or self.stage2_model
        payload: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        resp = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        self.budget.record(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            resolved_model,
        )
        content = data["choices"][0]["message"]["content"]
        return content

    def estimate_tokens(self, text: str) -> int:
        # Rough approximation: 1 token ≈ 4 chars
        return max(1, len(text) // 4)

    async def aclose(self) -> None:
        await self._client.aclose()
