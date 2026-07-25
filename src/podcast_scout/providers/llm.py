"""Google Gemini LLM provider."""
from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseLLMProvider, LLMMessage, LLMResponse


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self._model = model
        self._client = genai.Client(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def complete(
        self,
        messages: list[LLMMessage],
        response_format: type[BaseModel] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        # Split system message from conversation turns
        system_parts: list[str] = []
        contents: list[types.Content] = []

        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                role = "user" if m.role == "user" else "model"
                contents.append(types.Content(role=role, parts=[types.Part(text=m.content)]))

        system_instruction = "\n\n".join(system_parts) if system_parts else None

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if response_format is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_format
        else:
            # Ask Gemini to return JSON even without a strict schema
            config_kwargs["response_mime_type"] = "application/json"

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        gen_config = types.GenerateContentConfig(**config_kwargs)

        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=gen_config,
        )

        content = resp.text or ""
        usage = resp.usage_metadata
        return LLMResponse(
            content=content,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            model=self._model,
        )

    def estimate_tokens(self, text: str) -> int:
        # ~4 chars per token is a safe approximation
        return max(1, len(text) // 4)

    async def aclose(self) -> None:
        pass  # google-genai client has no explicit close
