"""Base interfaces for all external service providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None = None,
        model: str | None = None,
        max_tokens: int = 2000,
    ) -> str: ...

    @abstractmethod
    def estimate_tokens(self, text: str) -> int: ...


class BasePodcastSearchProvider(ABC):
    @abstractmethod
    async def search_episodes(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, Any]]: ...


class BaseWebSearchProvider(ABC):
    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]: ...


class BaseTranscriptionProvider(ABC):
    @abstractmethod
    async def transcribe(
        self,
        audio_url: str,
        max_minutes: float = 30.0,
    ) -> str: ...
