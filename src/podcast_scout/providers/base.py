"""Abstract base classes for all provider types."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class TranscriptResult(BaseModel):
    text: str = ""
    confidence: str = "none"  # high | medium | low | none
    source: str = "none"      # whisper | description | none


class EpisodeSearchResult(BaseModel):
    feed_url: str
    show_title: str = ""
    episode_title: str = ""
    description: str = ""
    duration_seconds: int = 0
    episode_url: str = ""
    enclosure_url: str = ""
    image_url: str = ""


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str
    input_tokens: int = 0
    output_tokens: int = 0


class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 4096,
    ) -> LLMResponse: ...


class BasePodcastSearchProvider(ABC):
    @abstractmethod
    async def search_episodes(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[EpisodeSearchResult]: ...


class BaseWebSearchProvider(ABC):
    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict]: ...
