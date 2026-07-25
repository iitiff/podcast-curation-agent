"""Base interfaces for all external service providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: str  # system | user | assistant
    content: str


class LLMResponse(BaseModel):
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        response_format: type[BaseModel] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...

    @abstractmethod
    def estimate_tokens(self, text: str) -> int: ...


class PodcastSearchResult(BaseModel):
    feed_url: str
    show_title: str
    episode_title: str
    description: str = ""
    published_raw: str = ""
    duration_seconds: int = 0
    episode_url: str = ""
    enclosure_url: str = ""
    image_url: str = ""
    source: str = ""  # podcast_index | itunes | web


class BasePodcastSearchProvider(ABC):
    @abstractmethod
    async def search_episodes(
        self, query: str, max_results: int = 10
    ) -> list[PodcastSearchResult]: ...


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class BaseWebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]: ...


class TranscriptResult(BaseModel):
    text: str
    source: str  # p20 | publisher | youtube | whisper | none
    confidence: str  # high | medium | low
    word_count: int = 0


class BaseTranscriptionProvider(ABC):
    @abstractmethod
    async def get_transcript(
        self,
        episode_url: str,
        audio_url: str = "",
        max_audio_bytes: int = 0,
    ) -> TranscriptResult: ...
