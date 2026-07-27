"""Abstract base classes for external service providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Podcast search
# ---------------------------------------------------------------------------

@dataclass
class PodcastSearchResult:
    feed_url: str
    show_title: str
    episode_title: str
    description: str = ""
    duration_seconds: int = 0
    episode_url: str = ""
    enclosure_url: str = ""
    image_url: str = ""
    published_timestamp: Optional[int] = None  # Unix timestamp from Podcast Index
    source: str = ""


class BasePodcastSearchProvider(ABC):
    @abstractmethod
    async def search_episodes(
        self, query: str, max_results: int = 10
    ) -> list[PodcastSearchResult]:
        ...

    async def fetch_recent_episodes(
        self, feed_url: str, max_results: int = 5
    ) -> list[PodcastSearchResult]:
        """Optional: fetch recent episodes by feed URL. Default returns empty list."""
        return []


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

@dataclass
class WebSearchResult:
    title: str = ""
    url: str = ""
    snippet: str = ""


class BaseWebSearchProvider(ABC):
    @abstractmethod
    async def search(
        self, query: str, max_results: int = 5
    ) -> list[WebSearchResult]:
        ...


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------

@dataclass
class LLMMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0


class BaseLLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        ...


# ---------------------------------------------------------------------------
# Transcription provider
# ---------------------------------------------------------------------------

@dataclass
class TranscriptResult:
    text: str = ""
    source: str = "none"      # "whisper" | "publisher" | "description" | "none"
    confidence: str = "low"   # "high" | "medium" | "low" | "none"


class BaseTranscriptionProvider(ABC):
    @abstractmethod
    async def transcribe(
        self, episode_url: str, description: str = ""
    ) -> TranscriptResult:
        ...
