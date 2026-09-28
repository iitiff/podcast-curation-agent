"""Abstract base classes for external service providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..normalize import NormalizedEpisode

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
    published_timestamp: int | None = None  # Unix timestamp from Podcast Index
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
    # "publisher" | "audio" (Gemini) | "whisper" (OpenAI) | "description" | "none"
    source: str = "none"
    confidence: str = "low"   # "high" | "medium" | "low" | "none"

    @property
    def is_full(self) -> bool:
        """Is this the whole episode, rather than a description standing in for it?"""
        return bool(self.text) and self.source in FULL_TRANSCRIPT_SOURCES


# Sources that carry the entire spoken episode. Everything else is the show
# notes, which say what an episode is about but not what was said in it.
FULL_TRANSCRIPT_SOURCES = frozenset({"publisher", "audio", "whisper"})


class BaseTranscriptionProvider(ABC):
    @abstractmethod
    async def transcribe(
        self, episode_url: str, description: str = ""
    ) -> TranscriptResult:
        ...

    async def transcribe_episode(self, episode: NormalizedEpisode) -> TranscriptResult:
        """Transcribe with everything known about the episode.

        The default keeps providers that only understand a URL working; the
        cascade overrides it to use the feed, the per-item transcript tags and
        the audio enclosure, none of which fit through `transcribe()`.
        """
        return await self.transcribe(episode.episode_url, episode.description)


def describe_exception(exc: BaseException) -> str:
    """Render an exception so the log line always says something.

    httpx's timeout exceptions carry no message, so `str(exc)` is "" and a
    log line built with %s reads "LLM call failed:  — falling back", which
    cannot distinguish a timeout from a refusal from a malformed response.
    Observed in production 2026-09-17: a two-minute NVIDIA fallback died with
    an empty reason and took the run's only personalization batch with it.
    """
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
