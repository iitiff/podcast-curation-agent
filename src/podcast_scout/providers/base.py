"""Abstract base classes for external service providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


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


class BaseWebSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        ...
