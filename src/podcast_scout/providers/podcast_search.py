"""Podcast search providers: Podcast Index + iTunes fallback."""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BasePodcastSearchProvider

logger = logging.getLogger(__name__)


class PodcastIndexProvider(BasePodcastSearchProvider):
    """https://api.podcastindex.org — requires API key + secret."""

    BASE_URL = "https://api.podcastindex.org/api/1.0"

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._key = api_key
        self._secret = api_secret
        self._client = httpx.AsyncClient(timeout=20.0)

    def _auth_headers(self) -> dict[str, str]:
        ts = str(int(time.time()))
        h = hashlib.sha1(f"{self._key}{self._secret}{ts}".encode()).hexdigest()  # noqa: S324
        return {
            "X-Auth-Key": self._key,
            "X-Auth-Date": ts,
            "Authorization": h,
            "User-Agent": "PodcastScout/0.1",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def search_episodes(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        resp = await self._client.get(
            f"{self.BASE_URL}/search/byterm",
            params={"q": query, "max": max_results, "fulltext": True},
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json().get("feeds", [])

    async def aclose(self) -> None:
        await self._client.aclose()


class ITunesFallbackProvider(BasePodcastSearchProvider):
    """Apple/iTunes podcast search — no auth required, rate-limited."""

    BASE_URL = "https://itunes.apple.com/search"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=15.0)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=15))
    async def search_episodes(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        resp = await self._client.get(
            self.BASE_URL,
            params={
                "term": query,
                "media": "podcast",
                "entity": "podcast",
                "limit": max_results,
            },
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    async def aclose(self) -> None:
        await self._client.aclose()


class NullPodcastSearchProvider(BasePodcastSearchProvider):
    """No-op provider used when no search credentials are configured."""

    async def search_episodes(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        logger.warning("No podcast search provider configured; skipping discovery.")
        return []
