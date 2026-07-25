"""Podcast Index search provider with iTunes fallback."""
from __future__ import annotations

import hashlib
import time
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BasePodcastSearchProvider


class PodcastIndexProvider(BasePodcastSearchProvider):
    BASE = "https://api.podcastindex.org/api/1.0"

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._key = api_key
        self._secret = api_secret

    def _headers(self) -> dict[str, str]:
        epoch = int(time.time())
        h = hashlib.sha1(f"{self._key}{self._secret}{epoch}".encode()).hexdigest()  # noqa: S324
        return {"X-Auth-Key": self._key, "X-Auth-Date": str(epoch), "Authorization": h, "User-Agent": "PodcastScout/1.0"}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(f"{self.BASE}/search/byterm", params={"q": query, "max": max_results}, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
            return data.get("feeds", [])


class ItunesFallbackProvider(BasePodcastSearchProvider):
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get("https://itunes.apple.com/search", params={"term": query, "media": "podcast", "limit": max_results})
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [{"title": r.get("collectionName"), "url": r.get("feedUrl"), "description": r.get("description", "")} for r in results if r.get("feedUrl")]


class NullPodcastSearchProvider(BasePodcastSearchProvider):
    async def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        return []
