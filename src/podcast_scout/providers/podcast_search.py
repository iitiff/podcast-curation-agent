"""Podcast search providers: Podcast Index (primary), iTunes (fallback)."""
from __future__ import annotations

import hashlib
import time

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BasePodcastSearchProvider, PodcastSearchResult


class PodcastIndexProvider(BasePodcastSearchProvider):
    _BASE = "https://api.podcastindex.org/api/1.0"

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._key = api_key
        self._secret = api_secret

    def _auth_headers(self) -> dict[str, str]:
        ts = str(int(time.time()))
        # Podcast Index auth: SHA-1 of (apiKey + apiSecret + unixTimestamp)
        # https://podcastindex-org.github.io/docs-api/#auth
        sig = hashlib.sha1(
            (self._key + self._secret + ts).encode("utf-8")
        ).hexdigest()
        return {
            "X-Auth-Key": self._key,
            "X-Auth-Date": ts,
            "Authorization": sig,
            "User-Agent": "PodcastScout/0.1",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15))
    async def search_episodes(
        self, query: str, max_results: int = 10
    ) -> list[PodcastSearchResult]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                f"{self._BASE}/search/byterm",
                params={"q": query, "max": max_results},
                headers=self._auth_headers(),
            )
            if resp.status_code != 200:
                return []
            feeds = resp.json().get("feeds", [])
            results = []
            for feed in feeds[:max_results]:
                results.append(
                    PodcastSearchResult(
                        feed_url=feed.get("url", ""),
                        show_title=feed.get("title", ""),
                        episode_title="",
                        description=feed.get("description", ""),
                        image_url=feed.get("image", ""),
                        source="podcast_index",
                    )
                )
            return results


class ITunesSearchProvider(BasePodcastSearchProvider):
    _BASE = "https://itunes.apple.com/search"

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10))
    async def search_episodes(
        self, query: str, max_results: int = 10
    ) -> list[PodcastSearchResult]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                self._BASE,
                params={
                    "term": query,
                    "media": "podcast",
                    "entity": "podcast",
                    "limit": max_results,
                },
            )
            if resp.status_code != 200:
                return []
            results = []
            for item in resp.json().get("results", [])[:max_results]:
                results.append(
                    PodcastSearchResult(
                        feed_url=item.get("feedUrl", ""),
                        show_title=item.get("collectionName", ""),
                        episode_title="",
                        description=item.get("description", ""),
                        image_url=item.get("artworkUrl600", ""),
                        source="itunes",
                    )
                )
            return results


class NullPodcastSearchProvider(BasePodcastSearchProvider):
    """No-op provider used when no API keys are configured."""
    async def search_episodes(self, query: str, max_results: int = 10) -> list[PodcastSearchResult]:
        return []
