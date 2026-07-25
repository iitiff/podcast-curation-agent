"""Web search providers: Brave Search + Serper fallback."""
from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseWebSearchProvider

logger = logging.getLogger(__name__)


class BraveSearchProvider(BaseWebSearchProvider):
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str) -> None:
        self._client = httpx.AsyncClient(
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            timeout=15.0,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        resp = await self._client.get(
            self.BASE_URL,
            params={"q": query, "count": max_results, "search_lang": "en"},
        )
        resp.raise_for_status()
        return resp.json().get("web", {}).get("results", [])

    async def aclose(self) -> None:
        await self._client.aclose()


class SerperProvider(BaseWebSearchProvider):
    BASE_URL = "https://google.serper.dev/search"

    def __init__(self, api_key: str) -> None:
        self._client = httpx.AsyncClient(
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=15.0,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        resp = await self._client.post(
            self.BASE_URL,
            json={"q": query, "num": max_results},
        )
        resp.raise_for_status()
        return resp.json().get("organic", [])

    async def aclose(self) -> None:
        await self._client.aclose()


class NullWebSearchProvider(BaseWebSearchProvider):
    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        logger.warning("No web search provider configured; skipping web discovery.")
        return []
