"""Web search providers: Brave, Serper, and null fallback."""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseWebSearchProvider


class BraveSearchProvider(BaseWebSearchProvider):
    BASE = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(self.BASE, params={"q": f"{query} podcast", "count": max_results}, headers={"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": self._key})
            resp.raise_for_status()
            return resp.json().get("web", {}).get("results", [])


class SerperProvider(BaseWebSearchProvider):
    def __init__(self, api_key: str) -> None:
        self._key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post("https://google.serper.dev/search", json={"q": f"{query} podcast", "num": max_results}, headers={"X-API-KEY": self._key, "Content-Type": "application/json"})
            resp.raise_for_status()
            return resp.json().get("organic", [])


class NullWebSearchProvider(BaseWebSearchProvider):
    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        return []
