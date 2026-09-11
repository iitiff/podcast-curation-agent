"""Web search providers: Brave (primary), Serper (alternative), Null fallback."""
from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseWebSearchProvider, WebSearchResult


class BraveSearchProvider(BaseWebSearchProvider):
    _BASE = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15))
    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                self._BASE,
                params={"q": query, "count": max_results},
                headers={"Accept": "application/json", "X-Subscription-Token": self._key},
            )
            if resp.status_code != 200:
                return []
            items = resp.json().get("web", {}).get("results", [])
            return [
                WebSearchResult(title=i.get("title", ""), url=i.get("url", ""), snippet=i.get("description", ""))
                for i in items[:max_results]
            ]


class SerperSearchProvider(BaseWebSearchProvider):
    _BASE = "https://google.serper.dev/search"

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15))
    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                self._BASE,
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": self._key, "Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                return []
            return [
                WebSearchResult(title=i.get("title", ""), url=i.get("link", ""), snippet=i.get("snippet", ""))
                for i in resp.json().get("organic", [])[:max_results]
            ]


class NullWebSearchProvider(BaseWebSearchProvider):
    async def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        return []
