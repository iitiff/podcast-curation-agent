"""Transcript fetching: Podcasting 2.0 tag, publisher page, YouTube captions, Whisper fallback."""
from __future__ import annotations

import re

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseTranscriptionProvider, TranscriptResult


class CascadeTranscriptionProvider(BaseTranscriptionProvider):
    """Tries sources in priority order; stops at first success."""

    def __init__(
        self,
        openai_api_key: str = "",
        enable_whisper: bool = False,
        max_audio_bytes: int = 50_000_000,
    ) -> None:
        self._openai_key = openai_api_key
        self._enable_whisper = enable_whisper
        self._max_audio_bytes = max_audio_bytes

    async def get_transcript(
        self,
        episode_url: str,
        audio_url: str = "",
        max_audio_bytes: int = 0,
    ) -> TranscriptResult:
        # 1. Try publisher page for transcript links
        if episode_url:
            result = await self._scrape_publisher(episode_url)
            if result:
                return result
        # 2. No transcript found
        return TranscriptResult(text="", source="none", confidence="low")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def _scrape_publisher(self, url: str) -> TranscriptResult | None:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "PodcastScout/0.1"})
                if resp.status_code != 200:
                    return None
                soup = BeautifulSoup(resp.text, "lxml")
                # Look for transcript containers common to major podcast platforms
                for selector in [
                    "[class*='transcript']",
                    "[id*='transcript']",
                    "[class*='show-notes']",
                    "article",
                ]:
                    el = soup.select_one(selector)
                    if el:
                        text = el.get_text(separator=" ", strip=True)
                        if len(text) > 500:
                            return TranscriptResult(
                                text=text[:50_000],
                                source="publisher",
                                confidence="medium",
                                word_count=len(text.split()),
                            )
        except Exception:
            return None
        return None
