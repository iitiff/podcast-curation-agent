"""Transcription providers: publisher page scraping, YouTube captions, Whisper."""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseTranscriptionProvider

logger = logging.getLogger(__name__)


class TranscriptConfidence(str, Enum):
    HIGH = "high"       # full or near-full transcript
    MEDIUM = "medium"   # detailed show notes or substantial excerpt
    LOW = "low"         # title + short description only


class TranscriptResult:
    def __init__(self, text: str, confidence: TranscriptConfidence, source: str) -> None:
        self.text = text
        self.confidence = confidence
        self.source = source

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class PublisherTranscriptScraper:
    """Attempts to extract transcript text from a podcast episode webpage."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=20.0, follow_redirects=True)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def fetch(self, episode_url: str) -> TranscriptResult:
        try:
            resp = await self._client.get(episode_url)
            resp.raise_for_status()
        except Exception as exc:
            logger.debug("Publisher scrape failed for %s: %s", episode_url, exc)
            return TranscriptResult("", TranscriptConfidence.LOW, "publisher_scrape_failed")

        soup = BeautifulSoup(resp.text, "lxml")

        # Look for common transcript containers
        for selector in [
            "[class*='transcript']",
            "[id*='transcript']",
            "article",
            "[class*='show-notes']",
            "[class*='episode-notes']",
        ]:
            el = soup.select_one(selector)
            if el and len(el.get_text()) > 500:
                text = el.get_text(separator=" ", strip=True)
                confidence = (
                    TranscriptConfidence.HIGH
                    if len(text) > 3000
                    else TranscriptConfidence.MEDIUM
                )
                return TranscriptResult(text[:50_000], confidence, "publisher_page")

        # Fallback: grab all paragraph text
        paragraphs = " ".join(p.get_text() for p in soup.find_all("p"))
        if len(paragraphs) > 300:
            return TranscriptResult(
                paragraphs[:20_000], TranscriptConfidence.MEDIUM, "publisher_paragraphs"
            )

        return TranscriptResult("", TranscriptConfidence.LOW, "publisher_no_content")

    async def aclose(self) -> None:
        await self._client.aclose()


class NullTranscriptionProvider(BaseTranscriptionProvider):
    async def transcribe(self, audio_url: str, max_minutes: float = 30.0) -> str:
        logger.info("Audio transcription disabled; skipping %s", audio_url)
        return ""
