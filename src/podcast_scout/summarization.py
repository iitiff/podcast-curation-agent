"""Transcript fetching and evidence-level determination for Stage 2."""
from __future__ import annotations

import logging

import httpx

from .normalize import NormalizedEpisode
from .providers.transcription import (
    PublisherTranscriptScraper,
    TranscriptConfidence,
    TranscriptResult,
)

logger = logging.getLogger(__name__)

MAX_SOURCE_CHARS = 12_000


async def get_best_source_text(
    ep: NormalizedEpisode,
    scraper: PublisherTranscriptScraper,
    client: httpx.AsyncClient,
) -> TranscriptResult:
    """
    Obtain the best available source text in priority order:
    1. Podcasting 2.0 transcript tag
    2. Publisher webpage transcript/show notes
    3. Episode description (fallback)
    """
    # 1. P2.0 transcript URL
    if ep.transcript_url:
        try:
            resp = await client.get(ep.transcript_url, follow_redirects=True)
            resp.raise_for_status()
            text = resp.text[:MAX_SOURCE_CHARS]
            if len(text) > 1000:
                logger.debug("P2.0 transcript for '%s': %d chars", ep.episode_title, len(text))
                return TranscriptResult(text, TranscriptConfidence.HIGH, "p20_transcript")
        except Exception as exc:
            logger.debug("P2.0 transcript fetch failed: %s", exc)

    # 2. Publisher page scrape
    if ep.episode_url:
        result = await scraper.fetch(ep.episode_url)
        if result and len(result.text) > 500:
            return result

    # 3. Show notes / description fallback
    if ep.show_notes_html and len(ep.show_notes_html) > 200:
        return TranscriptResult(
            ep.show_notes_html[:MAX_SOURCE_CHARS],
            TranscriptConfidence.MEDIUM,
            "show_notes",
        )
    if ep.description:
        return TranscriptResult(
            ep.description[:MAX_SOURCE_CHARS],
            TranscriptConfidence.LOW,
            "description_only",
        )

    return TranscriptResult("", TranscriptConfidence.LOW, "no_source")
