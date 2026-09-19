"""Transcription provider implementations."""
from __future__ import annotations

import logging
import re

import httpx

from .base import BaseLLMProvider, BaseTranscriptionProvider, TranscriptResult

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Publisher transcript (podcast:transcript RSS tag)
# ---------------------------------------------------------------------------

# Supported MIME types in preference order. VTT and SRT need stripping;
# plain text and JSON are used as-is (JSON is the Podcast Index native format).
_TRANSCRIPT_MIME_PREFERENCE = [
    "text/plain",
    "application/json",
    "text/vtt",
    "application/x-subrip",  # .srt
    "text/html",
]

_VTT_STRIP_RE = re.compile(
    r"^WEBVTT.*?(\n\n|\r\n\r\n)",  # header block
    re.DOTALL,
)
_VTT_CUE_RE = re.compile(
    r"(?:^|\n)\d{2}:\d{2}[^\n]*\n"   # timestamp line
    r"|(?:^|\n)\d+\s*\n"             # cue index line
    r"|<[^>]+>",                      # inline tags
    re.MULTILINE,
)


def _strip_vtt(raw: str) -> str:
    """Remove VTT/SRT headers, timestamps, cue indices and inline tags."""
    text = _VTT_STRIP_RE.sub("", raw)
    text = _VTT_CUE_RE.sub(" ", text)
    # Collapse whitespace runs produced by stripping
    return re.sub(r"[ \t]+", " ", text).strip()


def _extract_transcript_urls(rss_text: str) -> list[tuple[str, str]]:
    """Return [(url, type), ...] from podcast:transcript tags, best type first.

    Handles both namespace-prefixed and un-prefixed variants.  Sorted so that
    the caller tries the most-readable format (plain text) before falling back
    to VTT or HTML.
    """
    pattern = re.compile(
        r'<(?:[a-zA-Z0-9_]+:)?transcript\b[^>]*\burl=["\']([^"\']+)["\'][^>]*'
        r'(?:\btype=["\']([^"\']+)["\'])?[^>]*/?>',
        re.IGNORECASE,
    )
    found: list[tuple[str, str]] = []
    for m in pattern.finditer(rss_text):
        url = m.group(1).strip()
        mime = (m.group(2) or "text/vtt").strip().lower()
        found.append((url, mime))

    def _rank(item: tuple[str, str]) -> int:
        try:
            return _TRANSCRIPT_MIME_PREFERENCE.index(item[1])
        except ValueError:
            return len(_TRANSCRIPT_MIME_PREFERENCE)

    return sorted(found, key=_rank)


async def _fetch_rss_transcript(feed_url: str) -> TranscriptResult:
    """Try to retrieve a publisher transcript from the RSS feed.

    Steps:
      1. Fetch the RSS/Atom feed XML.
      2. Find all <podcast:transcript> tags.
      3. Download the best-ranked transcript file.
      4. Strip VTT/SRT markup if needed.

    Returns an empty TranscriptResult on any failure so the cascade can
    continue rather than raising.
    """
    if not feed_url:
        return TranscriptResult(text="", confidence="none", source="none")

    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": "PodcastScout/0.1"},
        ) as client:
            rss_resp = await client.get(feed_url)
            rss_resp.raise_for_status()
            rss_text = rss_resp.text
    except Exception as exc:
        log.debug("RSS fetch failed for %s: %s", feed_url, exc)
        return TranscriptResult(text="", confidence="none", source="none")

    candidates = _extract_transcript_urls(rss_text)
    if not candidates:
        log.debug("No podcast:transcript tags found in %s", feed_url)
        return TranscriptResult(text="", confidence="none", source="none")

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "PodcastScout/0.1"},
    ) as client:
        for url, mime in candidates:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                raw = resp.text
            except Exception as exc:
                log.debug("Transcript fetch failed (%s): %s", url, exc)
                continue

            if not raw.strip():
                continue

            if mime in ("text/vtt", "application/x-subrip"):
                text = _strip_vtt(raw)
            elif mime == "application/json":
                # Podcast Index JSON transcript: array of {startTime, body}
                import json as _json
                try:
                    data = _json.loads(raw)
                    segments = data if isinstance(data, list) else data.get("segments", [])
                    text = " ".join(
                        seg.get("body", seg.get("text", ""))
                        for seg in segments
                        if isinstance(seg, dict)
                    )
                except Exception:
                    text = raw  # fall through with raw if JSON is malformed
            elif mime == "text/html":
                text = re.sub(r"<[^>]+>", " ", raw)
                text = re.sub(r"[ \t]+", " ", text).strip()
            else:
                text = raw

            if len(text) > 200:
                log.debug(
                    "Publisher transcript fetched from %s (%d chars, %s)",
                    url, len(text), mime,
                )
                return TranscriptResult(
                    text=text,
                    confidence="high",
                    source="publisher",
                )

    return TranscriptResult(text="", confidence="none", source="none")


class PublisherTranscriptProvider(BaseTranscriptionProvider):
    """Fetches the transcript published by the show itself via the
    podcast:transcript RSS tag.  Free, instant, no compute cost.

    ``feed_url`` is the RSS feed URL (not the episode enclosure URL).
    The ``episode_url`` parameter passed to ``transcribe()`` is ignored;
    the caller should pass ``feed_url`` as the first argument, or wire this
    provider via ``CascadeTranscriptionProvider`` which does that automatically.
    """

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        return await _fetch_rss_transcript(episode_url)


# ---------------------------------------------------------------------------
# Existing providers (unchanged)
# ---------------------------------------------------------------------------

class NullTranscriptionProvider(BaseTranscriptionProvider):
    """Returns empty transcript — used when no transcription API is configured."""

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        return TranscriptResult(text="", confidence="none", source="none")


class WhisperTranscriptionProvider(BaseTranscriptionProvider):
    """OpenAI Whisper transcription provider."""

    def __init__(self, api_key: str, max_audio_mb: float = 25.0) -> None:
        self.api_key = api_key
        self.max_audio_bytes = int(max_audio_mb * 1024 * 1024)

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                head = await client.head(episode_url)
                content_length = int(head.headers.get("content-length", 0))
                if content_length > self.max_audio_bytes:
                    log.info("Audio too large (%d bytes), skipping Whisper", content_length)
                    return TranscriptResult(text="", confidence="none", source="none")
                resp = await client.get(episode_url)
                resp.raise_for_status()
                audio_data = resp.content
        except Exception as exc:
            log.warning("Failed to fetch audio from %s: %s", episode_url, exc)
            return TranscriptResult(text="", confidence="none", source="none")

        import openai
        client = openai.AsyncOpenAI(api_key=self.api_key)
        try:
            result = await client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.mp3", audio_data, "audio/mpeg"),
            )
            return TranscriptResult(
                text=result.text,
                confidence="high",
                source="whisper",
            )
        except Exception as exc:
            log.warning("Whisper transcription failed: %s", exc)
            return TranscriptResult(text="", confidence="none", source="none")


class DescriptionFallbackProvider(BaseTranscriptionProvider):
    """Uses episode description as a transcript substitute."""

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        if not description:
            return TranscriptResult(text="", confidence="none", source="none")
        return TranscriptResult(
            text=description[:3000],
            confidence="low",
            source="description",
        )


class LLMDescriptionEnhancer:
    """Uses an LLM to expand a short description into a richer pseudo-transcript."""

    def __init__(self, llm: BaseLLMProvider) -> None:
        self.llm = llm

    async def enhance(self, show: str, episode: str, description: str) -> TranscriptResult:
        from .base import LLMMessage
        prompt = (
            f"You are a podcast analyst. Given the show '{show}', episode '{episode}', "
            f"and this description:\n\n{description[:1000]}\n\n"
            "Write a 200-word expanded summary of what this episode likely covers, "
            "including probable key ideas and any named guests or companies."
        )
        try:
            resp = await self.llm.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                max_tokens=400,
            )
            return TranscriptResult(
                text=resp.content,
                confidence="low",
                source="description",
            )
        except Exception as exc:
            log.warning("LLM description enhancement failed: %s", exc)
            return TranscriptResult(text=description[:1000], confidence="low", source="description")


# ---------------------------------------------------------------------------
# Cascade provider — publisher transcript first, then Whisper, then description
# ---------------------------------------------------------------------------

class CascadeTranscriptionProvider(BaseTranscriptionProvider):
    """Tries publisher transcript → Whisper → description fallback → empty.

    The publisher transcript check (podcast:transcript RSS tag) is free and
    instant and should always be attempted first.  ``feed_url`` is passed to
    it instead of the episode enclosure URL because the tag lives in the RSS
    feed, not the audio file.
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        llm: BaseLLMProvider | None = None,
        enable_whisper: bool = False,
    ) -> None:
        self._publisher = PublisherTranscriptProvider()
        self._whisper = WhisperTranscriptionProvider(openai_api_key) if (openai_api_key and enable_whisper) else None
        self._llm_enhancer = LLMDescriptionEnhancer(llm) if llm else None
        self._description = DescriptionFallbackProvider()

    async def transcribe(
        self,
        episode_url: str,
        description: str = "",
        feed_url: str = "",
    ) -> TranscriptResult:
        # 1. Publisher transcript via podcast:transcript RSS tag
        result = await self._publisher.transcribe(feed_url or episode_url)
        if result.text:
            return result

        # 2. Whisper (audio transcription) — only if explicitly enabled
        if self._whisper:
            result = await self._whisper.transcribe(episode_url)
            if result.text:
                return result

        # 3. LLM description enhancer
        if self._llm_enhancer and description:
            return await self._llm_enhancer.enhance("", "", description)

        # 4. Raw description fallback
        return await self._description.transcribe(episode_url, description)
