"""Transcription provider implementations."""
from __future__ import annotations

import asyncio
import base64
import html
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import httpx

from .base import (
    BaseLLMProvider,
    BaseTranscriptionProvider,
    TranscriptResult,
    describe_exception,
)

if TYPE_CHECKING:
    from ..normalize import NormalizedEpisode

log = logging.getLogger(__name__)

_EMPTY = TranscriptResult(text="", confidence="none", source="none")


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
    "application/srt",
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

_TRANSCRIPT_TAG_RE = re.compile(r"<(?:[a-zA-Z0-9_]+:)?transcript\b([^>]*)/?>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""\b([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_ITEM_RE = re.compile(r"<item\b[^>]*>(.*?)</item>", re.IGNORECASE | re.DOTALL)
_GUID_RE = re.compile(r"<guid\b[^>]*>(.*?)</guid>", re.IGNORECASE | re.DOTALL)
_ENCLOSURE_RE = re.compile(r"<enclosure\b([^>]*)>", re.IGNORECASE)
_CDATA_RE = re.compile(r"^\s*<!\[CDATA\[(.*?)\]\]>\s*$", re.DOTALL)


def _strip_vtt(raw: str) -> str:
    """Remove VTT/SRT headers, timestamps, cue indices and inline tags."""
    text = _VTT_STRIP_RE.sub("", raw)
    text = _VTT_CUE_RE.sub(" ", text)
    # Collapse whitespace runs produced by stripping
    return re.sub(r"[ \t]+", " ", text).strip()


def _attrs(fragment: str) -> dict[str, str]:
    return {
        m.group(1).lower(): html.unescape(m.group(2) if m.group(2) is not None else m.group(3))
        for m in _ATTR_RE.finditer(fragment)
    }


def _rank(item: tuple[str, str]) -> int:
    try:
        return _TRANSCRIPT_MIME_PREFERENCE.index(item[1])
    except ValueError:
        return len(_TRANSCRIPT_MIME_PREFERENCE)


def _extract_transcript_urls(rss_text: str) -> list[tuple[str, str]]:
    """Return [(url, type), ...] from podcast:transcript tags, best type first.

    Handles both namespace-prefixed and un-prefixed variants, and attributes
    in either order (the old single regex silently read `type` as missing
    whenever it came before `url`). Sorted so that the caller tries the
    most-readable format (plain text) before falling back to VTT or HTML.

    Scans whatever it is given: pass ONE item's XML, not a whole feed, or the
    result mixes every episode's transcripts together.
    """
    found: list[tuple[str, str]] = []
    for m in _TRANSCRIPT_TAG_RE.finditer(rss_text):
        attrs = _attrs(m.group(1))
        url = attrs.get("url", "").strip()
        if not url:
            continue
        found.append((url, (attrs.get("type") or "text/vtt").strip().lower()))
    return sorted(found, key=_rank)


def _item_text(raw: str) -> str:
    m = _CDATA_RE.match(raw)
    return html.unescape(m.group(1) if m else raw).strip()


def extract_item_transcripts(rss_text: str) -> dict[str, list[tuple[str, str]]]:
    """Map each <item>'s guid AND enclosure URL to that item's own transcripts.

    Both keys are recorded because feedparser's entry id is the guid when
    there is one but falls back to the link, and episodes that arrive from
    Podcast Index carry only the enclosure URL.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for item in _ITEM_RE.finditer(rss_text):
        body = item.group(1)
        candidates = _extract_transcript_urls(body)
        if not candidates:
            continue
        guid = _GUID_RE.search(body)
        if guid and _item_text(guid.group(1)):
            out[_item_text(guid.group(1))] = candidates
        enc = _ENCLOSURE_RE.search(body)
        if enc:
            url = _attrs(enc.group(1)).get("url", "").strip()
            if url:
                out[url] = candidates
    return out


def _clean_transcript(raw: str, mime: str) -> str:
    if mime in ("text/vtt", "application/x-subrip", "application/srt"):
        return _strip_vtt(raw)
    if mime == "application/json":
        # Podcast Index JSON transcript: array of {startTime, body}
        try:
            data = json.loads(raw)
            segments = data if isinstance(data, list) else data.get("segments", [])
            return " ".join(
                str(seg.get("body") or seg.get("text") or "")
                for seg in segments
                if isinstance(seg, dict)
            )
        except Exception:
            return raw  # fall through with raw if JSON is malformed
    if mime == "text/html":
        text = re.sub(r"<(script|style)\b.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        return re.sub(r"[ \t]+", " ", text).strip()
    return raw


async def fetch_transcript_file(candidates: list[tuple[str, str]]) -> TranscriptResult:
    """Download the first usable transcript from [(url, mime), ...], best first."""
    if not candidates:
        return _EMPTY
    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={"User-Agent": "PodcastScout/0.1"},
    ) as client:
        for url, mime in sorted(candidates, key=_rank):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                raw = resp.text
            except Exception as exc:
                log.debug("Transcript fetch failed (%s): %s", url, exc)
                continue

            if not raw.strip():
                continue
            text = _clean_transcript(raw, mime)
            if len(text) > 200:
                log.debug(
                    "Publisher transcript fetched from %s (%d chars, %s)",
                    url, len(text), mime,
                )
                return TranscriptResult(text=text, confidence="high", source="publisher")
    return _EMPTY


class PublisherTranscriptProvider(BaseTranscriptionProvider):
    """Fetches the transcript published by the show itself via the
    podcast:transcript RSS tag.  Free, instant, no compute cost.

    Prefers the tags captured from the episode's own <item> at parse time;
    otherwise re-reads the feed (once per feed per run) and matches the item
    by guid or enclosure URL.
    """

    def __init__(self) -> None:
        self._feed_cache: dict[str, dict[str, list[tuple[str, str]]]] = {}

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        # A bare URL cannot identify an item within a feed.
        return _EMPTY

    async def _item_map(self, feed_url: str) -> dict[str, list[tuple[str, str]]]:
        if feed_url not in self._feed_cache:
            try:
                async with httpx.AsyncClient(
                    timeout=20.0,
                    follow_redirects=True,
                    headers={"User-Agent": "PodcastScout/0.1"},
                ) as client:
                    resp = await client.get(feed_url)
                    resp.raise_for_status()
                    self._feed_cache[feed_url] = extract_item_transcripts(resp.text)
            except Exception as exc:
                log.debug("RSS fetch failed for %s: %s", feed_url, exc)
                self._feed_cache[feed_url] = {}
        return self._feed_cache[feed_url]

    async def transcribe_episode(self, episode: NormalizedEpisode) -> TranscriptResult:
        if episode.transcript_urls:
            return await fetch_transcript_file(
                [(t.url, t.mime_type) for t in episode.transcript_urls]
            )
        if episode.source_type != "podcast" or not episode.source_feed_url.startswith("http"):
            return _EMPTY
        by_item = await self._item_map(episode.source_feed_url)
        keys = [episode.original_guid]
        if episode.enclosure:
            keys.append(episode.enclosure.url)
        for key in keys:
            if key and key in by_item:
                return await fetch_transcript_file(by_item[key])
        return _EMPTY


# ---------------------------------------------------------------------------
# Audio transcription
# ---------------------------------------------------------------------------

# Most shows publish no transcript at all (2 of 26 followed feeds did when
# this was written), so the only way to read an entire episode is to
# transcribe its audio. Both backends cap one request well below a typical
# hour-long MP3 (Gemini inline data ~20MB, OpenAI 25MB), so MP3 audio is cut
# into chunks at frame boundaries and transcribed piece by piece.

_MPEG_TYPES = {"audio/mpeg", "audio/mp3", "audio/mpeg3", "audio/x-mpeg"}


def _is_frame_sync(data: bytes, i: int) -> bool:
    """MPEG audio frame header: 11 set sync bits, a valid layer and bitrate."""
    if i + 3 >= len(data) or data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
        return False
    layer = (data[i + 1] >> 1) & 0x03
    bitrate = data[i + 2] >> 4
    sample_rate = (data[i + 2] >> 2) & 0x03
    return layer != 0 and bitrate not in (0, 0x0F) and sample_rate != 0x03


def split_mp3(data: bytes, chunk_bytes: int) -> list[bytes]:
    """Split MP3 bytes into chunks of at most ~chunk_bytes, each starting on a frame.

    MPEG audio is a sequence of self-contained frames, so a chunk that starts
    on a frame header decodes on its own without re-encoding (no ffmpeg
    needed). A cut can still fall mid-word; that costs a word, not the chunk.
    """
    if len(data) <= chunk_bytes:
        return [data]
    chunks: list[bytes] = []
    start = 0
    while start < len(data):
        end = start + chunk_bytes
        if end >= len(data):
            chunks.append(data[start:])
            break
        # Walk forward to the next frame header; give up after 64KB and cut raw.
        cut = end
        limit = min(len(data), end + 65536)
        while cut < limit and not _is_frame_sync(data, cut):
            cut += 1
        if cut >= limit:
            cut = end
        chunks.append(data[start:cut])
        start = cut
    return chunks


_TRANSCRIBE_PROMPT = (
    "Transcribe this podcast audio verbatim in its original language. Output only the "
    "spoken words as plain text paragraphs. When the speaker changes, start a new "
    "paragraph and prefix it with the speaker's name if it is said, otherwise "
    "'Speaker 1:', 'Speaker 2:' and so on. Do not summarise, translate, add timestamps "
    "or comment on the audio. This is part {part} of {total} of one episode, so it may "
    "start or end mid-sentence."
)


class AudioChunkTranscriber(ABC):
    """Turns one chunk of audio into text. Raises on failure."""

    source: str = "audio"

    @abstractmethod
    async def transcribe_chunk(self, data: bytes, mime_type: str, part: int, total: int) -> str:
        ...


class GeminiAudioTranscriber(AudioChunkTranscriber):
    """Transcribes with Gemini's native audio input, using the key the ranker already has."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    source = "audio"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        thinking_budget: int | None = 0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.thinking_budget = thinking_budget

    async def transcribe_chunk(self, data: bytes, mime_type: str, part: int, total: int) -> str:
        generation: dict[str, object] = {"temperature": 0.0, "maxOutputTokens": 32768}
        if self.thinking_budget is not None:
            generation["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inline_data": {
                        "mime_type": "audio/mp3" if mime_type in _MPEG_TYPES else mime_type,
                        "data": base64.b64encode(data).decode("ascii"),
                    }},
                    {"text": _TRANSCRIBE_PROMPT.format(part=part, total=total)},
                ],
            }],
            "generationConfig": generation,
        }
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{self.BASE_URL}/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key},
                json=payload,
            )
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini audio HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        candidates = body.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini audio returned no candidates: {str(body)[:300]}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            reason = candidates[0].get("finishReason", "unknown")
            raise RuntimeError(f"Gemini audio returned empty text (finishReason={reason})")
        return text.strip()


class OpenAIAudioTranscriber(AudioChunkTranscriber):
    """OpenAI's transcription endpoint (whisper-1 by default)."""

    URL = "https://api.openai.com/v1/audio/transcriptions"
    source = "whisper"

    def __init__(self, api_key: str, model: str = "whisper-1") -> None:
        self.api_key = api_key
        self.model = model

    async def transcribe_chunk(self, data: bytes, mime_type: str, part: int, total: int) -> str:
        ext = "mp3" if mime_type in _MPEG_TYPES else (mime_type.rsplit("/", 1)[-1] or "mp3")
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                self.URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                data={"model": self.model, "response_format": "text"},
                files={"file": (f"part{part}.{ext}", data, mime_type or "audio/mpeg")},
            )
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI transcription HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.text.strip()


class AudioTranscriptionProvider(BaseTranscriptionProvider):
    """Downloads an episode's audio and transcribes all of it, chunk by chunk.

    A transcript missing a chunk is not the entire episode, so any chunk that
    still fails after retries fails the whole episode rather than archiving a
    partial transcript that looks complete.
    """

    def __init__(
        self,
        transcriber: AudioChunkTranscriber,
        chunk_mb: float = 12.0,
        max_audio_mb: float = 400.0,
        max_episodes: int = 8,
        retries: int = 2,
        retry_delay_s: float = 20.0,
    ) -> None:
        self.transcriber = transcriber
        self.chunk_bytes = int(chunk_mb * 1024 * 1024)
        self.max_audio_bytes = int(max_audio_mb * 1024 * 1024)
        self.max_episodes = max_episodes
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self.episodes_attempted = 0

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        return await self._transcribe_audio(episode_url, "audio/mpeg")

    async def transcribe_episode(self, episode: NormalizedEpisode) -> TranscriptResult:
        if episode.source_type != "podcast" or not episode.enclosure:
            return _EMPTY
        return await self._transcribe_audio(episode.enclosure.url, episode.enclosure.mime_type)

    async def _download(self, url: str) -> bytes | None:
        buf = bytearray()
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, read=120.0),
            follow_redirects=True,
            headers={"User-Agent": "PodcastScout/0.1"},
        ) as client, client.stream("GET", url) as resp:
            resp.raise_for_status()
            async for block in resp.aiter_bytes():
                buf.extend(block)
                if len(buf) > self.max_audio_bytes:
                    log.info("Audio over %d bytes, skipping: %s", self.max_audio_bytes, url)
                    return None
        return bytes(buf)

    async def _transcribe_audio(self, url: str, mime_type: str) -> TranscriptResult:
        if not url:
            return _EMPTY
        if self.max_episodes > 0 and self.episodes_attempted >= self.max_episodes:
            log.info("Audio transcription cap (%d/run) reached; skipping %s", self.max_episodes, url)
            return _EMPTY
        self.episodes_attempted += 1
        mime_type = (mime_type or "audio/mpeg").lower()

        try:
            data = await self._download(url)
        except Exception as exc:
            log.warning("Audio download failed for %s: %s", url, describe_exception(exc))
            return _EMPTY
        if not data:
            return _EMPTY

        if mime_type in _MPEG_TYPES:
            chunks = split_mp3(data, self.chunk_bytes)
        elif len(data) <= self.chunk_bytes:
            chunks = [data]
        else:
            # AAC/M4A cannot be cut at arbitrary offsets without re-encoding.
            log.info("Cannot chunk %s audio (%d bytes); skipping %s", mime_type, len(data), url)
            return _EMPTY

        texts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            text = await self._chunk_with_retry(chunk, mime_type, i, len(chunks), url)
            if text is None:
                return _EMPTY
            texts.append(text)

        full = "\n\n".join(texts).strip()
        if len(full) < 200:
            return _EMPTY
        log.info("Transcribed %s: %d chunk(s), %d chars", url, len(chunks), len(full))
        return TranscriptResult(text=full, confidence="high", source=self.transcriber.source)

    async def _chunk_with_retry(
        self, chunk: bytes, mime_type: str, part: int, total: int, url: str
    ) -> str | None:
        for attempt in range(self.retries + 1):
            try:
                return await self.transcriber.transcribe_chunk(chunk, mime_type, part, total)
            except Exception as exc:
                log.warning(
                    "Audio chunk %d/%d of %s failed (attempt %d): %s",
                    part, total, url, attempt + 1, describe_exception(exc),
                )
                if attempt < self.retries:
                    await asyncio.sleep(self.retry_delay_s * (attempt + 1))
        return None


# ---------------------------------------------------------------------------
# Existing providers
# ---------------------------------------------------------------------------

class NullTranscriptionProvider(BaseTranscriptionProvider):
    """Returns empty transcript — used when no transcription API is configured."""

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        return _EMPTY


class DescriptionFallbackProvider(BaseTranscriptionProvider):
    """Uses episode description as a transcript substitute."""

    async def transcribe(self, episode_url: str, description: str = "") -> TranscriptResult:
        if not description:
            return _EMPTY
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
# Cascade provider — publisher transcript, then audio, then description
# ---------------------------------------------------------------------------

class CascadeTranscriptionProvider(BaseTranscriptionProvider):
    """Tries publisher transcript → audio transcription → description → empty.

    The publisher transcript check (podcast:transcript RSS tag) is free and
    instant and is always attempted first. Audio transcription runs only when
    an `audio` provider is given (ENABLE_AUDIO_TRANSCRIPTION).
    """

    def __init__(
        self,
        audio: AudioTranscriptionProvider | None = None,
        llm: BaseLLMProvider | None = None,
    ) -> None:
        self._publisher = PublisherTranscriptProvider()
        self._audio = audio
        self._llm_enhancer = LLMDescriptionEnhancer(llm) if llm else None
        self._description = DescriptionFallbackProvider()

    async def transcribe(
        self,
        episode_url: str,
        description: str = "",
        feed_url: str = "",
    ) -> TranscriptResult:
        # Without an episode there is nothing to match a feed item or an
        # enclosure against, so only the description paths apply.
        return await self._fallback(description)

    async def transcribe_episode(self, episode: NormalizedEpisode) -> TranscriptResult:
        # 1. Publisher transcript via podcast:transcript RSS tag
        result = await self._publisher.transcribe_episode(episode)
        if result.text:
            return result

        # 2. Audio transcription — only if explicitly enabled
        if self._audio:
            result = await self._audio.transcribe_episode(episode)
            if result.text:
                return result

        return await self._fallback(episode.description)

    async def _fallback(self, description: str) -> TranscriptResult:
        # 3. LLM description enhancer
        if self._llm_enhancer and description:
            return await self._llm_enhancer.enhance("", "", description)

        # 4. Raw description fallback
        return await self._description.transcribe("", description)
