"""Episode normalisation and deduplication utilities."""
from __future__ import annotations

import hashlib
import html
import logging
import re
import unicodedata
from datetime import UTC, datetime

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


class Enclosure(BaseModel):
    url: str
    mime_type: str = "audio/mpeg"
    length: int = 0


class NormalizedEpisode(BaseModel):
    guid: str
    source_feed_url: str
    original_guid: str = ""
    show_title: str
    episode_title: str
    description: str = ""
    published: datetime
    duration_seconds: int = 0
    episode_url: str = ""
    enclosure: Enclosure | None = None
    image_url: str = ""
    guests: list[str] = Field(default_factory=list)
    is_outside_feed: bool = False
    is_followed_show: bool = False
    category: str = ""

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0


def make_guid(feed_url: str, original_guid: str) -> str:
    raw = f"{feed_url}|{original_guid}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def strip_html(text: str) -> str:
    """Remove HTML tags and normalise whitespace from raw RSS/podcast text.

    Podcast RSS <description> / <content:encoded> fields are almost always
    raw HTML (e.g. "<p><strong>Guest Name</strong> is the CEO of...</p>").
    When that text is used verbatim as a metadata-fallback "summary" (no LLM
    available), unescaped tags either render as literal text in plain
    contexts or, worse, get truncated mid-tag and break the surrounding
    HTML email layout. Always route raw episode text through this first.
    """
    if not text:
        return ""
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_snippet(text: str, limit: int = 300) -> str:
    """Strip HTML from `text` then truncate at a word boundary.

    Used as the fallback "summary" whenever the LLM could not be reached —
    this is show-notes text, not an AI-generated insight, but it should at
    least be readable prose instead of a raw-HTML fragment cut off mid-word.
    """
    cleaned = strip_html(text)
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit].rsplit(" ", 1)[0]
    return truncated.rstrip(".,;:-") + "…"


def _normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.lower().strip()


def _title_fingerprint(show: str, episode: str) -> str:
    return hashlib.sha256(
        f"{_normalise_text(show)}|{_normalise_text(episode)}".encode()
    ).hexdigest()[:12]


def dedup_episodes(
    candidates: list[NormalizedEpisode],
    seen_guids: set[str],
) -> tuple[list[NormalizedEpisode], list[NormalizedEpisode]]:
    """Split candidates into (new, already_seen)."""
    new: list[NormalizedEpisode] = []
    seen: list[NormalizedEpisode] = []
    title_fps: set[str] = set()

    for ep in candidates:
        if ep.guid in seen_guids:
            seen.append(ep)
            continue
        fp = _title_fingerprint(ep.show_title, ep.episode_title)
        if fp in title_fps:
            seen.append(ep)
            continue
        title_fps.add(fp)
        new.append(ep)

    return new, seen


__all__ = [
    "Enclosure", "NormalizedEpisode", "make_guid", "utcnow", "dedup_episodes",
    "strip_html", "clean_snippet",
]
