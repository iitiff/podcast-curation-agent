"""Episode normalisation and deduplication utilities."""
from __future__ import annotations

import hashlib
import logging
import unicodedata
from datetime import datetime, timezone, UTC
from urllib.parse import urlparse

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


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


# keep timezone available for any callers that import from normalize
__all__ = ["Enclosure", "NormalizedEpisode", "make_guid", "utcnow", "dedup_episodes", "timezone"]
