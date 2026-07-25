"""Episode normalization and deduplication utilities."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urlparse

from pydantic import BaseModel, Field


class Enclosure(BaseModel):
    url: str
    mime_type: str = "audio/mpeg"
    length: int = 0


class NormalizedEpisode(BaseModel):
    # Stable identity
    guid: str
    source_feed_url: str
    original_guid: str

    # Metadata
    show_title: str
    episode_title: str
    description: str = ""
    published: datetime
    duration_seconds: int = 0
    guests: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    episode_url: str = ""
    enclosure: Enclosure | None = None
    image_url: str = ""

    # Source tracking
    is_outside_feed: bool = False
    is_followed_show: bool = False  # True when fetched directly from a followed show's RSS feed
    transcript_url: str = ""
    show_notes_html: str = ""

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60


def make_guid(feed_url: str, original_guid: str) -> str:
    """Create a stable, globally unique GUID from source feed + original GUID."""
    raw = f"{_normalise_url(feed_url)}|{original_guid}"
    return hashlib.sha256(raw.encode()).hexdigest()[:40]


def _normalise_url(url: str) -> str:
    parsed = urlparse(url.lower().strip())
    # Drop trailing slashes, fragments, and sort query params for stability
    return parsed._replace(fragment="").geturl().rstrip("/")


def parse_duration(raw: str | None) -> int:
    """Parse iTunes-style duration (HH:MM:SS or seconds) into seconds."""
    if not raw:
        return 0
    raw = raw.strip()
    if ":" in raw:
        parts = raw.split(":")
        try:
            parts_int = [int(p) for p in parts]
            if len(parts_int) == 3:
                return parts_int[0] * 3600 + parts_int[1] * 60 + parts_int[2]
            if len(parts_int) == 2:
                return parts_int[0] * 60 + parts_int[1]
        except ValueError:
            return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def dedup_episodes(
    episodes: list[NormalizedEpisode],
    seen_guids: set[str],
) -> tuple[list[NormalizedEpisode], set[str]]:
    """Remove episodes whose GUID is already in seen_guids.
    Returns (new_episodes, updated_seen_guids).
    """
    new: list[NormalizedEpisode] = []
    for ep in episodes:
        if ep.guid not in seen_guids:
            new.append(ep)
            seen_guids.add(ep.guid)
    return new, seen_guids


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)
