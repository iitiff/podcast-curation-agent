"""RSS feed fetching and parsing utilities."""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx

from .normalize import Enclosure, NormalizedEpisode, make_guid, utcnow

log = logging.getLogger(__name__)

FEED_FETCH_TIMEOUT = 20.0


def _safe_str(val: Any, default: str = "") -> str:
    if val is None:
        return default
    return str(val).strip()


def _parse_struct_time(val: Any) -> datetime | None:
    if val:
        try:
            return datetime(*val[:6], tzinfo=UTC)
        except Exception:
            pass
    return None


async def fetch_feed_text(url: str) -> str:
    async with httpx.AsyncClient(timeout=FEED_FETCH_TIMEOUT, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "podcast-scout/1.0"})
        resp.raise_for_status()
        return resp.text


def parse_feed_entries(
    text: str,
    feed_url: str,
    show_name_override: str | None,
    cutoff: datetime,
    max_entries: int = 10,
) -> list[NormalizedEpisode]:
    parsed = feedparser.parse(text)
    show_title = show_name_override or _safe_str(parsed.feed.get("title"), "Unknown Show")
    image_url = ""
    if hasattr(parsed.feed, "image") and parsed.feed.image:
        image_url = _safe_str(getattr(parsed.feed.image, "href", ""))

    episodes: list[NormalizedEpisode] = []
    for entry in parsed.entries[:max_entries * 3]:
        pub = _parse_struct_time(getattr(entry, "published_parsed", None))
        if pub is None:
            pub = utcnow()
        if pub < cutoff:
            continue

        title = _safe_str(getattr(entry, "title", ""), "Untitled")
        link = _safe_str(getattr(entry, "link", ""))
        summary = _safe_str(getattr(entry, "summary", ""))
        orig_guid = _safe_str(getattr(entry, "id", "")) or link or title
        duration_str = _safe_str(getattr(entry, "itunes_duration", ""))
        duration_seconds = _parse_duration(duration_str)
        image = image_url
        if hasattr(entry, "image") and entry.image:
            image = _safe_str(getattr(entry.image, "href", "")) or image
        enclosure: Enclosure | None = None
        for enc in getattr(entry, "enclosures", []):
            url = _safe_str(getattr(enc, "href", ""))
            if url:
                enclosure = Enclosure(
                    url=url,
                    mime_type=_safe_str(getattr(enc, "type", "audio/mpeg")),
                    length=int(getattr(enc, "length", 0) or 0),
                )
                break

        ep = NormalizedEpisode(
            guid=make_guid(feed_url, orig_guid),
            source_feed_url=feed_url,
            original_guid=orig_guid,
            show_title=show_title,
            episode_title=title,
            description=summary,
            published=pub,
            duration_seconds=duration_seconds,
            episode_url=link,
            enclosure=enclosure,
            image_url=image,
        )
        episodes.append(ep)
        if len(episodes) >= max_entries:
            break

    return episodes


def _parse_duration(s: str) -> int:
    """Parse iTunes duration string (HH:MM:SS or MM:SS or seconds) to seconds."""
    if not s:
        return 0
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(s)
    except ValueError:
        return 0


__all__ = ["fetch_feed_text", "parse_feed_entries", "timezone", "timedelta"]
