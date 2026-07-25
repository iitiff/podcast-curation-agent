"""RSS/Atom feed fetching and episode extraction."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx
from dateutil import parser as dateparser
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Settings, ShowOverride, ShowsConfig
from .normalize import Enclosure, NormalizedEpisode, make_guid, parse_duration

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "PodcastScout/0.1 (github.com/iitiff/podcast-cpo-scout)"}
TIMEOUT = 20


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _fetch_raw(url: str) -> str:
    with httpx.Client(timeout=TIMEOUT, headers=HEADERS, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.text


def _parse_published(entry: Any) -> datetime:
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, None)
        if raw:
            try:
                dt = dateparser.parse(raw)
                if dt and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt or datetime.now(tz=timezone.utc)
            except Exception:
                continue
    return datetime.now(tz=timezone.utc)


def _get_enclosure(entry: Any) -> Enclosure | None:
    for enc in getattr(entry, "enclosures", []):
        url = getattr(enc, "url", "") or getattr(enc, "href", "")
        if url and ("audio" in getattr(enc, "type", "") or url.endswith(".mp3")):
            return Enclosure(
                url=url,
                mime_type=getattr(enc, "type", "audio/mpeg") or "audio/mpeg",
                length=int(getattr(enc, "length", 0) or 0),
            )
    return None


def _get_transcript_url(entry: Any) -> str:
    # Podcasting 2.0 transcript tag
    for link in getattr(entry, "links", []):
        rel = getattr(link, "rel", "")
        typ = getattr(link, "type", "")
        href = getattr(link, "href", "")
        if rel == "transcript" or "transcript" in typ:
            return href
    return ""


def _extract_guests(entry: Any) -> list[str]:
    """Best-effort guest extraction from title/description."""
    guests: list[str] = []
    title = getattr(entry, "title", "") or ""
    # Common patterns: "with Guest Name", "feat. Guest", "| Guest Name"
    import re
    patterns = [
        r"(?:with|feat\.?|featuring|guest[:]?)\s+([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)+)",
        r"\|\s*([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)+)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, title)
        if m:
            guests.append(m.group(1).strip())
            break
    return guests


def _resolve_show_override(
    show_title: str, shows_config: ShowsConfig
) -> ShowOverride | None:
    title_lower = show_title.lower()
    for override in shows_config.shows:
        if override.match.lower() in title_lower:
            return override
    return None


def fetch_feed(
    feed_url: str,
    show_title: str,
    since: datetime,
    settings: Settings,
    shows_config: ShowsConfig,
    is_outside: bool = False,
) -> list[NormalizedEpisode]:
    """Fetch a single feed and return NormalizedEpisodes published after `since`."""
    override = _resolve_show_override(show_title, shows_config)

    if override and not override.enabled:
        log.info("Skipping disabled show: %s", show_title)
        return []

    url = (override.canonical_feed_url if override and override.canonical_feed_url else feed_url)
    max_eps = override.max_episodes_per_run if override else 3

    try:
        raw = _fetch_raw(url)
    except Exception as exc:
        log.warning("Failed to fetch %s (%s): %s", show_title, url, exc)
        return []

    try:
        parsed = feedparser.parse(raw)
    except Exception as exc:
        log.warning("Failed to parse feed %s: %s", url, exc)
        return []

    feed_show_title = (
        (override.display_name if override and override.display_name else None)
        or getattr(parsed.feed, "title", show_title)
        or show_title
    )

    episodes: list[NormalizedEpisode] = []
    for entry in parsed.entries:
        pub = _parse_published(entry)
        if pub < since:
            continue

        original_guid = (
            getattr(entry, "id", None)
            or getattr(entry, "guid", None)
            or getattr(entry, "link", None)
            or f"{url}:{entry.get('title', '')}"
        )
        guid = make_guid(url, str(original_guid))

        description = (
            getattr(entry, "summary", "")
            or getattr(entry, "content", [{"value": ""}])[0].get("value", "")
            or ""
        )

        duration_raw = (
            getattr(entry, "itunes_duration", None)
            or getattr(entry, "duration", None)
        )

        episodes.append(
            NormalizedEpisode(
                guid=guid,
                source_feed_url=url,
                original_guid=str(original_guid),
                show_title=feed_show_title,
                episode_title=getattr(entry, "title", "Untitled"),
                description=description[:2000],
                published=pub,
                duration_seconds=parse_duration(str(duration_raw) if duration_raw else None),
                guests=_extract_guests(entry),
                keywords=[
                    t.get("term", "") for t in getattr(entry, "tags", []) if t.get("term")
                ],
                episode_url=getattr(entry, "link", "") or "",
                enclosure=_get_enclosure(entry),
                image_url=(
                    getattr(entry, "image", {}).get("href", "")
                    or getattr(parsed.feed, "image", {}).get("href", "")
                    or ""
                ),
                is_outside_feed=is_outside,
                transcript_url=_get_transcript_url(entry),
                show_notes_html=description,
            )
        )
        if len(episodes) >= max_eps:
            break

    log.info("Fetched %d new episodes from %s", len(episodes), feed_show_title)
    return episodes
