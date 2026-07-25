"""Fetch and parse RSS/Atom podcast feeds into NormalizedEpisode objects."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .normalize import Enclosure, NormalizedEpisode, make_guid, parse_duration, utcnow
from .opml import OPMLFeed

logger = logging.getLogger(__name__)


def _parse_date(entry: Any) -> datetime:
    for field in ("published_parsed", "updated_parsed"):
        t = getattr(entry, field, None)
        if t:
            try:
                import calendar
                ts = calendar.timegm(t)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                continue
    return utcnow()


def _extract_enclosure(entry: Any) -> Enclosure | None:
    for enc in getattr(entry, "enclosures", []):
        url = getattr(enc, "href", "") or getattr(enc, "url", "")
        if url and "audio" in getattr(enc, "type", "audio/"):
            return Enclosure(
                url=url,
                mime_type=getattr(enc, "type", "audio/mpeg"),
                length=int(getattr(enc, "length", 0) or 0),
            )
    # Some feeds put audio in media:content
    for mc in getattr(entry, "media_content", []):
        url = mc.get("url", "")
        if url and "audio" in mc.get("type", "audio/"):
            return Enclosure(url=url, mime_type=mc.get("type", "audio/mpeg"))
    return None


def _extract_guests(entry: Any) -> list[str]:
    """Best-effort guest extraction from tags and title."""
    guests: list[str] = []
    tags = getattr(entry, "tags", []) or []
    for tag in tags:
        term = getattr(tag, "term", "") or ""
        if term and len(term) < 60:
            guests.append(term)
    return guests[:10]


def _find_p20_transcript(entry: Any) -> str:
    """Look for a Podcasting 2.0 <podcast:transcript> tag."""
    # feedparser exposes custom namespaces via entry items
    for key, val in entry.items():
        if "transcript" in key.lower() and isinstance(val, list):
            for item in val:
                url = item.get("url", "") if isinstance(item, dict) else ""
                if url:
                    return url
    return ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
async def fetch_feed_raw(url: str, client: httpx.AsyncClient) -> str:
    resp = await client.get(url, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


async def fetch_episodes_from_feed(
    feed: OPMLFeed,
    lookback_days: int,
    max_episodes: int,
    client: httpx.AsyncClient,
) -> list[NormalizedEpisode]:
    """Fetch and normalize episodes from a single RSS feed."""
    cutoff = utcnow() - timedelta(days=lookback_days)
    episodes: list[NormalizedEpisode] = []

    try:
        raw = await fetch_feed_raw(feed.xml_url, client)
    except Exception as exc:
        logger.warning("Failed to fetch feed %s: %s", feed.xml_url, exc)
        return []

    try:
        parsed = feedparser.parse(raw)
    except Exception as exc:
        logger.warning("Failed to parse feed %s: %s", feed.xml_url, exc)
        return []

    show_title = (
        getattr(parsed.feed, "title", "") or feed.title or "Unknown Show"
    ).strip()
    show_image = getattr(parsed.feed, "image", {}).get("href", "") or ""

    count = 0
    for entry in parsed.entries:
        if count >= max_episodes:
            break

        pub_date = _parse_date(entry)
        if pub_date < cutoff:
            continue

        original_guid = (
            getattr(entry, "id", "")
            or getattr(entry, "guid", "")
            or getattr(entry, "link", "")
            or f"{feed.xml_url}#{entry.get('title', count)}"
        )
        guid = make_guid(feed.xml_url, original_guid)

        enclosure = _extract_enclosure(entry)
        duration_raw = (
            getattr(entry, "itunes_duration", None)
            or getattr(entry, "duration", None)
        )
        description = (
            getattr(entry, "summary", "")
            or getattr(entry, "description", "")
            or ""
        )[:5000]

        episodes.append(
            NormalizedEpisode(
                guid=guid,
                source_feed_url=feed.xml_url,
                original_guid=original_guid,
                show_title=show_title,
                episode_title=(
                    getattr(entry, "title", "Untitled Episode") or "Untitled Episode"
                ).strip(),
                description=description,
                published=pub_date,
                duration_seconds=parse_duration(str(duration_raw) if duration_raw else None),
                guests=_extract_guests(entry),
                keywords=[t.get("term", "") for t in getattr(entry, "tags", []) if isinstance(t, dict)][:15],
                episode_url=getattr(entry, "link", "") or "",
                enclosure=enclosure,
                image_url=getattr(entry, "image", {}).get("href", "") or show_image,
                transcript_url=_find_p20_transcript(entry),
                show_notes_html=description,
            )
        )
        count += 1

    logger.info("Fetched %d episodes from %s", len(episodes), show_title)
    return episodes
