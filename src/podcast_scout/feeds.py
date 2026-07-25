"""RSS/Atom feed fetching and episode extraction."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .normalize import Enclosure, NormalizedEpisode, make_guid, parse_duration, utcnow
from .opml import OPMLFeed

log = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
async def fetch_feed_text(url: str, timeout: int = 20) -> str:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "PodcastScout/0.1 +https://github.com"})
        resp.raise_for_status()
        return resp.text


def _parse_date(entry: Any) -> datetime:
    for field in ("published_parsed", "updated_parsed"):
        val = getattr(entry, field, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return utcnow()


def _extract_enclosure(entry: Any) -> Enclosure | None:
    for enc in getattr(entry, "enclosures", []):
        url = getattr(enc, "href", "") or getattr(enc, "url", "")
        if url and "audio" in getattr(enc, "type", "audio/mpeg"):
            return Enclosure(
                url=url,
                mime_type=getattr(enc, "type", "audio/mpeg"),
                length=int(getattr(enc, "length", 0) or 0),
            )
    return None


def _extract_guests(title: str, description: str) -> list[str]:
    """Heuristic guest extraction from episode title/description."""
    guests: list[str] = []
    import re
    patterns = [
        r"with ([A-Z][a-z]+ [A-Z][a-z]+)",
        r"feat\.?\s+([A-Z][a-z]+ [A-Z][a-z]+)",
        r"featuring ([A-Z][a-z]+ [A-Z][a-z]+)",
        r"guest[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)",
    ]
    for text in (title, description[:500]):
        for pat in patterns:
            matches = re.findall(pat, text)
            guests.extend(matches)
    return list(dict.fromkeys(guests))[:5]  # dedupe, cap at 5


def parse_feed_entries(
    feed_text: str,
    feed_url: str,
    show_title: str,
    lookback_cutoff: datetime,
    max_episodes: int = 3,
) -> list[NormalizedEpisode]:
    parsed = feedparser.parse(feed_text)
    episodes: list[NormalizedEpisode] = []

    # Fall back to feed title if show_title not provided
    feed_show_title = show_title or parsed.feed.get("title", "Unknown Show")

    for entry in parsed.entries:
        pub_date = _parse_date(entry)
        if pub_date < lookback_cutoff:
            continue

        original_guid = entry.get("id") or entry.get("guid") or entry.get("link", "")
        if not original_guid:
            continue

        guid = make_guid(feed_url, original_guid)
        title = entry.get("title", "Untitled")
        desc = entry.get("summary") or entry.get("description") or ""
        duration_raw = entry.get("itunes_duration") or ""
        enclosure = _extract_enclosure(entry)
        ep_url = entry.get("link", "")
        image = (
            entry.get("image", {}).get("href", "")
            or parsed.feed.get("image", {}).get("href", "")
        )
        # Podcasting 2.0 transcript tag
        transcript_url = ""
        for link in entry.get("links", []):
            if "transcript" in link.get("rel", "").lower():
                transcript_url = link.get("href", "")
                break

        episodes.append(
            NormalizedEpisode(
                guid=guid,
                source_feed_url=feed_url,
                original_guid=original_guid,
                show_title=feed_show_title,
                episode_title=title,
                description=desc[:2000],
                published=pub_date,
                duration_seconds=parse_duration(duration_raw),
                guests=_extract_guests(title, desc),
                episode_url=ep_url,
                enclosure=enclosure,
                image_url=image,
                transcript_url=transcript_url,
                show_notes_html=desc,
            )
        )
        if len(episodes) >= max_episodes:
            break

    return episodes


async def fetch_episodes_from_feed(
    feed: OPMLFeed,
    lookback_days: int,
    max_episodes: int = 3,
) -> tuple[list[NormalizedEpisode], str | None]:
    """Fetch and parse episodes from a single feed. Returns (episodes, error_msg)."""
    cutoff = utcnow() - timedelta(days=lookback_days)
    try:
        text = await fetch_feed_text(feed.xml_url)
        episodes = parse_feed_entries(
            text, feed.xml_url, feed.title, cutoff, max_episodes
        )
        return episodes, None
    except Exception as exc:
        log.warning("Feed fetch failed for %s: %s", feed.title, exc)
        return [], str(exc)


async def fetch_all_feeds(
    feeds: list[OPMLFeed],
    lookback_days: int,
    max_episodes_per_feed: int = 3,
    concurrency: int = 8,
) -> tuple[list[NormalizedEpisode], dict[str, str]]:
    """Fetch all feeds concurrently. Returns (all_episodes, {feed_title: error})."""
    semaphore = asyncio.Semaphore(concurrency)
    errors: dict[str, str] = {}
    all_episodes: list[NormalizedEpisode] = []

    async def bounded_fetch(feed: OPMLFeed) -> None:
        async with semaphore:
            episodes, err = await fetch_episodes_from_feed(
                feed, lookback_days, max_episodes_per_feed
            )
            if err:
                errors[feed.title] = err
            all_episodes.extend(episodes)

    await asyncio.gather(*[bounded_fetch(f) for f in feeds])
    return all_episodes, errors
