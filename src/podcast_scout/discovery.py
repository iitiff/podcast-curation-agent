"""Bounded outside-feed discovery layer."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import feedparser
import httpx

from .normalize import Enclosure, NormalizedEpisode, make_guid, parse_duration, utcnow
from .providers.base import BasePodcastSearchProvider, BaseWebSearchProvider

if TYPE_CHECKING:
    from .config import DiscoveryConfig, Preferences

logger = logging.getLogger(__name__)


def build_discovery_queries(
    prefs: "Preferences",
    discovery_cfg: "DiscoveryConfig",
    dynamic_topics: list[str] | None = None,
) -> list[str]:
    """Combine static seeds with dynamic topics from this week's scan."""
    queries: list[str] = []

    # Static seeds
    for seed in discovery_cfg.static_seeds:
        if seed.get("enabled", True):
            queries.append(seed["query"])

    # Dynamic topics from current run
    if dynamic_topics:
        for topic in dynamic_topics[:5]:
            queries.append(f"{topic} podcast episode")

    # Entity seeds
    for entity in discovery_cfg.entity_seeds.get("competitors", [])[:3]:
        queries.append(f"{entity} strategy podcast 2025")
    for entity in discovery_cfg.entity_seeds.get("ai_companies", [])[:2]:
        queries.append(f"{entity} product AI podcast")

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    return deduped[: discovery_cfg.discovery.max_queries]


async def discover_outside_episodes(
    queries: list[str],
    podcast_search: BasePodcastSearchProvider,
    web_search: BaseWebSearchProvider,
    discovery_cfg: "DiscoveryConfig",
    lookback_days: int,
    client: httpx.AsyncClient,
) -> list[NormalizedEpisode]:
    """Run discovery queries and return raw candidate episodes."""
    raw_candidates: list[dict[str, Any]] = []
    limits = discovery_cfg.discovery

    for query in queries[: limits.max_queries]:
        try:
            results = await podcast_search.search_episodes(query, max_results=5)
            raw_candidates.extend(results)
        except Exception as exc:
            logger.warning("Podcast search failed for '%s': %s", query, exc)

        if len(raw_candidates) >= limits.max_raw_candidates:
            break

    # Convert raw candidates to NormalizedEpisode objects
    episodes: list[NormalizedEpisode] = []
    seen_feed_urls: set[str] = set()
    cutoff = utcnow().replace(tzinfo=None)

    for candidate in raw_candidates[: limits.max_raw_candidates]:
        feed_url = candidate.get("url") or candidate.get("feedUrl", "")
        if not feed_url or feed_url in seen_feed_urls:
            continue
        seen_feed_urls.add(feed_url)

        # Fetch the feed to get actual episodes
        try:
            resp = await client.get(feed_url, follow_redirects=True, timeout=15.0)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.text)
        except Exception as exc:
            logger.debug("Failed to fetch discovery feed %s: %s", feed_url, exc)
            continue

        show_title = (
            getattr(parsed.feed, "title", "")
            or candidate.get("title", "Unknown Show")
        ).strip()

        for entry in parsed.entries[:3]:
            original_guid = getattr(entry, "id", "") or getattr(entry, "link", feed_url)
            guid = make_guid(feed_url, original_guid)

            enclosure = None
            for enc in getattr(entry, "enclosures", []):
                url = getattr(enc, "href", "") or getattr(enc, "url", "")
                if url:
                    enclosure = Enclosure(
                        url=url,
                        mime_type=getattr(enc, "type", "audio/mpeg"),
                        length=int(getattr(enc, "length", 0) or 0),
                    )
                    break

            episodes.append(
                NormalizedEpisode(
                    guid=guid,
                    source_feed_url=feed_url,
                    original_guid=original_guid,
                    show_title=show_title,
                    episode_title=(
                        getattr(entry, "title", "Untitled") or "Untitled"
                    ).strip(),
                    description=(
                        getattr(entry, "summary", "") or ""
                    )[:3000],
                    published=utcnow(),
                    duration_seconds=parse_duration(
                        str(getattr(entry, "itunes_duration", "") or "")
                    ),
                    episode_url=getattr(entry, "link", "") or "",
                    enclosure=enclosure,
                    is_outside_feed=True,
                )
            )

        if len(episodes) >= limits.max_deep_analysis_candidates:
            break

    logger.info("Discovery found %d outside candidates", len(episodes))
    return episodes
