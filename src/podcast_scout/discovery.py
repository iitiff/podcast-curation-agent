"""Discovery-only ingestion layer — no OPML, no RSS polling.

All episodes are found via Podcast Index + web search queries derived
directly from the user's preferences.yaml.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from .config import DiscoveryConfig, Preferences
from .normalize import Enclosure, NormalizedEpisode, make_guid, utcnow
from .providers.base import (
    BaseLLMProvider,
    BasePodcastSearchProvider,
    BaseWebSearchProvider,
)

log = logging.getLogger(__name__)


def _build_queries(prefs: Preferences, cfg: DiscoveryConfig) -> list[str]:
    """Derive search queries purely from preferences — no static seeds needed."""
    queries: list[str] = []

    # One query per interest topic
    for topic in prefs.interests:
        readable = topic.replace("_", " ")
        queries.append(f"{readable} podcast episode")

    # One query per show prior (the shows you care about)
    for show in prefs.show_priors:
        queries.append(f"{show} podcast latest episode")

    # Guest watchlist
    for guest in prefs.guest_watchlist:
        queries.append(f"{guest} podcast interview")

    # Competitor / entity watchlist
    for entity in prefs.competitor_watchlist:
        queries.append(f"{entity} podcast strategy")

    # Entity seeds from discovery config (optional extras)
    for entity in (
        cfg.entity_seeds.get("competitors", [])
        + cfg.entity_seeds.get("ai_companies", [])
    )[:8]:
        queries.append(f"{entity} podcast")

    # Dedupe, preserve order, cap at max_queries
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    limit = cfg.discovery.max_queries if cfg else 40
    return unique[:limit]


async def _search_one(
    query: str,
    podcast_search: BasePodcastSearchProvider,
    max_results: int = 5,
) -> list[NormalizedEpisode]:
    results = []
    try:
        hits = await podcast_search.search_episodes(query, max_results=max_results)
        for r in hits:
            if not r.feed_url:
                continue
            ep = NormalizedEpisode(
                guid=make_guid(r.feed_url, r.episode_title or r.feed_url),
                source_feed_url=r.feed_url,
                original_guid=r.episode_title or r.feed_url,
                show_title=r.show_title,
                episode_title=r.episode_title or query,
                description=r.description,
                published=utcnow(),  # approximate; refined later if transcript available
                duration_seconds=r.duration_seconds,
                episode_url=r.episode_url,
                enclosure=Enclosure(url=r.enclosure_url) if r.enclosure_url else None,
                image_url=r.image_url,
                is_outside_feed=False,  # no concept of "outside" anymore
            )
            results.append(ep)
    except Exception as exc:
        log.warning("Podcast search failed for '%s': %s", query, exc)
    return results


async def discover_episodes(
    prefs: Preferences,
    cfg: DiscoveryConfig,
    podcast_search: BasePodcastSearchProvider,
    web_search: BaseWebSearchProvider,
    lookback_days: int = 8,
    concurrency: int = 8,
) -> list[NormalizedEpisode]:
    """Discover all candidate episodes from preferences. Returns deduplicated list."""
    queries = _build_queries(prefs, cfg)
    log.info("Running %d discovery queries", len(queries))

    semaphore = asyncio.Semaphore(concurrency)
    all_candidates: list[NormalizedEpisode] = []

    async def bounded(q: str) -> None:
        async with semaphore:
            eps = await _search_one(q, podcast_search)
            all_candidates.extend(eps)

    await asyncio.gather(*[bounded(q) for q in queries])

    # Dedupe by guid, cap at raw candidate limit
    seen_guids: set[str] = set()
    unique: list[NormalizedEpisode] = []
    max_raw = cfg.discovery.max_raw_candidates if cfg else 200
    for ep in all_candidates:
        if ep.guid not in seen_guids and len(unique) < max_raw:
            seen_guids.add(ep.guid)
            unique.append(ep)

    log.info("Discovery complete: %d unique candidates from %d queries", len(unique), len(queries))
    return unique
