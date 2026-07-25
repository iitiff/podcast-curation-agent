"""Bounded outside-feed discovery layer."""
from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from .config import DiscoveryConfig, Preferences
from .normalize import NormalizedEpisode, utcnow
from .providers.base import (
    BaseLLMProvider, BasePodcastSearchProvider, BaseWebSearchProvider, LLMMessage,
)
from .ranking import RankedEpisode

log = logging.getLogger(__name__)


def _build_queries(
    cfg: DiscoveryConfig,
    prefs: Preferences,
    weekly_topics: list[str],
    max_queries: int,
) -> list[str]:
    """Combine static seeds with dynamic topic queries."""
    queries: list[str] = []

    # Static seeds
    for seed in cfg.static_seeds:
        if seed.get("enabled", True):
            queries.append(seed["query"])

    # Dynamic from weekly topics
    for topic in weekly_topics[:5]:
        queries.append(f"{topic} podcast episode 2025")

    # Entity seeds
    for entity in (cfg.entity_seeds.get("competitors", []) + cfg.entity_seeds.get("ai_companies", []))[:6]:
        queries.append(f"{entity} podcast interview strategy")

    # Dedupe and cap
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:max_queries]


async def discover_outside_episodes(
    cfg: DiscoveryConfig,
    prefs: Preferences,
    llm: BaseLLMProvider,
    podcast_search: BasePodcastSearchProvider,
    web_search: BaseWebSearchProvider,
    followed_feed_urls: set[str],
    weekly_topics: list[str],
    lookback_days: int = 8,
) -> list[NormalizedEpisode]:
    """Run bounded discovery. Returns raw candidates (not yet ranked)."""
    limits = cfg.discovery
    queries = _build_queries(cfg, prefs, weekly_topics, limits.max_queries)

    candidates: list[NormalizedEpisode] = []
    cutoff = utcnow() - timedelta(days=lookback_days)

    async def run_query(query: str) -> None:
        # Try podcast search first
        try:
            results = await podcast_search.search_episodes(query, max_results=5)
            for r in results:
                if r.feed_url in followed_feed_urls:
                    continue  # already covered
                if not r.feed_url:
                    continue
                from .normalize import make_guid, Enclosure
                ep = NormalizedEpisode(
                    guid=make_guid(r.feed_url, r.episode_title or r.feed_url),
                    source_feed_url=r.feed_url,
                    original_guid=r.episode_title or r.feed_url,
                    show_title=r.show_title,
                    episode_title=r.episode_title or query,
                    description=r.description,
                    published=utcnow(),  # approximate
                    duration_seconds=r.duration_seconds,
                    episode_url=r.episode_url,
                    enclosure=Enclosure(url=r.enclosure_url) if r.enclosure_url else None,
                    image_url=r.image_url,
                    is_outside_feed=True,
                )
                candidates.append(ep)
        except Exception as exc:
            log.warning("Podcast search failed for query '%s': %s", query, exc)

    semaphore = asyncio.Semaphore(5)

    async def bounded(q: str) -> None:
        async with semaphore:
            await run_query(q)

    await asyncio.gather(*[bounded(q) for q in queries])

    # Dedupe by guid, cap at max_raw
    seen_guids: set[str] = set()
    unique: list[NormalizedEpisode] = []
    for ep in candidates:
        if ep.guid not in seen_guids and len(unique) < limits.max_raw_candidates:
            seen_guids.add(ep.guid)
            unique.append(ep)

    log.info("Outside discovery: %d raw candidates from %d queries", len(unique), len(queries))
    return unique[:limits.max_deep_analysis_candidates]
