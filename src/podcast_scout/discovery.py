"""Discovery-only ingestion layer — no OPML, no RSS polling.

All episodes are found via two complementary paths:

1. **Followed shows** (show_priors in preferences.yaml) — resolved to real
   RSS feed URLs and polled directly so episodes always appear with correct
   publish dates and are never lost due to stale search indices.

2. **Search-based discovery** — Podcast Index / iTunes keyword queries
   derived from the user's interests, guest watchlist, competitor watchlist,
   and entity seeds.  show_priors are intentionally excluded here because
   they are already covered by path 1.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import httpx

from .config import DiscoveryConfig, Preferences, ShowsConfig
from .feeds import fetch_feed_text, parse_feed_entries
from .normalize import Enclosure, NormalizedEpisode, make_guid, utcnow
from .opml import OPMLFeed
from .providers.base import (
    BasePodcastSearchProvider,
    BaseWebSearchProvider,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path 1 — Direct RSS polling for followed shows
# ---------------------------------------------------------------------------

async def _resolve_feed_url_itunes(show_name: str) -> str | None:
    """Look up a show's RSS feed URL via the iTunes Search API."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://itunes.apple.com/search",
                params={
                    "term": show_name,
                    "media": "podcast",
                    "entity": "podcast",
                    "limit": 1,
                },
            )
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            if results and results[0].get("feedUrl"):
                return results[0]["feedUrl"]
    except Exception as exc:
        log.warning("iTunes lookup failed for '%s': %s", show_name, exc)
    return None


async def _fetch_followed_show(
    show_name: str,
    feed_url: str,
    lookback_days: int,
    max_episodes: int = 3,
) -> list[NormalizedEpisode]:
    """Fetch recent episodes from a single followed show's RSS feed."""
    cutoff = utcnow() - timedelta(days=lookback_days)
    try:
        text = await fetch_feed_text(feed_url)
        episodes = parse_feed_entries(text, feed_url, show_name, cutoff, max_episodes)
        # Mark every episode as coming from a directly followed show
        for ep in episodes:
            ep.is_followed_show = True
        log.info(
            "Followed show '%s': fetched %d episode(s) via RSS",
            show_name,
            len(episodes),
        )
        return episodes
    except Exception as exc:
        log.warning("RSS fetch failed for followed show '%s' (%s): %s", show_name, feed_url, exc)
        return []


async def poll_followed_shows(
    prefs: Preferences,
    shows_cfg: ShowsConfig,
    lookback_days: int,
    concurrency: int = 8,
) -> list[NormalizedEpisode]:
    """Resolve each show_prior to a feed URL and poll its RSS feed directly.

    Resolution order for feed URL:
      1. canonical_feed_url from shows.yaml (exact match on show name)
      2. iTunes Search API lookup by show name

    Shows that cannot be resolved are skipped with a warning.
    """
    if not prefs.show_priors:
        return []

    # Build a name → canonical_feed_url map from shows.yaml
    canonical: dict[str, str] = {}
    for override in shows_cfg.shows:
        if override.canonical_feed_url and override.enabled:
            canonical[override.match.lower()] = override.canonical_feed_url

    semaphore = asyncio.Semaphore(concurrency)
    all_episodes: list[NormalizedEpisode] = []

    async def resolve_and_fetch(show_name: str) -> None:
        async with semaphore:
            # 1. Check shows.yaml first
            feed_url = canonical.get(show_name.lower())

            # 2. Fall back to iTunes lookup
            if not feed_url:
                log.debug("No canonical_feed_url for '%s' — trying iTunes lookup", show_name)
                feed_url = await _resolve_feed_url_itunes(show_name)

            if not feed_url:
                log.warning(
                    "Could not resolve feed URL for followed show '%s'. "
                    "Add a canonical_feed_url in config/shows.yaml to fix this.",
                    show_name,
                )
                return

            episodes = await _fetch_followed_show(show_name, feed_url, lookback_days)
            all_episodes.extend(episodes)

    await asyncio.gather(*[resolve_and_fetch(name) for name in prefs.show_priors])
    return all_episodes


# ---------------------------------------------------------------------------
# Path 2 — Search-based discovery
# ---------------------------------------------------------------------------

def _build_queries(prefs: Preferences, cfg: DiscoveryConfig) -> list[str]:
    """Derive search queries purely from preferences.

    Note: show_priors are intentionally excluded — those shows are polled
    directly via RSS (poll_followed_shows) so keyword search is redundant
    and unreliable for them.  We also avoid generic queries that pull in
    off-topic episodes; every query is anchored to a specific topic, guest,
    or competitor rather than just '{topic} podcast episode'.
    """
    # Build a lowercase set of all followed show names so we can filter
    # search results that come back from those shows (already covered by
    # path 1 above).
    followed_show_names: set[str] = {
        name.lower() for name in prefs.show_priors
    }

    queries: list[str] = []

    # One query per interest topic — but anchored to the persona's focus
    # area so the query is specific enough to avoid off-topic results.
    persona_anchor = prefs.persona.focus.split(",")[0].strip()  # first focus area
    for topic in prefs.interests:
        readable = topic.replace("_", " ")
        queries.append(f"{readable} {persona_anchor}")

    # Guest watchlist — direct name search for interviews
    for guest in prefs.guest_watchlist:
        queries.append(f"{guest} podcast interview")

    # Competitor / entity watchlist — strategy angle
    for entity in prefs.competitor_watchlist:
        queries.append(f"{entity} strategy podcast")

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
    followed_show_names: set[str],
    max_results: int = 5,
) -> list[NormalizedEpisode]:
    results = []
    try:
        hits = await podcast_search.search_episodes(query, max_results=max_results)
        for r in hits:
            if not r.feed_url:
                continue
            # Skip episodes from shows already covered by direct RSS polling
            if r.show_title and r.show_title.lower() in followed_show_names:
                log.debug(
                    "Search discovery: skipping '%s' — already a followed show",
                    r.show_title,
                )
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
                is_outside_feed=False,
            )
            results.append(ep)
    except Exception as exc:
        log.warning("Podcast search failed for '%s': %s", query, exc)
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def discover_episodes(
    prefs: Preferences,
    cfg: DiscoveryConfig,
    podcast_search: BasePodcastSearchProvider,
    web_search: BaseWebSearchProvider,
    lookback_days: int = 8,
    concurrency: int = 8,
    shows_cfg: ShowsConfig | None = None,
) -> list[NormalizedEpisode]:
    """Discover all candidate episodes.  Returns a deduplicated list.

    Followed shows (show_priors) are fetched first via direct RSS polling
    so they always appear at the front of the candidate list with real
    publish timestamps.  Search-based discovery fills the rest.
    """
    if shows_cfg is None:
        shows_cfg = ShowsConfig()

    # --- Path 1: directly polled followed shows ---
    followed_episodes = await poll_followed_shows(
        prefs, shows_cfg, lookback_days, concurrency
    )
    log.info(
        "Followed-show RSS poll: %d episode(s) from %d show(s)",
        len(followed_episodes),
        len(prefs.show_priors),
    )

    # Build a set of followed show names for search-result filtering
    followed_show_names: set[str] = {name.lower() for name in prefs.show_priors}

    # --- Path 2: search-based discovery ---
    queries = _build_queries(prefs, cfg)
    log.info("Running %d search discovery queries", len(queries))

    semaphore = asyncio.Semaphore(concurrency)
    search_candidates: list[NormalizedEpisode] = []

    async def bounded(q: str) -> None:
        async with semaphore:
            eps = await _search_one(q, podcast_search, followed_show_names)
            search_candidates.extend(eps)

    await asyncio.gather(*[bounded(q) for q in queries])

    # --- Merge: followed shows first, then search candidates ---
    # Dedupe by guid across both lists; followed-show episodes take priority.
    seen_guids: set[str] = set()
    unique: list[NormalizedEpisode] = []
    max_raw = cfg.discovery.max_raw_candidates if cfg else 200

    for ep in followed_episodes + search_candidates:
        if ep.guid not in seen_guids and len(unique) < max_raw:
            seen_guids.add(ep.guid)
            unique.append(ep)

    log.info(
        "Discovery complete: %d unique candidates (%d followed + %d from search)",
        len(unique),
        sum(1 for e in unique if e.is_followed_show),
        sum(1 for e in unique if not e.is_followed_show),
    )
    return unique
