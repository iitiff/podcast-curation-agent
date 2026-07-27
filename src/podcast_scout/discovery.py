"""Discovery-only ingestion layer.

All episodes are found via two complementary paths:

1. **Followed shows** (show_priors in preferences.yaml) — resolved to real
   RSS feed URLs and polled directly so episodes always appear with correct
   publish dates and are never lost due to stale search indices.

2. **Search-based discovery** — Podcast Index / iTunes keyword queries
   derived from the persona focus, guest watchlist, and competitor watchlist.
   show_priors are intentionally excluded here because they are already
   covered by path 1.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import timedelta

import httpx

from .config import DiscoveryConfig, Preferences, ShowsConfig
from .feeds import fetch_feed_text, parse_feed_entries
from .normalize import Enclosure, NormalizedEpisode, make_guid, utcnow
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
                return results[0]["feedUrl"]  # type: ignore[no-any-return]
    except Exception as exc:
        log.warning("iTunes lookup failed for '%s': %s", show_name, exc)
    return None


async def _fetch_followed_show_via_podcast_index(
    show_name: str,
    podcast_search: BasePodcastSearchProvider,
    lookback_days: int,
    max_episodes: int = 3,
) -> list[NormalizedEpisode]:
    """Fallback: fetch recent episodes via Podcast Index search when RSS is blocked (e.g. 403)."""
    cutoff = utcnow() - timedelta(days=lookback_days)
    try:
        hits = await podcast_search.search_episodes(show_name, max_results=max_episodes * 2)
        episodes: list[NormalizedEpisode] = []
        for r in hits:
            # Only keep results that look like they belong to this show
            if r.show_title and show_name.lower() not in r.show_title.lower():
                continue
            ep = NormalizedEpisode(
                guid=make_guid(r.feed_url or show_name, r.episode_title or show_name),
                source_feed_url=r.feed_url or "",
                original_guid=r.episode_title or show_name,
                show_title=r.show_title or show_name,
                episode_title=r.episode_title or "",
                description=r.description,
                published=utcnow(),
                duration_seconds=r.duration_seconds,
                episode_url=r.episode_url,
                enclosure=Enclosure(url=r.enclosure_url) if r.enclosure_url else None,
                image_url=r.image_url,
                is_followed_show=True,
                is_outside_feed=False,
            )
            if ep.published >= cutoff:
                episodes.append(ep)
            if len(episodes) >= max_episodes:
                break
        log.info(
            "Followed show '%s': fetched %d episode(s) via Podcast Index fallback",
            show_name,
            len(episodes),
        )
        return episodes
    except Exception as exc:
        log.warning("Podcast Index fallback failed for followed show '%s': %s", show_name, exc)
        return []


async def _fetch_followed_show(
    show_name: str,
    feed_url: str,
    lookback_days: int,
    max_episodes: int = 3,
    podcast_search: BasePodcastSearchProvider | None = None,
) -> list[NormalizedEpisode]:
    """Fetch recent episodes from a single followed show's RSS feed.

    On a 403 response (e.g. Substack blocking CI IPs), falls back to
    Podcast Index episode search if a provider is supplied.
    """
    cutoff = utcnow() - timedelta(days=lookback_days)
    try:
        text = await fetch_feed_text(feed_url)
        episodes = parse_feed_entries(text, feed_url, show_name, cutoff, max_episodes)
        for ep in episodes:
            ep.is_followed_show = True
        log.info(
            "Followed show '%s': fetched %d episode(s) via RSS",
            show_name,
            len(episodes),
        )
        return episodes
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403 and podcast_search is not None:
            log.warning(
                "RSS fetch blocked (403) for followed show '%s' (%s) — trying Podcast Index fallback",
                show_name,
                feed_url,
            )
            return await _fetch_followed_show_via_podcast_index(
                show_name, podcast_search, lookback_days, max_episodes
            )
        log.warning("RSS fetch failed for followed show '%s' (%s): %s", show_name, feed_url, exc)
        return []
    except Exception as exc:
        log.warning("RSS fetch failed for followed show '%s' (%s): %s", show_name, feed_url, exc)
        return []


async def poll_followed_shows(
    prefs: Preferences,
    shows_cfg: ShowsConfig,
    lookback_days: int,
    concurrency: int = 8,
    podcast_search: BasePodcastSearchProvider | None = None,
) -> list[NormalizedEpisode]:
    """Resolve each show_prior to a feed URL and poll its RSS feed directly."""
    if not prefs.show_priors:
        return []

    canonical: dict[str, str] = {}
    for override in shows_cfg.shows:
        if override.canonical_feed_url and override.enabled:
            canonical[override.match.lower()] = override.canonical_feed_url

    semaphore = asyncio.Semaphore(concurrency)
    all_episodes: list[NormalizedEpisode] = []

    async def resolve_and_fetch(show_name: str) -> None:
        async with semaphore:
            feed_url = canonical.get(show_name.lower())
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
            episodes = await _fetch_followed_show(
                show_name, feed_url, lookback_days,
                podcast_search=podcast_search,
            )
            all_episodes.extend(episodes)

    await asyncio.gather(*[resolve_and_fetch(name) for name in prefs.show_priors])
    return all_episodes


# ---------------------------------------------------------------------------
# Path 2 — Search-based discovery
# ---------------------------------------------------------------------------

_FOCUS_SPLIT_RE = re.compile(r"[,\u2014\u2013]")


def _build_queries(prefs: Preferences, cfg: DiscoveryConfig) -> list[str]:
    """Derive search queries from persona focus, guest watchlist, and competitor watchlist."""
    queries: list[str] = []

    raw_focus_parts = _FOCUS_SPLIT_RE.split(prefs.persona.focus)
    focus_areas: list[str] = []
    for part in raw_focus_parts:
        part = part.strip()
        if part and not re.match(r"(?i)^(for a|for an|growing|as a)", part):
            focus_areas.append(part)

    for area in focus_areas[:6]:
        queries.append(f"{area} podcast")

    for guest in prefs.guest_watchlist:
        queries.append(f"{guest} podcast interview")

    for entity in prefs.competitor_watchlist:
        queries.append(f"{entity} strategy podcast")

    for entity in (
        cfg.entity_seeds.get("competitors", [])
        + cfg.entity_seeds.get("ai_companies", [])
    )[:8]:
        queries.append(f"{entity} podcast")

    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    limit = cfg.max_queries if cfg else 40
    return unique[:limit]


async def _search_one(
    query: str,
    podcast_search: BasePodcastSearchProvider,
    followed_show_names: set[str],
    max_results: int = 5,
) -> list[NormalizedEpisode]:
    results: list[NormalizedEpisode] = []
    try:
        hits = await podcast_search.search_episodes(query, max_results=max_results)
        for r in hits:
            if not r.feed_url:
                continue
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
                published=utcnow(),
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


async def discover_episodes(
    prefs: Preferences,
    cfg: DiscoveryConfig,
    podcast_search: BasePodcastSearchProvider,
    web_search: BaseWebSearchProvider,
    lookback_days: int = 8,
    concurrency: int = 8,
    shows_cfg: ShowsConfig | None = None,
) -> list[NormalizedEpisode]:
    """Discover all candidate episodes. Returns a deduplicated list."""
    if shows_cfg is None:
        shows_cfg = ShowsConfig()

    followed_episodes = await poll_followed_shows(
        prefs, shows_cfg, lookback_days, concurrency,
        podcast_search=podcast_search,
    )
    log.info(
        "Followed-show RSS poll: %d episode(s) from %d show(s)",
        len(followed_episodes),
        len(prefs.show_priors),
    )

    followed_show_names: set[str] = {name.lower() for name in prefs.show_priors}

    queries = _build_queries(prefs, cfg)
    log.info("Running %d search discovery queries", len(queries))

    semaphore = asyncio.Semaphore(concurrency)
    search_candidates: list[NormalizedEpisode] = []

    async def bounded(q: str) -> None:
        async with semaphore:
            eps = await _search_one(q, podcast_search, followed_show_names)
            search_candidates.extend(eps)

    await asyncio.gather(*[bounded(q) for q in queries])

    seen_guids: set[str] = set()
    unique: list[NormalizedEpisode] = []
    max_raw = cfg.max_raw_candidates if cfg else 200

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
