"""Bounded outside-feed discovery layer."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .config import DiscoveryConfig, Preferences, Settings
from .normalize import NormalizedEpisode

log = logging.getLogger(__name__)


class DiscoveryProvider:
    """Abstract base for podcast search providers."""

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        raise NotImplementedError


class PodcastIndexProvider(DiscoveryProvider):
    def __init__(self, api_key: str, api_secret: str) -> None:
        self._key = api_key
        self._secret = api_secret

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        import hashlib
        import time
        import httpx

        epoch = int(time.time())
        auth_hash = hashlib.sha1(
            f"{self._key}{self._secret}{epoch}".encode()
        ).hexdigest()
        headers = {
            "X-Auth-Key": self._key,
            "X-Auth-Date": str(epoch),
            "Authorization": auth_hash,
            "User-Agent": "PodcastScout/0.1",
        }
        try:
            with httpx.Client(timeout=15) as client:
                r = client.get(
                    "https://api.podcastindex.org/api/1.0/search/byterm",
                    params={"q": query, "max": max_results, "clean": True},
                    headers=headers,
                )
                r.raise_for_status()
                return r.json().get("feeds", [])
        except Exception as exc:
            log.warning("PodcastIndex search failed for '%s': %s", query, exc)
            return []


class BraveSearchProvider:
    def __init__(self, api_key: str) -> None:
        self._key = api_key

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        import httpx
        try:
            with httpx.Client(timeout=15) as client:
                r = client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": f"{query} podcast episode site:podcastindex.org OR site:open.spotify.com", "count": max_results},
                    headers={"Accept": "application/json", "X-Subscription-Token": self._key},
                )
                r.raise_for_status()
                return r.json().get("web", {}).get("results", [])
        except Exception as exc:
            log.warning("Brave search failed for '%s': %s", query, exc)
            return []


def build_dynamic_queries(
    followed_episodes: list[NormalizedEpisode],
    prefs: Preferences,
    discovery_cfg: DiscoveryConfig,
) -> list[str]:
    """Build discovery queries from this week's episode topics + static seeds."""
    queries: list[str] = []

    # Static seeds
    for seed in discovery_cfg.static_seeds:
        if seed.get("enabled", True):
            queries.append(seed["query"])

    # Dynamic: extract keywords from followed-feed episode titles this week
    all_keywords: list[str] = []
    for ep in followed_episodes[:20]:
        all_keywords.extend(ep.keywords[:3])
        # Pull meaningful words from title
        words = [w for w in ep.episode_title.split() if len(w) > 4]
        all_keywords.extend(words[:3])

    # Deduplicate and build compound queries from entity seeds
    for entity in discovery_cfg.entity_seeds.get("competitors", [])[:5]:
        queries.append(f"{entity} strategy podcast episode 2025")
    for topic in discovery_cfg.entity_seeds.get("topic_areas", [])[:5]:
        queries.append(f"{topic} podcast expert interview")

    max_q = discovery_cfg.discovery.max_queries
    return list(dict.fromkeys(queries))[:max_q]  # dedup, cap


def discover_outside_episodes(
    followed_episodes: list[NormalizedEpisode],
    prefs: Preferences,
    settings: Settings,
    discovery_cfg: DiscoveryConfig,
    since: datetime,
) -> list[NormalizedEpisode]:
    """Run bounded outside-feed discovery. Returns raw candidates."""
    if not settings.podcast_index_key and not settings.web_search_api_key:
        log.info("No discovery API keys configured — skipping outside-feed discovery")
        return []

    provider: DiscoveryProvider | None = None
    if settings.podcast_index_key:
        provider = PodcastIndexProvider(settings.podcast_index_key, settings.podcast_index_secret)
    # Web search provider not used for feed discovery directly — used for validation

    if not provider:
        return []

    queries = build_dynamic_queries(followed_episodes, prefs, discovery_cfg)
    log.info("Running %d outside-feed discovery queries", len(queries))

    from .feeds import fetch_feed
    from .config import ShowsConfig
    seen_urls: set[str] = {ep.source_feed_url for ep in followed_episodes}
    candidates: list[NormalizedEpisode] = []
    max_raw = discovery_cfg.discovery.max_raw_candidates

    for query in queries:
        if len(candidates) >= max_raw:
            break
        feeds = provider.search(query, max_results=5)
        for feed_info in feeds:
            feed_url = feed_info.get("url") or feed_info.get("feedUrl") or ""
            if not feed_url or feed_url in seen_urls:
                continue
            show_title = feed_info.get("title") or feed_info.get("titleOriginal") or "Unknown"
            seen_urls.add(feed_url)
            eps = fetch_feed(
                feed_url=feed_url,
                show_title=show_title,
                since=since,
                settings=settings,
                shows_config=ShowsConfig(),
                is_outside=True,
            )
            candidates.extend(eps)
            if len(candidates) >= max_raw:
                break

    log.info("Outside-feed discovery found %d raw candidates", len(candidates))
    return candidates[:max_raw]
