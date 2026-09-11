"""Non-podcast source adapters: trade press, vendor blogs, research feeds.

The radar was podcast-only, and not by accident: every discovery seed was
phrased "... podcast" and discovery ran through podcast directories. A question
like "what comes after next-best-action" is answered far better by a paper or
an engineering blog than by any interview, so the radar has to be able to reach
somewhere other than iTunes.

Articles are the same shape as episodes minus an enclosure and a duration, so
they flow through the existing filter and get judged by the same evergreen
rubric. `NormalizedEpisode.is_playable` keeps them out of the audio feeds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import yaml

from .edgar import EdgarConfig, parse_edgar_config
from .feeds import parse_feed_entries
from .normalize import NormalizedEpisode, utcnow

log = logging.getLogger(__name__)

# Default trust per source class, applied when a source does not state its own.
#
# Vendor material is a PATTERN source, never neutral evidence: it exists to
# sell something, and the design docs are explicit that its performance claims
# need corroboration from earnings, independent case studies or engineering
# detail. Earnings is the only class that can contradict it, which is why it
# starts highest.
CLASS_DEFAULTS: dict[str, tuple[str, str]] = {
    "earnings-call": (
        "high",
        "Management commentary: selective in emphasis, but legally constrained "
        "on fact. The one class that can contradict vendor marketing.",
    ),
    "research-paper": (
        "high",
        "Peer-reviewed or preprint. Strong on mechanism, weak on whether "
        "anything works at production scale.",
    ),
    "trade-press": (
        "medium",
        "Access-driven reporting: tends to amplify whoever granted the "
        "interview, and rarely revisits a prediction that did not land.",
    ),
    "vendor": (
        "low",
        "Vendor marketing. Use as a pattern source only; discount every "
        "performance claim until corroborated elsewhere.",
    ),
}


@dataclass
class ArticleSource:
    name: str
    url: str
    source_type: str = "trade-press"
    credibility: str = ""
    bias_notes: str = ""
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    max_items_per_run: int = 5

    def resolved(self) -> tuple[str, str]:
        """Credibility and bias, falling back to the class default."""
        default_cred, default_bias = CLASS_DEFAULTS.get(
            self.source_type, ("medium", "")
        )
        return (self.credibility or default_cred, self.bias_notes or default_bias)


@dataclass
class SourcesConfig:
    sources: list[ArticleSource] = field(default_factory=list)
    # Earnings does not arrive by RSS -- most IR sites publish no feed at all --
    # so it is configured as companies rather than URLs and fetched from EDGAR.
    # See edgar.py.
    edgar: EdgarConfig = field(default_factory=EdgarConfig)


def load_sources(config_dir: Path, sec_user_agent: str = "") -> SourcesConfig:
    """Load config/sources.yaml. Absent file means podcasts only, as before."""
    path = config_dir / "sources.yaml"
    if not path.exists():
        return SourcesConfig()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = [
        ArticleSource(
            name=item.get("name", item.get("url", "unnamed")),
            url=item["url"],
            source_type=item.get("source_type", "trade-press"),
            credibility=item.get("credibility", ""),
            bias_notes=item.get("bias_notes", ""),
            tags=item.get("tags", []),
            enabled=item.get("enabled", True),
            max_items_per_run=item.get("max_items_per_run", 5),
        )
        for item in raw.get("sources", [])
        if item.get("url")
    ]
    return SourcesConfig(
        sources=sources,
        edgar=parse_edgar_config(raw, user_agent=sec_user_agent),
    )


async def fetch_articles(
    config: SourcesConfig,
    lookback_days: int,
    client: httpx.AsyncClient | None = None,
) -> list[NormalizedEpisode]:
    """Fetch recent items from every enabled article source.

    Reuses parse_feed_entries rather than reimplementing feed handling: it
    already does the date cutoff, title/link/guid fallbacks and enclosure
    detection, and a second parser would drift from the first.

    One source failing must never take down the run. A blog that 404s or serves
    malformed XML is logged and skipped, exactly as an unreachable podcast feed
    already is.
    """
    enabled = [s for s in config.sources if s.enabled]
    if not enabled:
        return []

    cutoff = utcnow() - timedelta(days=lookback_days)
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    items: list[NormalizedEpisode] = []

    try:
        for source in enabled:
            try:
                resp = await client.get(source.url)
                if resp.status_code != 200:
                    log.warning(
                        "Source '%s' returned %s — skipping.",
                        source.name, resp.status_code,
                    )
                    continue
                parsed = parse_feed_entries(
                    resp.text,
                    feed_url=source.url,
                    show_name_override=source.name,
                    cutoff=cutoff,
                    max_entries=source.max_items_per_run,
                )
            except Exception as exc:
                log.warning("Source '%s' failed: %s", source.name, exc)
                continue

            credibility, bias = source.resolved()
            for item in parsed[: source.max_items_per_run]:
                # An article feed occasionally carries an enclosure (a podcast
                # cross-post). Cleared so it cannot leak into the audio feeds
                # under a non-podcast source_type.
                if source.source_type != "podcast":
                    item.enclosure = None
                    item.duration_seconds = 0
                item.source_type = source.source_type
                item.credibility = credibility
                item.bias_notes = bias
                items.append(item)
            log.info(
                "Source '%s' (%s): %d item(s)",
                source.name, source.source_type, len(parsed[: source.max_items_per_run]),
            )
    finally:
        if owns_client:
            await client.aclose()

    return items


# ---------------------------------------------------------------------------
# Questions drive the radar.
#
# The evergreen seeds all end in "podcast", so no query they generate can ever
# surface a paper however relevant it is. That is the filter's phrasing leaking
# into the radar: the filter decides what is worth reading, the radar decides
# where to look, and they should not share vocabulary.
# ---------------------------------------------------------------------------

# Which source classes a question reaches for, by tag. A question about
# architecture is answered by papers and engineering writing; a competitive one
# by earnings and trade press.
TAG_SOURCE_AFFINITY: dict[str, tuple[str, ...]] = {
    "architecture": ("research-paper", "vendor"),
    "research": ("research-paper",),
    "competitive": ("earnings-call", "trade-press"),
    "market": ("earnings-call", "trade-press"),
    "vendor": ("vendor", "trade-press"),
}


def queries_for_questions(
    questions: list[Any],
    max_per_question: int = 2,
) -> list[str]:
    """Turn open questions into discovery queries.

    Deliberately plain: the question text itself is the best query, because it
    is already how its holder phrases the problem. A second, keyword-stripped
    variant is added because search engines handle noun phrases better than
    full interrogatives.

    No "podcast" suffix. That word is what confined the radar to one medium.
    """
    queries: list[str] = []
    for question in questions:
        text = (getattr(question, "question", "") or getattr(question, "title", "")).strip()
        if not text:
            continue
        queries.append(text.rstrip("?"))
        if max_per_question > 1:
            # Strip the interrogative opener: "What comes after next-best-action"
            # searches worse than "after next-best-action".
            stripped = text.rstrip("?")
            for opener in ("what comes after ", "what is ", "what are ", "how is ",
                           "how are ", "how do ", "why is ", "why do ", "is "):
                if stripped.lower().startswith(opener):
                    stripped = stripped[len(opener):]
                    break
            if stripped and stripped.lower() != text.rstrip("?").lower():
                queries.append(stripped)
    return queries


def sources_for_questions(
    questions: list[Any],
    config: SourcesConfig,
) -> SourcesConfig:
    """Narrow the source list to the classes the open questions call for.

    Returns everything when no question carries a recognised tag: an untagged
    question is not a reason to read less.
    """
    wanted: set[str] = set()
    for question in questions:
        for tag in getattr(question, "tags", []) or []:
            wanted.update(TAG_SOURCE_AFFINITY.get(str(tag).lower(), ()))
    if not wanted:
        return config
    return SourcesConfig(
        sources=[s for s in config.sources if s.source_type in wanted],
        # Earnings is a source class like any other: if no open question calls
        # for it, do not spend the requests. Carried explicitly because it does
        # not live in the `sources` list the filter above walks.
        edgar=config.edgar if "earnings-call" in wanted else EdgarConfig(
            user_agent=config.edgar.user_agent
        ),
    )
