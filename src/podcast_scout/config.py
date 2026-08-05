"""Settings and config loaders."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FeedConfig:
    title: str = "My Podcast Scout"
    description: str = "A ranked weekly listening queue curated by AI."
    owner_name: str = ""
    owner_email: str = ""
    language: str = "en-us"
    base_url: str = ""


@dataclass
class CategoryFeedConfig:
    slug: str
    title: str
    description: str = ""
    max_listen_fully: int = 3
    max_read_summary: int = 5


@dataclass
class PersonaConfig:
    role: str = "Senior product leader"
    focus: str = "AI, retail, eCommerce"
    seniority: str = "senior"
    preferred_depth: str = "strategic"


@dataclass
class LengthConfig:
    preferred_min_minutes: int = 20
    preferred_max_minutes: int = 90
    hard_max_minutes: int = 240
    max_weekly_listen_hours: float = 4.0


@dataclass
class ClassificationConfig:
    listen_fully_min_score: float = 75.0
    read_summary_min_score: float = 50.0
    boundary_override_max: float = 5.0


@dataclass
class OutputCapsConfig:
    max_listen_fully: int = 3
    max_read_summary: int = 5
    max_outside_feed: int = 3
    max_total_surfaced: int = 10


@dataclass
class Preferences:
    feed: FeedConfig = field(default_factory=FeedConfig)
    categories: dict[str, CategoryFeedConfig] = field(default_factory=dict)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    geography: dict[str, float] = field(default_factory=dict)
    length: LengthConfig = field(default_factory=LengthConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    output_caps: OutputCapsConfig = field(default_factory=OutputCapsConfig)
    show_priors: dict[str, float] = field(default_factory=dict)
    topic_exclusions: list[str] = field(default_factory=list)
    guest_watchlist: list[str] = field(default_factory=list)
    competitor_watchlist: list[str] = field(default_factory=list)


@dataclass
class DiscoveryConfig:
    max_queries: int = 15
    max_raw_candidates: int = 40
    max_deep_analysis_candidates: int = 10
    max_surfaced_outside_episodes: int = 3
    outside_quality_threshold: float = 60.0
    static_seeds: list[dict[str, Any]] = field(default_factory=list)
    entity_seeds: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ShowConfig:
    match: str
    canonical_feed_url: str | None = None
    display_name: str | None = None
    priority: float | None = None
    enabled: bool = True
    language: str | None = None
    transcript_source: str | None = None
    max_episodes_per_run: int | None = None
    category: str | None = None
    notes: str | None = None


@dataclass
class ShowsConfig:
    shows: list[ShowConfig] = field(default_factory=list)



def _env(name: str, default: str = "") -> str:
    """Read an env var with surrounding whitespace stripped.

    GitHub Actions injects secrets verbatim, so a value pasted with a stray
    leading/trailing space or newline arrives with it intact. That has already
    caused two production bugs in this repo:

      - SMTP_USER / SMTP_PASSWORD containing U+00A0 broke smtplib's AUTH
        (see cli._ascii_clean).
      - PAGES_BASE_URL with a leading space emitted
        `<atom:link href=" https://...">` into every generated feed. Some
        podcast clients use that self-link when refreshing and reject the
        malformed URL, which presents as "the feed never updates".

    A var that is unset OR whitespace-only is treated as absent, so the default
    applies rather than an empty string silently propagating.
    """
    return (os.getenv(name) or "").strip() or default


class Settings:
    def __init__(self) -> None:
        self.config_dir = Path(_env("CONFIG_DIR", "config"))
        self.data_dir = Path(_env("DATA_DIR", "data"))
        self.public_dir = Path(_env("PUBLIC_DIR", "public"))
        self.templates_dir = Path("src/podcast_scout/templates")

        # GITHUB_TOKEN is still read because the workflow uses it for git
        # commit/push, but it is NO LONGER an LLM credential: GitHub Models was
        # permanently retired 2026-07-30 and its endpoint returns 410 Gone.
        self.github_token = _env("GITHUB_TOKEN")

        # ---- LLM providers ------------------------------------------------
        # PRIMARY: Gemini. Chosen for strongest adherence to the strict
        # "return ONLY a raw JSON array" contract that stage2_batch_rank parses,
        # plus large context headroom for batched episodes.
        self.gemini_api_key = _env("GEMINI_API_KEY")
        self.gemini_stage1_model = _env("GEMINI_STAGE1_MODEL", "gemini-2.5-flash")
        self.gemini_stage2_model = _env("GEMINI_STAGE2_MODEL", "gemini-2.5-flash")

        # FALLBACK: any OpenAI-compatible endpoint. Defaults target NVIDIA's
        # hosted NIM API (build.nvidia.com). Deliberately generic -- to switch to
        # OpenRouter / Groq / Together / a self-hosted NIM, change only
        # LLM_FALLBACK_BASE_URL and LLM_FALLBACK_MODEL, no code edits required.
        # Legacy NVIDIA_* names are still honoured for convenience.
        self.fallback_api_key = _env("LLM_FALLBACK_API_KEY") or _env("NVIDIA_API_KEY")
        self.fallback_base_url = (
            _env("LLM_FALLBACK_BASE_URL")
            or _env("NVIDIA_BASE_URL")
            or "https://integrate.api.nvidia.com/v1"
        )
        self.fallback_model = (
            _env("LLM_FALLBACK_MODEL")
            or _env("NVIDIA_MODEL")
            or "meta/llama-3.3-70b-instruct"
        )
        self.fallback_provider_name = _env("LLM_FALLBACK_NAME", "NVIDIA NIM")

        self.podcast_index_key = _env("PODCAST_INDEX_KEY")
        self.podcast_index_secret = _env("PODCAST_INDEX_SECRET")
        self.web_search_api_key = _env("WEB_SEARCH_API_KEY")
        self.web_search_provider = _env("WEB_SEARCH_PROVIDER", "brave")
        self.enable_audio_transcription = _env("ENABLE_AUDIO_TRANSCRIPTION", "false").lower() == "true"
        self.max_cost_usd_per_run = float(_env("MAX_COST_USD_PER_RUN", "2.00"))
        self.max_llm_tokens_per_run = int(_env("MAX_LLM_TOKENS_PER_RUN", "500000"))
        self.lookback_days = int(_env("LOOKBACK_DAYS", "3"))
        # .strip() via _env() is the actual fix for the malformed atom:link;
        # .rstrip("/") keeps URL joins from producing a double slash.
        self.pages_base_url = _env("PAGES_BASE_URL").rstrip("/")


def _parse_feed(raw: dict[str, Any]) -> FeedConfig:
    f = raw.get("feed", {})
    return FeedConfig(
        title=f.get("title", "My Podcast Scout"),
        description=f.get("description", ""),
        owner_name=f.get("owner_name", ""),
        owner_email=f.get("owner_email", ""),
        language=f.get("language", "en-us"),
        base_url=f.get("base_url", ""),
    )


def _parse_categories(raw: dict[str, Any]) -> dict[str, CategoryFeedConfig]:
    result: dict[str, CategoryFeedConfig] = {}
    for key, val in raw.get("categories", {}).items():
        result[key] = CategoryFeedConfig(
            slug=val.get("slug", key),
            title=val.get("title", key),
            description=val.get("description", ""),
            max_listen_fully=val.get("max_listen_fully", 3),
            max_read_summary=val.get("max_read_summary", 5),
        )
    return result


def load_preferences(config_dir: Path) -> Preferences:
    path = config_dir / "preferences.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

    caps = raw.get("output_caps", {})
    cls = raw.get("classification", {})
    length = raw.get("length", {})
    persona = raw.get("persona", {})

    return Preferences(
        feed=_parse_feed(raw),
        categories=_parse_categories(raw),
        persona=PersonaConfig(
            role=persona.get("role", "Senior product leader"),
            focus=persona.get("focus", "AI, retail"),
            seniority=persona.get("seniority", "senior"),
            preferred_depth=persona.get("preferred_depth", "strategic"),
        ),
        geography=raw.get("geography", {}),
        length=LengthConfig(
            preferred_min_minutes=length.get("preferred_min_minutes", 20),
            preferred_max_minutes=length.get("preferred_max_minutes", 90),
            hard_max_minutes=length.get("hard_max_minutes", 240),
            max_weekly_listen_hours=length.get("max_weekly_listen_hours", 4.0),
        ),
        classification=ClassificationConfig(
            listen_fully_min_score=cls.get("listen_fully_min_score", 75.0),
            read_summary_min_score=cls.get("read_summary_min_score", 50.0),
            boundary_override_max=cls.get("boundary_override_max", 5.0),
        ),
        output_caps=OutputCapsConfig(
            max_listen_fully=caps.get("max_listen_fully", 3),
            max_read_summary=caps.get("max_read_summary", 5),
            max_outside_feed=caps.get("max_outside_feed", 3),
            max_total_surfaced=caps.get("max_total_surfaced", 10),
        ),
        show_priors=raw.get("show_priors", {}),
        topic_exclusions=raw.get("topic_exclusions", []),
        guest_watchlist=raw.get("guest_watchlist", []),
        competitor_watchlist=raw.get("competitor_watchlist", []),
    )


def load_discovery(config_dir: Path) -> DiscoveryConfig:
    path = config_dir / "discovery_queries.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    disc = raw.get("discovery", {})
    return DiscoveryConfig(
        max_queries=disc.get("max_queries", 15),
        max_raw_candidates=disc.get("max_raw_candidates", 40),
        max_deep_analysis_candidates=disc.get("max_deep_analysis_candidates", 10),
        max_surfaced_outside_episodes=disc.get("max_surfaced_outside_episodes", 3),
        outside_quality_threshold=disc.get("outside_quality_threshold", 60.0),
        static_seeds=raw.get("static_seeds", []),
        entity_seeds=raw.get("entity_seeds", {}),
    )


def load_show_config(config_dir: Path) -> ShowsConfig:
    path = config_dir / "shows.yaml"
    if not path.exists():
        return ShowsConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    return ShowsConfig(shows=[ShowConfig(**show) for show in raw.get("shows", [])])
