"""Configuration loading and validation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeedConfig(BaseModel):
    title: str = "My Podcast Scout"
    description: str = "A ranked weekly listening queue curated by AI."
    owner_name: str = "Podcast Scout User"
    owner_email: str = ""
    language: str = "en-us"
    base_url: str = ""


class PersonaConfig(BaseModel):
    role: str = "Product leader"
    focus: str = "AI, technology, and product strategy"
    seniority: str = "senior"
    preferred_depth: str = "strategic"


class LengthConfig(BaseModel):
    preferred_min_minutes: int = 20
    preferred_max_minutes: int = 90
    hard_max_minutes: int = 240
    max_weekly_listen_hours: float = 4.0


class ClassificationConfig(BaseModel):
    listen_fully_min_score: int = 75
    read_summary_min_score: int = 50
    boundary_override_max: int = 5


class OutputCapsConfig(BaseModel):
    max_listen_fully: int = 3
    max_read_summary: int = 5
    max_outside_feed: int = 3
    max_total_surfaced: int = 10


class Preferences(BaseModel):
    feed: FeedConfig = Field(default_factory=FeedConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    interests: dict[str, float] = Field(default_factory=dict)
    geography: dict[str, float] = Field(default_factory=dict)
    length: LengthConfig = Field(default_factory=LengthConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    output_caps: OutputCapsConfig = Field(default_factory=OutputCapsConfig)
    show_priors: dict[str, float] = Field(default_factory=dict)
    topic_exclusions: list[str] = Field(default_factory=list)
    guest_watchlist: list[str] = Field(default_factory=list)
    competitor_watchlist: list[str] = Field(default_factory=list)


class ShowOverride(BaseModel):
    match: str
    display_name: str | None = None
    canonical_feed_url: str | None = None
    priority: float | None = None
    enabled: bool = True
    language: str | None = None
    transcript_source: str | None = None  # auto | p20 | publisher | youtube | none
    max_episodes_per_run: int = 3
    notes: str | None = None


class ShowsConfig(BaseModel):
    shows: list[ShowOverride] = Field(default_factory=list)


class DiscoveryLimits(BaseModel):
    max_queries: int = 15
    max_raw_candidates: int = 40
    max_deep_analysis_candidates: int = 10
    max_surfaced_outside_episodes: int = 3
    outside_quality_threshold: int = 60


class DiscoveryConfig(BaseModel):
    discovery: DiscoveryLimits = Field(default_factory=DiscoveryLimits)
    static_seeds: list[dict[str, Any]] = Field(default_factory=list)
    entity_seeds: dict[str, list[str]] = Field(default_factory=dict)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_stage1_model: str = "gpt-4o-mini"
    openai_stage2_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"

    podcast_index_key: str = ""
    podcast_index_secret: str = ""

    web_search_api_key: str = ""
    web_search_provider: str = "brave"  # brave | serper | none

    enable_audio_transcription: bool = False

    max_cost_usd_per_run: float = 2.00
    max_llm_tokens_per_run: int = 500_000

    pages_base_url: str = ""

    # Pipeline behaviour
    lookback_days: int = 8
    config_dir: Path = Path("config")
    data_dir: Path = Path("data")
    public_dir: Path = Path("public")
    templates_dir: Path = Path("templates")

    @model_validator(mode="after")
    def _apply_pages_base_url(self) -> "Settings":
        if not self.pages_base_url and os.getenv("GITHUB_REPOSITORY"):
            repo = os.getenv("GITHUB_REPOSITORY", "")
            owner = repo.split("/")[0] if "/" in repo else ""
            name = repo.split("/")[1] if "/" in repo else repo
            self.pages_base_url = f"https://{owner}.github.io/{name}"
        return self


def load_preferences(config_dir: Path) -> Preferences:
    path = config_dir / "preferences.yaml"
    if not path.exists():
        return Preferences()
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return Preferences.model_validate(data)


def load_shows(config_dir: Path) -> ShowsConfig:
    path = config_dir / "shows.yaml"
    if not path.exists():
        return ShowsConfig()
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return ShowsConfig.model_validate(data)


def load_discovery(config_dir: Path) -> DiscoveryConfig:
    path = config_dir / "discovery_queries.yaml"
    if not path.exists():
        return DiscoveryConfig()
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return DiscoveryConfig.model_validate(data)
