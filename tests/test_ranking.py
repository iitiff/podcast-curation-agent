"""Unit tests for the ranking engine."""
import pytest
from datetime import datetime, timezone

from podcast_scout.normalize import NormalizedEpisode
from podcast_scout.ranking import RubricScore, stage1_metadata_score
from podcast_scout.config import Preferences


def _make_prefs(**kwargs) -> Preferences:
    data = {
        "persona": {"role": "strategy director", "focus": "AI, retail", "seniority": "senior", "preferred_depth": "deep"},
        "interests": {"ai_strategy": 1.0, "ecommerce": 0.8},
        "show_priors": {},
        "guest_watchlist": [],
        "competitor_watchlist": [],
        "topic_exclusions": [],
        "length": {},
        "classification": {},
        "output_caps": {},
        "feed": {},
    }
    data.update(kwargs)
    return Preferences.model_validate(data)


def _make_ep(**kwargs) -> NormalizedEpisode:
    base = {
        "guid": "test-guid-1",
        "source_feed_url": "https://feeds.example.com/show",
        "original_guid": "ep1",
        "show_title": "Test Show",
        "episode_title": "Episode About AI Strategy",
        "description": "Deep dive into AI strategy for retailers",
        "published": datetime(2025, 1, 15, tzinfo=timezone.utc),
        "duration_seconds": 3600,
    }
    base.update(kwargs)
    return NormalizedEpisode(**base)


def test_stage1_relevance_boost():
    ep = _make_ep()
    prefs = _make_prefs()
    result = stage1_metadata_score(ep, prefs)
    assert result.score > 0
    assert result.guid == ep.guid
    assert isinstance(result.should_deep_process, bool)


def test_stage1_topic_exclusion():
    ep = _make_ep(episode_title="All About Crypto NFT Speculation")
    prefs = _make_prefs(topic_exclusions=["crypto", "nft"])
    result = stage1_metadata_score(ep, prefs)
    assert result.score < 10  # penalised heavily


def test_rubric_total_capped_at_100():
    r = RubricScore(
        relevance=30, novelty=15, guest_authority=15,
        actionability=15, evidence=10, strategic_importance=10,
        learning_per_minute=5,
    )
    assert r.total <= 100


def test_rubric_total_not_negative():
    r = RubricScore(
        relevance_penalty=-20, repetition_penalty=-15,
        generic_penalty=-15, confidence_penalty=-15,
    )
    assert r.total >= 0
