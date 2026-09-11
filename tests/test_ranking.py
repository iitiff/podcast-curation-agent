"""Unit tests for the ranking engine."""
from datetime import UTC, datetime

from podcast_scout.config import PersonaConfig, Preferences
from podcast_scout.normalize import NormalizedEpisode
from podcast_scout.ranking import RubricScore, stage1_metadata_score


def _make_prefs(**kwargs) -> Preferences:
    """Build a Preferences for tests.

    Preferences is a dataclass (not a pydantic model), so it is constructed
    directly rather than via model_validate; every field already carries a
    default_factory, so only the overrides under test need to be supplied.
    """
    persona = kwargs.pop(
        "persona",
        {
            "role": "strategy director",
            "focus": "AI, retail",
            "seniority": "senior",
            "preferred_depth": "deep",
        },
    )
    return Preferences(persona=PersonaConfig(**persona), **kwargs)


def _make_ep(**kwargs) -> NormalizedEpisode:
    base = {
        "guid": "test-guid-1",
        "source_feed_url": "https://feeds.example.com/show",
        "original_guid": "ep1",
        "show_title": "Test Show",
        "episode_title": "Episode About AI Strategy",
        "description": "Deep dive into AI strategy for retailers",
        "published": datetime(2025, 1, 15, tzinfo=UTC),
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


# ---------------------------------------------------------------------------
# Total LLM degradation must fail the run BEFORE state is persisted.
#
# Metadata-floor scores cannot reach the "Listen Fully" threshold, so a fully
# degraded run surfaces nothing. Persisting it would mark every episode as
# seen, deduping them out of tomorrow's run and freezing the feed until
# someone runs `rescore`.
# ---------------------------------------------------------------------------

_FALLBACK_REASON = "LLM batch entry missing; metadata fallback"


def _ranked(reason: str, score: float = 50.0):
    from podcast_scout.ranking import RankedEpisode

    return RankedEpisode(
        episode=_make_ep(), score=score, rubric=RubricScore(),
        classification="Skip", classification_reason=reason,
    )


def _all_degraded(newly_ranked: dict, llm_configured: bool = True) -> bool:
    """Mirror of the guard in cli._run_pipeline."""
    if not llm_configured:
        return False
    fresh = [r for cat in newly_ranked.values() for r in cat]
    degraded = [r for r in fresh if "metadata fallback" in r.classification_reason]
    return bool(fresh) and len(degraded) == len(fresh)


def test_total_degradation_is_detected():
    assert _all_degraded({"ai_retail": [_ranked(_FALLBACK_REASON)] * 3}) is True


def test_partial_degradation_is_not_fatal():
    """A few truncated entries are normal; surviving scores are still useful."""
    batch = {"ai_retail": [_ranked(_FALLBACK_REASON), _ranked("Strong relevance", 82.0)]}
    assert _all_degraded(batch) is False


def test_no_llm_configured_does_not_trip_the_guard():
    """Metadata-only is the expected mode with no LLM, not a failure."""
    batch = {"ai_retail": [_ranked(_FALLBACK_REASON)] * 2}
    assert _all_degraded(batch, llm_configured=False) is False


def test_empty_run_does_not_trip_the_guard():
    """Nothing discovered is not the same as everything failing."""
    assert _all_degraded({"ai_retail": []}) is False


def _attempted_all_degraded(newly_ranked: dict, llm_configured: bool = True) -> bool:
    """Mirror of the corrected guard: only Stage-2 attempts count."""
    if not llm_configured:
        return False
    fresh = [r for cat in newly_ranked.values() for r in cat]
    attempted = [
        r for r in fresh
        if r.classification_reason not in {"stage1 only", "token budget exhausted"}
    ]
    degraded = [r for r in attempted if "metadata fallback" in r.classification_reason]
    return bool(attempted) and len(degraded) == len(attempted)


def test_stage1_filtered_episodes_excluded_from_denominator():
    """The bug that kept this guard silent through two failed live runs.

    Stage-1 filtering legitimately stops episodes before any LLM call, so
    counting them made 100% degradation unreachable.
    """
    batch = {"ai_retail": [
        _ranked(_FALLBACK_REASON),
        _ranked(_FALLBACK_REASON),
        _ranked("stage1 only", 44.0),
        _ranked("token budget exhausted", 41.0),
    ]}
    assert _attempted_all_degraded(batch) is True


def test_guard_still_silent_when_one_stage2_call_succeeded():
    batch = {"ai_retail": [
        _ranked(_FALLBACK_REASON),
        _ranked("Strong relevance to AI retail", 81.0),
        _ranked("stage1 only", 44.0),
    ]}
    assert _attempted_all_degraded(batch) is False


def test_guard_silent_when_nothing_reached_stage2():
    batch = {"ai_retail": [_ranked("stage1 only", 44.0)]}
    assert _attempted_all_degraded(batch) is False
