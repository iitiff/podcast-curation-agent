"""Unit tests for the ranking engine."""
import json
from datetime import UTC, datetime

import pytest

from podcast_scout.config import PersonaConfig, Preferences
from podcast_scout.normalize import NormalizedEpisode
from podcast_scout.ranking import (
    RankedEpisode,
    RubricScore,
    _build_item_block,
    build_daily_queue,
    stage1_metadata_score,
    stage2_batch_rank,
)


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


# -- budget separation: reading must not draw on the listening budget --------

def _queue_item(
    classification: str,
    score: float,
    source_type: str = "podcast",
    duration: int = 3600,
    **kwargs,
) -> RankedEpisode:
    from podcast_scout.normalize import Enclosure

    enclosure = (
        Enclosure(url="https://cdn.example.com/a.mp3", mime_type="audio/mpeg", length=1)
        if source_type == "podcast"
        else None
    )
    ep = _make_ep(
        guid=f"guid-{score}-{source_type}-{kwargs.pop('n', 0)}",
        source_type=source_type,
        duration_seconds=duration,
        enclosure=enclosure,
        **kwargs,
    )
    return RankedEpisode(
        episode=ep, score=score, rubric=RubricScore(), classification=classification
    )


def test_articles_never_take_a_listening_slot():
    """A paper scored "Listen Fully" must not consume max_listen_fully.

    rss.py drops enclosure-less items from the feed, so before the split each
    one silently shrank the published queue.
    """
    items = [
        _queue_item("Listen Fully", 95.0, source_type="research-paper", duration=0, n=1),
        _queue_item("Listen Fully", 94.0, source_type="trade-press", duration=0, n=2),
        _queue_item("Listen Fully", 80.0, n=3),
        _queue_item("Listen Fully", 78.0, n=4),
    ]
    rss, email_only, reading = build_daily_queue(items, max_listen_fully=2)

    assert [r.score for r in rss] == [80.0, 78.0]
    assert all(r.episode.is_playable for r in rss)
    assert len(reading) == 2
    assert not email_only


def test_articles_do_not_crowd_out_podcast_summaries():
    """max_reading is a separate pot from max_email_only."""
    items = [
        _queue_item("Listen Fully", 90.0 - i, source_type="research-paper", duration=0, n=i)
        for i in range(12)
    ] + [_queue_item("Read Summary Only", 60.0, n=99)]

    _, email_only, reading = build_daily_queue(
        items, max_reading=3, max_email_only=5
    )

    assert len(reading) == 3
    # The podcast summary survives even though 12 articles outscored it.
    assert [r.episode.guid for r in email_only] == ["guid-60.0-podcast-99"]


def test_listen_minutes_budget_ignores_articles():
    """Zero-duration articles must not be what exhausts the minute budget."""
    items = [
        _queue_item("Listen Fully", 99.0, source_type="trade-press", duration=0, n=1),
        _queue_item("Listen Fully", 70.0, duration=3600, n=2),
    ]
    rss, _, reading = build_daily_queue(items, max_minutes=60.0, max_listen_fully=5)

    assert len(rss) == 1 and rss[0].episode.source_type == "podcast"
    assert len(reading) == 1


# -- the rubric must not judge written sources on podcast-shaped fields ------

def _written(source_type="research-paper", **kwargs):
    return _make_ep(
        guid=f"g-{source_type}",
        show_title=kwargs.pop("show", "arXiv"),
        episode_title="Goal arbitration in agent stacks",
        description=kwargs.pop(
            "description", "We formalise arbitration between competing objectives."
        ),
        duration_seconds=0,
        source_type=source_type,
        **kwargs,
    )


def _no_transcript():
    from podcast_scout.summarization import TranscriptResult

    return TranscriptResult(text="", confidence="low", source="none")


def test_written_sources_declare_class_not_guests_and_duration():
    """"GUESTS: unknown / DURATION: 0 min" on a paper invites a penalty for
    something that was never missing."""
    block = _build_item_block(0, _written(credibility="high"), _no_transcript())

    assert "SOURCE CLASS: research-paper (credibility: high)" in block
    assert "GUESTS:" not in block
    assert "DURATION:" not in block


def test_podcasts_keep_guests_and_duration():
    block = _build_item_block(0, _make_ep(guests=["Ben"]), _no_transcript())

    assert "GUESTS: Ben" in block
    assert "DURATION: 60 min" in block
    assert "SOURCE CLASS:" not in block


def test_known_bias_reaches_the_prompt_for_vendor_material():
    block = _build_item_block(
        0,
        _written(source_type="vendor", credibility="low",
                 bias_notes="Vendor marketing; discount performance claims."),
        _no_transcript(),
    )

    assert "credibility: low" in block
    assert "KNOWN BIAS: Vendor marketing" in block


def test_long_source_text_is_capped():
    """An earnings exhibit carries thousands of words. The description path was
    previously uncapped, which was only safe while every item was a podcast."""
    huge = _written(description="x" * 50_000)
    block = _build_item_block(0, huge, _no_transcript())

    assert len(block) < 4_000


class _CapturingLLM:
    """Records the prompt it was given and returns a minimal valid response."""

    def __init__(self, count: int) -> None:
        self.prompts: list[str] = []
        self.count = count

    async def complete(self, messages, max_tokens=4096):
        from podcast_scout.providers.base import LLMResponse

        self.prompts.append("\n".join(m.content for m in messages))
        body = json.dumps([
            {
                "rubric": {"relevance": 20},
                "classification": "Read Summary Only",
                "classification_reason": "stub",
                "summary": "stub",
                "key_ideas": [],
                "implications": "",
                "who_should_listen": "",
                "summary_captures_value": "partial",
                "listen_nuance": "",
            }
            for _ in range(self.count)
        ])
        return LLMResponse(content=body, input_tokens=1, output_tokens=1)


async def _prompt_for(episodes):
    llm = _CapturingLLM(len(episodes))
    await stage2_batch_rank(
        [(ep, _no_transcript()) for ep in episodes], _make_prefs(), llm
    )
    return llm.prompts[0]


@pytest.mark.asyncio
async def test_media_guidance_is_added_when_the_batch_is_not_all_podcasts():
    prompt = await _prompt_for([_make_ep(), _written(source_type="vendor")])

    assert "THIS BATCH MIXES MEDIA" in prompt
    # The source-class weighting is the point: vendor claims must be discounted.
    assert "vendor (low)" in prompt
    assert "guest_authority -> AUTHOR authority" in prompt


@pytest.mark.asyncio
async def test_media_guidance_is_omitted_for_an_all_podcast_batch():
    """The common case should not pay tokens for guidance it cannot use."""
    prompt = await _prompt_for([_make_ep(), _make_ep(guid="g2")])

    assert "THIS BATCH MIXES MEDIA" not in prompt


@pytest.mark.asyncio
async def test_the_prompt_no_longer_claims_every_item_is_a_podcast():
    prompt = await _prompt_for([_written()])

    assert "podcast episode(s)" not in prompt
    # The confidence penalty must not fire on a written source for lacking a
    # transcript -- the text is the source.
    assert "Does NOT apply to a written source" in prompt


# ---------------------------------------------------------------------------
# Topic-based category routing
# ---------------------------------------------------------------------------

def _routed(guid: str, category: str, assigned: str = "") -> RankedEpisode:
    return RankedEpisode(
        episode=_make_ep(guid=guid, category=category),
        score=80.0,
        rubric=RubricScore(),
        classification="Listen Fully",
        assigned_category=assigned,
    )


def test_routing_moves_episode_into_the_lane_stage2_chose():
    from podcast_scout.cli import _apply_assigned_categories

    buckets = {"ai_retail": [_routed("a", "ai_retail", "personalization")], "personalization": []}
    moved = _apply_assigned_categories(buckets, {"ai_retail", "personalization"})

    assert moved == 1
    assert [r.episode.guid for r in buckets["personalization"]] == ["a"]
    assert buckets["ai_retail"] == []
    assert buckets["personalization"][0].episode.category == "personalization"


def test_routing_ignores_a_category_that_was_never_offered():
    """A hallucinated lane must not silently create a feed nobody configured."""
    from podcast_scout.cli import _apply_assigned_categories

    buckets = {"ai_retail": [_routed("a", "ai_retail", "quantum_basketball")]}
    moved = _apply_assigned_categories(buckets, {"ai_retail", "personalization"})

    assert moved == 0
    assert [r.episode.guid for r in buckets["ai_retail"]] == ["a"]
    assert buckets["ai_retail"][0].episode.category == "ai_retail"


def test_routing_leaves_the_show_lane_alone_when_stage2_abstains():
    """Empty assignment is the metadata-floor path: no LLM, no opinion."""
    from podcast_scout.cli import _apply_assigned_categories

    buckets = {"startup": [_routed("a", "startup", "")]}
    moved = _apply_assigned_categories(buckets, {"startup", "personalization"})

    assert moved == 0
    assert buckets["startup"][0].episode.category == "startup"


def test_carryover_keeps_a_routed_lane_instead_of_re_deriving_it():
    """Without this, a routed episode snaps back to its show's lane tomorrow."""
    from podcast_scout.cli import _category_for_record
    from podcast_scout.state import EpisodeRecord

    category_map = {"lenny's podcast": "ai_retail"}
    routed = EpisodeRecord(guid="a", show_title="Lenny's Podcast", category="personalization")
    assert _category_for_record(routed, category_map) == "personalization"


def test_carryover_falls_back_for_records_written_before_the_field_existed():
    from podcast_scout.cli import _category_for_record
    from podcast_scout.state import EpisodeRecord

    category_map = {"lenny's podcast": "ai_retail"}
    legacy = EpisodeRecord(guid="a", show_title="Lenny's Podcast")
    assert _category_for_record(legacy, category_map) == "ai_retail"


class _RoutingLLM:
    """Returns a fixed category for every item, so the parse path is exercised."""

    def __init__(self, count: int, category: str) -> None:
        self.count = count
        self.category = category

    async def complete(self, messages, max_tokens=4096):
        from podcast_scout.providers.base import LLMResponse

        body = json.dumps([
            {
                "rubric": {"relevance": 25},
                "classification": "Listen Fully",
                "classification_reason": "stub",
                "summary": "stub",
                "key_ideas": [],
                "implications": "",
                "who_should_listen": "",
                "summary_captures_value": "partial",
                "listen_nuance": "",
                "category": self.category,
            }
        ] * self.count)
        return LLMResponse(content=body, input_tokens=1, output_tokens=1)


@pytest.mark.asyncio
async def test_stage2_carries_the_category_back_off_the_wire():
    ranked = await stage2_batch_rank(
        [(_make_ep(), _no_transcript())], _make_prefs(), _RoutingLLM(1, "personalization")
    )

    assert ranked[0].assigned_category == "personalization"


@pytest.mark.asyncio
async def test_the_prompt_offers_only_configured_categories():
    from podcast_scout.config import CategoryFeedConfig

    prefs = _make_prefs(categories={
        "personalization": CategoryFeedConfig(
            slug="personalization", title="P", routing_hint="how a decision gets made"
        ),
        "startup": CategoryFeedConfig(slug="startup", title="S", description="founders"),
    })
    llm = _CapturingLLM(1)
    await stage2_batch_rank([(_make_ep(), _no_transcript())], prefs, llm)
    prompt = llm.prompts[0]

    assert '"personalization": how a decision gets made' in prompt
    # description is the documented fallback when no routing_hint is set
    assert '"startup": founders' in prompt


@pytest.mark.asyncio
async def test_no_category_is_asked_for_when_only_one_lane_exists():
    """A single-lane setup should not pay tokens for a choice it cannot make."""
    from podcast_scout.config import CategoryFeedConfig

    prefs = _make_prefs(categories={
        "ai_retail": CategoryFeedConfig(slug="ai-retail", title="A", description="everything"),
    })
    llm = _CapturingLLM(1)
    await stage2_batch_rank([(_make_ep(), _no_transcript())], prefs, llm)

    assert "CATEGORY — which lane" not in llm.prompts[0]
