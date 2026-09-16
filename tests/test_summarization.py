"""Stage 2 slot allocation.

Stage 2 is the scarce resource — the free Gemini tier is capped per day, not
per token — so which candidates get an LLM call decides what the feed can
contain. These cover the per-lane reservation that keeps a scarce lane from
being outbid by an abundant one.
"""
from datetime import UTC, datetime

from podcast_scout.config import CategoryFeedConfig, Preferences
from podcast_scout.normalize import NormalizedEpisode
from podcast_scout.ranking import Stage1Result
from podcast_scout.summarization import _allocate_deep_slots


def _ep(guid: str) -> NormalizedEpisode:
    return NormalizedEpisode(
        guid=guid,
        source_feed_url="https://feeds.example.com/show",
        original_guid=guid,
        show_title="Test Show",
        episode_title=f"Episode {guid}",
        description="",
        published=datetime(2025, 1, 15, tzinfo=UTC),
        duration_seconds=3600,
    )


def _prefs(min_deep_slots: int = 2) -> Preferences:
    return Preferences(categories={
        "scarce": CategoryFeedConfig(
            slug="scarce", title="Scarce", min_deep_slots=min_deep_slots,
        ),
        "abundant": CategoryFeedConfig(slug="abundant", title="Abundant"),
    })


def _scenario(scarce: int, abundant: int) -> tuple[list[NormalizedEpisode], dict[str, Stage1Result]]:
    """Abundant items score higher, so without a reservation they take everything."""
    eps, s1 = [], {}
    for i in range(abundant):
        guid = f"abundant-{i}"
        eps.append(_ep(guid))
        s1[guid] = Stage1Result(
            guid=guid, score=90 - i, reason="", should_deep_process=True,
            predicted_category="abundant",
        )
    for i in range(scarce):
        guid = f"scarce-{i}"
        eps.append(_ep(guid))
        s1[guid] = Stage1Result(
            guid=guid, score=50 - i, reason="", should_deep_process=True,
            predicted_category="scarce",
        )
    return eps, s1


def test_scarce_lane_is_outbid_without_a_reservation():
    """The behaviour the reservation exists to change."""
    eps, s1 = _scenario(scarce=3, abundant=10)
    chosen = _allocate_deep_slots(eps, 5, s1, _prefs(min_deep_slots=0))
    assert all(ep.guid.startswith("abundant") for ep in chosen)


def test_reservation_guarantees_the_scarce_lane_its_slots():
    eps, s1 = _scenario(scarce=3, abundant=10)
    chosen = _allocate_deep_slots(eps, 5, s1, _prefs(min_deep_slots=2))
    guids = {ep.guid for ep in chosen}
    assert len(chosen) == 5
    assert {"scarce-0", "scarce-1"} <= guids, "the lane's two best must be reserved"
    assert "scarce-2" not in guids, "a reservation is a floor, not a quota"


def test_unused_reservation_costs_nothing():
    """A lane with nothing to offer must not shrink the run."""
    eps, s1 = _scenario(scarce=0, abundant=10)
    chosen = _allocate_deep_slots(eps, 5, s1, _prefs(min_deep_slots=2))
    assert len(chosen) == 5
    assert all(ep.guid.startswith("abundant") for ep in chosen)


def test_partial_reservation_takes_only_what_exists():
    eps, s1 = _scenario(scarce=1, abundant=10)
    chosen = _allocate_deep_slots(eps, 5, s1, _prefs(min_deep_slots=2))
    assert len(chosen) == 5
    assert "scarce-0" in {ep.guid for ep in chosen}


def test_reservation_never_exceeds_the_limit():
    eps, s1 = _scenario(scarce=9, abundant=1)
    chosen = _allocate_deep_slots(eps, 3, s1, _prefs(min_deep_slots=5))
    assert len(chosen) == 3


def test_no_candidates_or_no_limit_yields_nothing():
    eps, s1 = _scenario(scarce=2, abundant=2)
    assert _allocate_deep_slots(eps, 0, s1, _prefs()) == []
    assert _allocate_deep_slots([], 5, {}, _prefs()) == []
