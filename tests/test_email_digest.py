"""The email digest body.

The reading track carries no enclosure, so no playable feed can ever hold it.
The email is its only delivery path, which is why these cover it specifically.
"""
from datetime import UTC, datetime

from podcast_scout.email_digest import build_email_html
from podcast_scout.normalize import NormalizedEpisode
from podcast_scout.ranking import RankedEpisode, RubricScore
from podcast_scout.state import EpisodeRecord


def _ranked(
    guid: str = "g1",
    *,
    source_type: str = "podcast",
    duration_seconds: int = 3600,
    score: float = 80.0,
    title: str = "An Episode",
) -> RankedEpisode:
    ep = NormalizedEpisode(
        guid=guid,
        source_feed_url="https://feeds.example.com/show",
        original_guid=guid,
        show_title="Test Source",
        episode_title=title,
        description="Body text.",
        published=datetime(2026, 9, 16, tzinfo=UTC),
        duration_seconds=duration_seconds,
        source_type=source_type,
        episode_url="https://example.com/item",
    )
    return RankedEpisode(
        episode=ep,
        score=score,
        rubric=RubricScore(),
        classification="Read Summary Only",
        classification_reason="",
        evidence_confidence="medium",
        summary="What it covers.",
    )


def test_reading_track_is_rendered():
    paper = _ranked("p1", source_type="research-paper", duration_seconds=0, title="A Paper")
    html = build_email_html([], [], "2026-09-16", reading=[paper])
    assert "Worth Reading" in html
    assert "A Paper" in html


def test_reading_track_is_omitted_when_empty():
    html = build_email_html([_ranked()], [], "2026-09-16", reading=[])
    assert "Worth Reading" not in html


def test_written_source_names_its_kind_instead_of_a_runtime():
    """"unknown length" on a paper reads as missing data, not as "not audio"."""
    paper = _ranked("p1", source_type="research-paper", duration_seconds=0)
    html = build_email_html([], [], "2026-09-16", reading=[paper])
    assert "research paper" in html
    assert "unknown length" not in html


def test_podcast_still_shows_its_runtime():
    html = build_email_html([_ranked(duration_seconds=1800)], [], "2026-09-16")
    assert "30m" in html


def test_source_type_survives_a_state_round_trip():
    """Carryover rebuilds from this record; a default of "podcast" would relabel papers."""
    rec = EpisodeRecord(guid="p1", source_type="research-paper")
    assert EpisodeRecord(**rec.model_dump(mode="json")).source_type == "research-paper"


def test_source_type_defaults_for_state_written_before_the_field_existed():
    assert EpisodeRecord(guid="p1").source_type == "podcast"
