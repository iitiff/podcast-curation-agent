"""Unit tests for state manager."""
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from podcast_scout.state import EpisodeRecord, StateManager


def _make_state(tmp_path: Path) -> StateManager:
    return StateManager(tmp_path)


def test_seen_guids_empty_initial(tmp_path):
    state = _make_state(tmp_path)
    assert state.seen_guids() == set()


def test_mark_processed_and_seen(tmp_path):
    state = _make_state(tmp_path)
    rec = EpisodeRecord(
        guid="guid-1",
        show_title="Test Show",
        episode_title="Ep 1",
        published=datetime(2025, 1, 1, tzinfo=timezone.utc),
        processed_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        score=75.0,
        classification="Listen Fully",
    )
    state.mark_processed(rec)
    assert "guid-1" in state.seen_guids()


def test_state_persists_after_save_load(tmp_path):
    state = _make_state(tmp_path)
    rec = EpisodeRecord(
        guid="guid-persist",
        show_title="Show",
        episode_title="Ep",
        published=datetime(2025, 1, 1, tzinfo=timezone.utc),
        processed_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        score=80.0,
        classification="Read Summary Only",
    )
    state.mark_processed(rec)
    state.save()

    state2 = _make_state(tmp_path)
    assert "guid-persist" in state2.seen_guids()


# ---------------------------------------------------------------------------
# published_guids() tests
# ---------------------------------------------------------------------------

def test_published_guids_empty_initial(tmp_path):
    state = _make_state(tmp_path)
    assert state.published_guids() == set()


def test_published_guids_after_add(tmp_path):
    state = _make_state(tmp_path)
    state.add_published("guid-pub-1")
    state.add_published("guid-pub-2")
    assert state.published_guids() == {"guid-pub-1", "guid-pub-2"}


def test_published_guids_no_duplicates(tmp_path):
    """add_published is idempotent; published_guids should deduplicate."""
    state = _make_state(tmp_path)
    state.add_published("guid-dup")
    state.add_published("guid-dup")
    assert state.published_guids() == {"guid-dup"}


def test_published_guids_persists_after_save_load(tmp_path):
    state = _make_state(tmp_path)
    state.add_published("guid-saved")
    state.save()

    state2 = _make_state(tmp_path)
    assert "guid-saved" in state2.published_guids()


# ---------------------------------------------------------------------------
# Regression: published episodes must be excluded from carryover candidates
# ---------------------------------------------------------------------------

def test_published_episodes_excluded_from_carryover(tmp_path):
    """_load_carryover_candidates must not return episodes already published.

    Previously published episodes occupy daily cap slots and prevent new
    episodes from ever reaching the curated playlist.  The RSS feed's own
    prior-item persistence (rss.py _load_prior_items) handles their retention
    independently, so they must not enter the ranking pool a second time.
    """
    from podcast_scout.cli import _load_carryover_candidates

    state = _make_state(tmp_path)

    published_guid = "pub-guid"
    unqueued_guid = "unqueued-guid"

    # Both episodes were scored on a previous run
    for guid, score in [(published_guid, 85.0), (unqueued_guid, 78.0)]:
        state.mark_processed(EpisodeRecord(
            guid=guid,
            show_title="Test Show",
            episode_title=f"Episode {guid}",
            published=datetime(2026, 7, 28, tzinfo=timezone.utc),
            processed_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
            score=score,
            classification="Listen Fully",
        ))

    # Only the first was actually published to the RSS feed
    state.add_published(published_guid)

    carryover = _load_carryover_candidates(
        state=state,
        category_map={},
        lookback_days=7,
        already_scored_this_run=set(),
    )

    all_guids = {r.episode.guid for eps in carryover.values() for r in eps}
    assert published_guid not in all_guids, (
        "Published episode must be excluded from carryover to avoid consuming daily cap slots"
    )
    assert unqueued_guid in all_guids, (
        "Unqueued scored episode must be in carryover so it gets another chance"
    )
