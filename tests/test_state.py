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
# playlist_guids() / add_to_playlist() tests
# ---------------------------------------------------------------------------

def test_playlist_guids_empty_initial(tmp_path):
    state = _make_state(tmp_path)
    assert state.playlist_guids() == set()


def test_add_to_playlist_sets_both_flags(tmp_path):
    """add_to_playlist also marks the episode as published."""
    state = _make_state(tmp_path)
    state.add_to_playlist("guid-plist")
    assert "guid-plist" in state.playlist_guids()
    assert "guid-plist" in state.published_guids()


def test_published_without_playlist(tmp_path):
    """add_published alone does NOT add to playlist_guids."""
    state = _make_state(tmp_path)
    state.add_published("guid-email-only")
    assert "guid-email-only" in state.published_guids()
    assert "guid-email-only" not in state.playlist_guids()


def test_playlist_guids_no_duplicates(tmp_path):
    """add_to_playlist is idempotent."""
    state = _make_state(tmp_path)
    state.add_to_playlist("guid-dup")
    state.add_to_playlist("guid-dup")
    assert state.playlist_guids() == {"guid-dup"}


def test_playlist_guids_persists_after_save_load(tmp_path):
    state = _make_state(tmp_path)
    state.add_to_playlist("guid-saved-plist")
    state.save()

    state2 = _make_state(tmp_path)
    assert "guid-saved-plist" in state2.playlist_guids()


# ---------------------------------------------------------------------------
# Regression: playlisted episodes must be excluded from carryover candidates.
# Email-only (published but not playlisted) episodes MUST remain in carryover.
# ---------------------------------------------------------------------------

def test_playlisted_episodes_excluded_from_carryover(tmp_path):
    """_load_carryover_candidates must not return episodes already in the playlist.

    An episode that won a category-feed slot is already in Pocket Casts; it
    must not re-enter the carryover pool and consume a daily cap slot.

    An episode that appeared in email / all.xml only (published but NOT
    playlisted) MUST remain in carryover so it gets another chance to win a
    playlist slot.
    """
    from podcast_scout.cli import _load_carryover_candidates

    state = _make_state(tmp_path)

    playlisted_guid = "plist-guid"        # won a category-feed slot → exclude
    email_only_guid = "email-only-guid"   # published to all.xml only → include
    unqueued_guid = "unqueued-guid"       # scored, never surfaced → include

    for guid, score in [
        (playlisted_guid, 85.0),
        (email_only_guid, 75.0),
        (unqueued_guid, 70.0),
    ]:
        state.mark_processed(EpisodeRecord(
            guid=guid,
            show_title="Test Show",
            episode_title=f"Episode {guid}",
            published=datetime(2026, 7, 28, tzinfo=timezone.utc),
            processed_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
            score=score,
            classification="Listen Fully",
        ))

    # playlisted_guid won a category-feed slot
    state.add_to_playlist(playlisted_guid)
    # email_only_guid was published to all.xml but never won a category slot
    state.add_published(email_only_guid)
    # unqueued_guid was never published at all

    carryover = _load_carryover_candidates(
        state=state,
        category_map={},
        lookback_days=7,
        already_scored_this_run=set(),
    )

    all_guids = {r.episode.guid for eps in carryover.values() for r in eps}

    assert playlisted_guid not in all_guids, (
        "Playlisted episode must be excluded from carryover — it is already in Pocket Casts"
    )
    assert email_only_guid in all_guids, (
        "Email-only episode must remain in carryover so it can still win a playlist slot"
    )
    assert unqueued_guid in all_guids, (
        "Unqueued scored episode must be in carryover so it gets another chance"
    )
