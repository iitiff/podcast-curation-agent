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
