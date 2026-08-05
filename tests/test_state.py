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


# ---------------------------------------------------------------------------
# forget_processed() / all_records() — support for the `rescore` command
# ---------------------------------------------------------------------------

def _rec(guid, score, classification="Read Summary Only", processed_day=4):
    return EpisodeRecord(
        guid=guid,
        show_title="Show",
        episode_title=f"Ep {guid}",
        published=datetime(2026, 8, processed_day, tzinfo=timezone.utc),
        processed_at=datetime(2026, 8, processed_day, tzinfo=timezone.utc),
        score=score,
        classification=classification,
    )


def test_all_records_returns_parsed_records(tmp_path):
    state = _make_state(tmp_path)
    state.mark_processed(_rec("a", 50.0))
    state.mark_processed(_rec("b", 77.0, "Listen Fully"))

    records = state.all_records()
    assert {r.guid for r in records} == {"a", "b"}
    assert {r.score for r in records} == {50.0, 77.0}


def test_all_records_skips_malformed_rows(tmp_path):
    """A single corrupt entry must not break state-wide operations."""
    state = _make_state(tmp_path)
    state.mark_processed(_rec("good", 50.0))
    # Inject a row that cannot parse into EpisodeRecord
    state._state["processed"]["bad"] = {"not_a_valid": "record", "score": "abc"}

    records = state.all_records()
    assert [r.guid for r in records] == ["good"]


def test_forget_processed_removes_and_counts(tmp_path):
    state = _make_state(tmp_path)
    state.mark_processed(_rec("a", 50.0))
    state.mark_processed(_rec("b", 50.0))
    state.mark_processed(_rec("c", 77.0, "Listen Fully"))

    removed = state.forget_processed(["a", "b"])
    assert removed == 2
    assert state.seen_guids() == {"c"}


def test_forget_processed_ignores_unknown_guids(tmp_path):
    state = _make_state(tmp_path)
    state.mark_processed(_rec("a", 50.0))

    removed = state.forget_processed(["a", "does-not-exist"])
    assert removed == 1
    assert state.seen_guids() == set()


def test_forget_processed_preserves_published_and_playlist(tmp_path):
    """Forgetting must not drop feed history — otherwise a re-scored episode
    could be re-added to the curated feed as a duplicate."""
    state = _make_state(tmp_path)
    state.mark_processed(_rec("a", 50.0, "Listen Fully"))
    state.add_to_playlist("a")

    state.forget_processed(["a"])

    assert "a" not in state.seen_guids()
    assert "a" in state.published_guids()
    assert "a" in state.playlist_guids()


def test_forget_processed_survives_save_load(tmp_path):
    state = _make_state(tmp_path)
    state.mark_processed(_rec("a", 50.0))
    state.mark_processed(_rec("b", 50.0))
    state.forget_processed(["a"])
    state.save()

    state2 = _make_state(tmp_path)
    assert state2.seen_guids() == {"b"}


# ---------------------------------------------------------------------------
# LLM insight persistence — regression guard
#
# Stage 2 runs ONCE per episode. If summary/key_ideas are not persisted, every
# carried-over episode is rebuilt as an insight-free stub and the email digest
# shows "No AI analysis available" for everything except that day's new items.
# ---------------------------------------------------------------------------

def test_episode_record_persists_llm_insights(tmp_path):
    state = _make_state(tmp_path)
    state.mark_processed(EpisodeRecord(
        guid="rich",
        show_title="The a16z Show",
        episode_title="OpenAI's Joshua Achiam",
        published=datetime(2026, 8, 5, tzinfo=timezone.utc),
        processed_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
        score=77.0,
        classification="Listen Fully",
        classification_reason="High relevance to AI/emerging tech.",
        summary="Joshua Achiam explores whether society already entered an AGI era.",
        key_ideas=["Idea one.", "Idea two.", "Idea three."],
        episode_url="https://a16z.simplecast.com/episodes/xyz",
        duration_seconds=1860,
    ))
    state.save()

    # Reload from disk — this is the path a later run takes.
    reloaded = _make_state(tmp_path).get_record("rich")
    assert reloaded is not None
    assert reloaded.summary.startswith("Joshua Achiam")
    assert reloaded.key_ideas == ["Idea one.", "Idea two.", "Idea three."]
    assert reloaded.episode_url == "https://a16z.simplecast.com/episodes/xyz"
    assert reloaded.duration_seconds == 1860
    assert reloaded.classification_reason == "High relevance to AI/emerging tech."


def test_episode_record_defaults_keep_old_state_loadable(tmp_path):
    """A state.json written before these fields existed must still load."""
    state = _make_state(tmp_path)
    # Simulate a legacy row: no summary/key_ideas/episode_url/duration_seconds.
    state._state["processed"]["legacy"] = {
        "guid": "legacy",
        "show_title": "Old Show",
        "episode_title": "Old Ep",
        "score": 50.0,
        "classification": "Read Summary Only",
    }
    state.save()

    reloaded = _make_state(tmp_path).get_record("legacy")
    assert reloaded is not None
    assert reloaded.summary == ""
    assert reloaded.key_ideas == []
    assert reloaded.episode_url == ""
    assert reloaded.duration_seconds == 0


def test_carryover_restores_insights(tmp_path):
    """_load_carryover_candidates must rebuild stubs WITH their insights."""
    from podcast_scout.cli import _load_carryover_candidates

    state = _make_state(tmp_path)
    state.mark_processed(EpisodeRecord(
        guid="carried",
        show_title="Some Show",
        episode_title="Some Episode",
        published=datetime(2026, 8, 4, tzinfo=timezone.utc),
        processed_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
        score=68.0,
        classification="Read Summary Only",
        classification_reason="Solid but not top tier.",
        summary="A useful discussion about product strategy.",
        key_ideas=["Takeaway A.", "Takeaway B."],
        episode_url="https://example.com/ep",
        duration_seconds=2400,
    ))

    carryover = _load_carryover_candidates(
        state=state,
        category_map={},
        lookback_days=30,
        already_scored_this_run=set(),
    )
    items = [r for eps in carryover.values() for r in eps]
    assert len(items) == 1
    r = items[0]
    assert r.key_ideas == ["Takeaway A.", "Takeaway B."], "insights must survive carryover"
    assert r.summary == "A useful discussion about product strategy."
    assert r.episode.episode_url == "https://example.com/ep"
    assert r.episode.duration_seconds == 2400
    assert "carried over" in r.classification_reason


# ---------------------------------------------------------------------------
# emailed_guids() — one email per episode
#
# Carryover keys off `playlist` so an episode that missed a feed slot keeps
# competing on later days. Without a separate `emailed` set that also meant the
# same episode reappeared in every daily digest until it won a slot or aged out.
# ---------------------------------------------------------------------------

def test_emailed_guids_empty_initial(tmp_path):
    assert _make_state(tmp_path).emailed_guids() == set()


def test_add_emailed_is_idempotent(tmp_path):
    state = _make_state(tmp_path)
    state.add_emailed("a")
    state.add_emailed("a")
    assert state.emailed_guids() == {"a"}


def test_emailed_guids_persists(tmp_path):
    state = _make_state(tmp_path)
    state.add_emailed("a")
    state.save()
    assert "a" in _make_state(tmp_path).emailed_guids()


def test_emailed_is_independent_of_playlist_and_published(tmp_path):
    """The three sets must not leak into each other.

    An episode can be emailed without being playlisted (the common case: it
    surfaced in the digest but lost the feed-slot contest), and vice versa.
    """
    state = _make_state(tmp_path)
    state.add_emailed("emailed-only")
    state.add_to_playlist("playlisted-only")

    assert state.emailed_guids() == {"emailed-only"}
    assert state.playlist_guids() == {"playlisted-only"}
    assert "emailed-only" not in state.playlist_guids()
    assert "playlisted-only" not in state.emailed_guids()
    # add_to_playlist implies published; add_emailed does not.
    assert "playlisted-only" in state.published_guids()
    assert "emailed-only" not in state.published_guids()


def test_forget_processed_preserves_emailed(tmp_path):
    """Re-scoring must not cause an episode to be emailed twice."""
    state = _make_state(tmp_path)
    state.mark_processed(_rec("a", 50.0))
    state.add_emailed("a")

    state.forget_processed(["a"])

    assert "a" not in state.seen_guids()
    assert "a" in state.emailed_guids(), "re-score must not re-open the email gate"


def test_emailed_survives_legacy_state_without_the_key(tmp_path):
    """A state.json written before `emailed` existed must still load."""
    state = _make_state(tmp_path)
    state._state.pop("emailed", None)
    state.save()

    reloaded = _make_state(tmp_path)
    assert reloaded.emailed_guids() == set()
    reloaded.add_emailed("x")
    assert reloaded.emailed_guids() == {"x"}
