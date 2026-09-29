"""The transcript archive that downstream readers consume."""
import json
from datetime import UTC, datetime

from podcast_scout.normalize import NormalizedEpisode
from podcast_scout.providers.base import TranscriptResult
from podcast_scout.ranking import RankedEpisode, RubricScore
from podcast_scout.transcripts import TranscriptArchive


def _ep(guid: str = "g1") -> NormalizedEpisode:
    return NormalizedEpisode(
        guid=guid,
        source_feed_url="https://feeds.example.com/show",
        show_title="Show",
        episode_title=f"Episode {guid}",
        published=datetime(2025, 1, 13, tzinfo=UTC),
        duration_seconds=3600,
        episode_url=f"https://example.com/{guid}",
        category="startup",
    )


FULL = TranscriptResult(text="Every word that was said. " * 50, source="publisher", confidence="high")


def test_only_full_transcripts_are_archived(tmp_path):
    archive = TranscriptArchive(tmp_path)
    assert archive.save(_ep("a"), FULL)
    assert not archive.save(_ep("b"), TranscriptResult(text="notes", source="description"))
    archive.flush()
    index = json.loads((tmp_path / "index.json").read_text())
    assert [e["guid"] for e in index["episodes"]] == ["a"]
    entry = index["episodes"][0]
    assert (tmp_path / entry["path"]).read_text() == FULL.text
    assert entry["transcript_source"] == "publisher"
    assert entry["chars"] == len(FULL.text)


def test_annotate_adds_the_curators_verdict(tmp_path):
    archive = TranscriptArchive(tmp_path)
    archive.save(_ep("a"), FULL)
    archive.annotate([RankedEpisode(
        episode=_ep("a"), score=81.25, rubric=RubricScore(), classification="Listen Fully",
        summary="Why it matters.", key_ideas=["one", "two"], assigned_category="ai_retail",
    )])
    archive.flush()
    entry = json.loads((tmp_path / "index.json").read_text())["episodes"][0]
    assert entry["score"] == round(81.25, 1)
    assert entry["key_ideas"] == ["one", "two"]
    assert entry["category"] == "ai_retail"


def test_index_survives_runs_and_retention_prunes_text(tmp_path):
    archive = TranscriptArchive(tmp_path, retention_days=30)
    archive.save(_ep("old"), FULL)
    archive.flush()
    index = json.loads((tmp_path / "index.json").read_text())
    index["episodes"][0]["archived_at"] = "2020-01-01T00:00:00+00:00"
    (tmp_path / "index.json").write_text(json.dumps(index))

    later = TranscriptArchive(tmp_path, retention_days=30)
    assert later.load("old") is not None
    later.save(_ep("new"), FULL)
    later.flush()
    index = json.loads((tmp_path / "index.json").read_text())
    assert [e["guid"] for e in index["episodes"]] == ["new"]
    assert not (tmp_path / "old.txt").exists()


def test_load_returns_archived_text_for_reuse(tmp_path):
    archive = TranscriptArchive(tmp_path)
    archive.save(_ep("a"), FULL)
    archive.flush()
    loaded = TranscriptArchive(tmp_path).load("a")
    assert loaded is not None and loaded.text == FULL.text and loaded.is_full
    assert TranscriptArchive(tmp_path).load("missing") is None
