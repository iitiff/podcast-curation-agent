"""Persistent episode state manager."""
from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EpisodeRecord(BaseModel):
    """A processed episode as persisted to state.json.

    The `summary` / `key_ideas` / `episode_url` / `duration_seconds` fields
    exist because LLM analysis is expensive and only runs ONCE per episode.
    Without persisting them, an episode carried over to a later run was rebuilt
    as a stub with `summary=""` and `key_ideas=[]` -- so the insights were
    computed, shown in one email, then silently thrown away. Every subsequent
    digest showed that episode with no takeaways and a dead RSS-root link.

    All new fields default to empty, so older state.json files load unchanged.
    """

    guid: str
    show_title: str = ""
    episode_title: str = ""
    published: datetime | None = None
    processed_at: datetime | None = None
    score: float = 0.0
    classification: str = ""
    classification_reason: str = ""
    is_outside_feed: bool = False
    source_feed_url: str = ""
    # --- persisted LLM output + episode metadata (see docstring) ---
    summary: str = ""
    key_ideas: list[str] = Field(default_factory=list)
    episode_url: str = ""
    duration_seconds: int = 0
    # Enclosure must persist too: an item with no <enclosure> is not a playable
    # episode, and most podcast clients hide it entirely. Carried-over episodes
    # were losing theirs and silently becoming unplayable rows in the feed.
    enclosure_url: str = ""
    enclosure_type: str = "audio/mpeg"
    enclosure_length: int = 0
    # When this episode first entered a curated feed. Used as the feed <pubDate>
    # instead of the show's ORIGINAL publish date: clients read pubDate as "when
    # did this become available in THIS feed", so a 6-day-old episode newly
    # added to the playlist was sorting into the past and being filtered out as
    # already-seen rather than surfacing as new.
    curated_at: datetime | None = None


class StateManager:
    """Manages a persistent JSON store of processed episode GUIDs and state."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / "state.json"
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                loaded: dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))
                return loaded
            except Exception:
                pass
        return {"processed": {}, "published": [], "playlist": [], "emailed": [], "last_run": None}

    def save(self) -> None:
        self._path.write_text(json.dumps(self._state, indent=2, default=str), encoding="utf-8")

    def seen_guids(self) -> set[str]:
        return set(self._state.get("processed", {}).keys())

    def published_guids(self) -> set[str]:
        """Return the set of episode GUIDs that have appeared in any RSS feed or email digest."""
        return set(self._state.get("published", []))

    def playlist_guids(self) -> set[str]:
        """Return the set of episode GUIDs that have won a category feed slot (Pocket Casts queue).

        This is a strict subset of published_guids: an episode can be published to
        all.xml or the email digest without ever winning a category-feed slot.
        Only playlist_guids are excluded from the daily carryover pool — episodes
        that appeared in email/all.xml only are still eligible to compete for a
        category-feed slot on future days.
        """
        return set(self._state.get("playlist", []))

    def emailed_guids(self) -> set[str]:
        """Episode GUIDs that have already appeared in a sent email digest.

        Tracked SEPARATELY from `playlist` on purpose. Carryover eligibility
        uses `playlist` so an episode that missed a feed slot keeps competing on
        later days -- but that also meant it reappeared in every daily digest
        until it won a slot or aged out. Filtering the email on this set instead
        gives both properties: keep competing for the playlist, get emailed once.

        Only written after a digest is SENT successfully, so a send failure does
        not silently suppress an episode from the next attempt.
        """
        return set(self._state.get("emailed", []))

    def add_emailed(self, guid: str) -> None:
        if guid not in self._state.get("emailed", []):
            self._state.setdefault("emailed", []).append(guid)

    def all_records(self) -> list[EpisodeRecord]:
        """Return every processed episode as a parsed EpisodeRecord.

        Malformed rows are skipped rather than raising, so a single bad entry
        can't break state-wide operations like rescore selection.
        """
        records: list[EpisodeRecord] = []
        for rec_data in self._state.get("processed", {}).values():
            try:
                records.append(EpisodeRecord(**rec_data))
            except Exception:
                continue
        return records

    def mark_processed(self, record: EpisodeRecord) -> None:
        self._state.setdefault("processed", {})[record.guid] = record.model_dump(mode="json")

    def get_record(self, guid: str) -> EpisodeRecord | None:
        data = self._state.get("processed", {}).get(guid)
        if data:
            try:
                return EpisodeRecord(**data)
            except Exception:
                pass
        return None

    def forget_processed(self, guids: Iterable[str]) -> int:
        """Remove episodes from the processed map and return how many were removed.

        An episode in `processed` is treated as already-seen by dedup_episodes and
        is therefore never re-scored, even if its stored score came from a
        degraded run (e.g. the 2026-07-30..08-04 window when GitHub Models was
        returning 410 and every episode fell to the metadata floor). Forgetting
        an episode makes it eligible for discovery and a fresh LLM pass.

        Note this does NOT touch `published` / `playlist` / `emailed`: an
        episode that already won a feed slot or was already emailed keeps
        those records, so re-scoring can't cause a duplicate feed entry or a
        repeat email.
        """
        processed: dict[str, Any] = self._state.setdefault("processed", {})
        removed = 0
        for guid in list(guids):
            if guid in processed:
                del processed[guid]
                removed += 1
        return removed

    def add_published(self, guid: str) -> None:
        if guid not in self._state.get("published", []):
            self._state.setdefault("published", []).append(guid)

    def add_to_playlist(self, guid: str) -> None:
        """Record that this episode won a category-feed slot (Pocket Casts queue).

        Also marks the episode as published (superset relationship).
        """
        self.add_published(guid)
        if guid not in self._state.get("playlist", []):
            self._state.setdefault("playlist", []).append(guid)

    def update_last_run(self) -> None:
        self._state["last_run"] = datetime.now(tz=UTC).isoformat()

    def prune_old(self, max_age_days: int = 30) -> None:
        cutoff = (datetime.now(tz=UTC) - timedelta(days=max_age_days)).isoformat()
        processed: dict[str, Any] = self._state.get("processed", {})
        self._state["processed"] = {
            guid: rec for guid, rec in processed.items()
            if rec.get("processed_at", "") > cutoff
        }

    def snapshot_history(self, run_date: str, data: dict[str, Any]) -> None:
        history_dir = self.data_dir / "history"
        history_dir.mkdir(exist_ok=True)
        path = history_dir / f"{run_date}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
