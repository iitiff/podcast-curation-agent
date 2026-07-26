"""Persistent run state — tracks processed episodes and prevents duplicates."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .normalize import utcnow


class EpisodeRecord(BaseModel):
    guid: str
    show_title: str
    episode_title: str
    published: datetime
    processed_at: datetime
    score: float = 0.0
    classification: str = "Skip"  # Listen Fully | Read Summary Only | Skip
    is_outside_feed: bool = False
    source_feed_url: str = ""


class RunState(BaseModel):
    last_run: datetime | None = None
    processed_episodes: dict[str, EpisodeRecord] = Field(default_factory=dict)
    # GUIDs that have been published to RSS (used for retention logic)
    published_guids: list[str] = Field(default_factory=list)


class StateManager:
    def __init__(self, data_dir: Path, retention_days: int = 90) -> None:
        self._path = data_dir / "state.json"
        self._history_dir = data_dir / "history"
        self._retention_days = retention_days
        self._state = self._load()

    def _load(self) -> RunState:
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                return RunState.model_validate(raw)
            except Exception:
                return RunState()
        return RunState()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            self._state.model_dump_json(indent=2, exclude_none=False)
        )

    def seen_guids(self) -> set[str]:
        return set(self._state.processed_episodes.keys())

    def mark_processed(self, record: EpisodeRecord) -> None:
        self._state.processed_episodes[record.guid] = record

    def get_record(self, guid: str) -> EpisodeRecord | None:
        """Return the stored EpisodeRecord for a guid, or None."""
        return self._state.processed_episodes.get(guid)

    def is_published(self, guid: str) -> bool:
        return guid in self._state.published_guids

    def add_published(self, guid: str) -> None:
        if guid not in self._state.published_guids:
            self._state.published_guids.append(guid)

    def prune_old(self) -> None:
        cutoff = utcnow() - timedelta(days=self._retention_days)
        self._state.processed_episodes = {
            g: r
            for g, r in self._state.processed_episodes.items()
            if r.processed_at >= cutoff
        }
        # Rebuild published list from still-retained records
        retained = set(self._state.processed_episodes.keys())
        self._state.published_guids = [
            g for g in self._state.published_guids if g in retained
        ]

    def snapshot_history(self, run_date: str, payload: Any) -> None:
        self._history_dir.mkdir(parents=True, exist_ok=True)
        path = self._history_dir / f"{run_date}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))

    def update_last_run(self) -> None:
        self._state.last_run = utcnow()
