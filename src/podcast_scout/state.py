"""Persistent episode state manager."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class EpisodeRecord(BaseModel):
    guid: str
    show_title: str = ""
    episode_title: str = ""
    published: datetime | None = None
    processed_at: datetime | None = None
    score: float = 0.0
    classification: str = ""
    is_outside_feed: bool = False
    source_feed_url: str = ""


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
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"processed": {}, "published": [], "last_run": None}

    def save(self) -> None:
        self._path.write_text(json.dumps(self._state, indent=2, default=str), encoding="utf-8")

    def seen_guids(self) -> set[str]:
        return set(self._state.get("processed", {}).keys())

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

    def add_published(self, guid: str) -> None:
        if guid not in self._state.get("published", []):
            self._state.setdefault("published", []).append(guid)

    def update_last_run(self) -> None:
        self._state["last_run"] = datetime.utcnow().isoformat()

    def prune_old(self, max_age_days: int = 30) -> None:
        cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
        processed = self._state.get("processed", {})
        self._state["processed"] = {
            guid: rec for guid, rec in processed.items()
            if rec.get("processed_at", "") > cutoff
        }

    def snapshot_history(self, run_date: str, data: dict) -> None:
        history_dir = self.data_dir / "history"
        history_dir.mkdir(exist_ok=True)
        path = history_dir / f"{run_date}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
