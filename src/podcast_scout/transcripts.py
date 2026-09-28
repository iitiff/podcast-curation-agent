"""Durable archive of full episode transcripts.

The ranker reads a transcript and throws it away. Anything downstream that
wants to read what was actually said in an episode (an idea scout mining
interviews for pain points, a human checking a summary) had nothing to read,
so every full transcript is written here: one plain-text file per episode
plus an index.json describing them.

Only full transcripts are archived (publisher, Gemini audio or Whisper). A
description standing in for a transcript is already in the feeds and would
only make the archive look more complete than it is.

Layout, under TRANSCRIPTS_DIR (default DATA_DIR/transcripts):

    index.json      {generated_at, count, episodes: [entry, ...]}, newest first
    <guid>.txt      the transcript text

Entries older than the retention window are pruned together with their text,
so the directory does not grow without bound in a repo that commits it.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .normalize import NormalizedEpisode, utcnow
from .providers.base import TranscriptResult
from .ranking import RankedEpisode

log = logging.getLogger(__name__)

INDEX_NAME = "index.json"


class TranscriptArchive:
    def __init__(self, root: Path, retention_days: int = 60) -> None:
        self.root = root
        self.retention_days = retention_days
        self._entries: dict[str, dict[str, Any]] = {}
        index = root / INDEX_NAME
        if index.exists():
            try:
                for entry in json.loads(index.read_text()).get("episodes", []):
                    if isinstance(entry, dict) and entry.get("guid"):
                        self._entries[entry["guid"]] = entry
            except (OSError, ValueError) as exc:
                log.warning("Transcript index unreadable, starting fresh: %s", exc)

    def __len__(self) -> int:
        return len(self._entries)

    def load(self, guid: str) -> TranscriptResult | None:
        """A previously archived transcript, so a rescore never pays to transcribe twice."""
        entry = self._entries.get(guid)
        if entry is None:
            return None
        try:
            text = (self.root / entry.get("path", f"{guid}.txt")).read_text(encoding="utf-8")
        except OSError:
            return None
        if not text:
            return None
        return TranscriptResult(
            text=text, confidence="high", source=entry.get("transcript_source", "publisher"),
        )

    def save(self, ep: NormalizedEpisode, result: TranscriptResult) -> bool:
        """Write the transcript if it is a full one. Returns whether it was saved."""
        if not result.is_full:
            return False
        self.root.mkdir(parents=True, exist_ok=True)
        filename = f"{ep.guid}.txt"
        (self.root / filename).write_text(result.text, encoding="utf-8")
        self._entries[ep.guid] = {
            **self._entries.get(ep.guid, {}),
            "guid": ep.guid,
            "show": ep.show_title,
            "title": ep.episode_title,
            "published": ep.published.isoformat(),
            "episode_url": ep.episode_url,
            "feed_url": ep.source_feed_url,
            "audio_url": ep.enclosure.url if ep.enclosure else "",
            "duration_min": round(ep.duration_minutes),
            "category": ep.category,
            "transcript_source": result.source,
            "chars": len(result.text),
            "path": filename,
            "archived_at": utcnow().isoformat(),
        }
        return True

    def annotate(self, ranked: Iterable[RankedEpisode]) -> None:
        """Attach the curator's verdict, so a reader can prioritise without re-ranking."""
        for r in ranked:
            entry = self._entries.get(r.episode.guid)
            if entry is None:
                continue
            entry.update({
                "category": r.assigned_category or r.episode.category or entry.get("category", ""),
                "score": round(r.score, 1),
                "classification": r.classification,
                "summary": r.summary,
                "key_ideas": list(r.key_ideas),
            })

    def flush(self, now: datetime | None = None) -> Path:
        """Prune expired transcripts and write index.json."""
        now = now or utcnow()
        if self.retention_days > 0:
            cutoff = now - timedelta(days=self.retention_days)
            for guid, entry in list(self._entries.items()):
                if _parse(entry.get("archived_at")) < cutoff:
                    (self.root / entry.get("path", f"{guid}.txt")).unlink(missing_ok=True)
                    del self._entries[guid]
        # Drop entries whose text went missing, so the index never points at nothing.
        for guid, entry in list(self._entries.items()):
            if not (self.root / entry.get("path", f"{guid}.txt")).exists():
                del self._entries[guid]

        episodes = sorted(
            self._entries.values(),
            key=lambda e: (e.get("published", ""), e.get("guid", "")),
            reverse=True,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        index = self.root / INDEX_NAME
        index.write_text(json.dumps({
            "generated_at": now.isoformat(),
            "count": len(episodes),
            "episodes": episodes,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return index


def _parse(value: Any) -> datetime:
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=UTC)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
