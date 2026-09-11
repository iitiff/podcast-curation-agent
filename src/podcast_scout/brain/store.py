"""Read/write access to the brain: a directory of markdown entity pages.

Deliberately a filesystem + git store with no database. Retrieval is ripgrep
plus a generated index.json until that genuinely stops working; the brain is
meant to outlive any index built over it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .schema import (
    BrainPage,
    Company,
    Pattern,
    Source,
    Thesis,
    append_to_section,
    iso_now,
    join_frontmatter,
    slugify,
    split_frontmatter,
    utcnow,
)

log = logging.getLogger(__name__)

# Only four entity types to start. Add a type when a real page will not fit in
# one of these, not in anticipation.
SUBDIRS: dict[str, str] = {
    "Thesis": "theses",
    "Source": "sources",
    "Pattern": "patterns",
    "Company": "companies",
}

EVIDENCE_SECTION = "Evidence"
COUNTER_EVIDENCE_SECTION = "Counter-evidence"


class BrainStore:
    """A brain rooted at a directory of markdown pages."""

    def __init__(self, root: Path) -> None:
        self.root = root

    # -- layout ---------------------------------------------------------
    def dir_for(self, page_type: str) -> Path:
        return self.root / SUBDIRS.get(page_type, page_type.lower() + "s")

    def path_for(self, page_type: str, page_id: str) -> Path:
        return self.dir_for(page_type) / f"{page_id}.md"

    def ensure_dirs(self) -> None:
        # .gitkeep because the brain lives in git and an empty directory is not
        # tracked: without it the scaffold silently vanishes on clone and the
        # first write lands in a directory the reader never saw.
        for sub in [*SUBDIRS.values(), "syntheses"]:
            directory = self.root / sub
            directory.mkdir(parents=True, exist_ok=True)
            keep = directory / ".gitkeep"
            if not keep.exists() and not any(directory.glob("*.md")):
                keep.touch()

    # -- reading --------------------------------------------------------
    def _read_page(self, path: Path) -> tuple[dict[str, Any], str] | None:
        try:
            meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
        except OSError as exc:
            log.warning("Could not read brain page %s: %s", path, exc)
            return None
        if not meta:
            # A page with no frontmatter is a hand-written note, not a typed
            # entity. Skipped rather than coerced, so it is never rewritten.
            return None
        return meta, body

    def load_theses(self, active_only: bool = True) -> list[Thesis]:
        theses: list[Thesis] = []
        directory = self.dir_for("Thesis")
        if not directory.exists():
            return theses
        for path in sorted(directory.glob("*.md")):
            page = self._read_page(path)
            if page is None:
                continue
            meta, body = page
            if meta.get("type") != "Thesis":
                continue
            try:
                thesis = Thesis(**{**meta, "body": body})
            except Exception as exc:
                log.warning("Skipping malformed thesis %s: %s", path.name, exc)
                continue
            if active_only and thesis.status != "active":
                continue
            theses.append(thesis)
        return theses

    def load_page(self, page_type: str, page_id: str) -> BrainPage | None:
        path = self.path_for(page_type, page_id)
        if not path.exists():
            return None
        page = self._read_page(path)
        if page is None:
            return None
        meta, body = page
        models: dict[str, type[BrainPage]] = {
            "Thesis": Thesis,
            "Source": Source,
            "Pattern": Pattern,
            "Company": Company,
        }
        model = models.get(page_type, BrainPage)
        try:
            return model(**{**meta, "body": body})
        except Exception as exc:
            log.warning("Skipping malformed page %s: %s", path.name, exc)
            return None

    # -- writing --------------------------------------------------------
    def save(self, page: BrainPage) -> Path:
        path = self.path_for(page.type, page.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        page.updated_at = utcnow()
        path.write_text(page.to_markdown(), encoding="utf-8")
        return path

    def write_source(self, source: Source, overwrite: bool = False) -> Path | None:
        """Write a Source page. Returns None if it already exists.

        Sources are immutable evidence: re-running the pipeline over the same
        item must not rewrite history, so an existing page is left alone unless
        overwrite is explicitly requested.
        """
        path = self.path_for("Source", source.id)
        if path.exists() and not overwrite:
            return None
        return self.save(source)

    def append_evidence(
        self,
        thesis_id: str,
        line: str,
        supports: bool = True,
    ) -> bool:
        """Append a dated evidence line to a thesis. Returns False if absent.

        Idempotent: an identical line already present is not appended twice, so
        a re-run over the same signals does not inflate the evidence trail.
        """
        path = self.path_for("Thesis", thesis_id)
        if not path.exists():
            log.warning("Cannot attach evidence: no thesis page %s", thesis_id)
            return False
        page = self._read_page(path)
        if page is None:
            return False
        meta, body = page
        if line.strip() in body:
            return False
        section = EVIDENCE_SECTION if supports else COUNTER_EVIDENCE_SECTION
        new_body = append_to_section(body, section, line)
        meta["updated_at"] = iso_now()
        path.write_text(join_frontmatter(meta, new_body), encoding="utf-8")
        return True

    # -- index ----------------------------------------------------------
    def build_index(self) -> dict[str, Any]:
        """Write index.json: a flat manifest of every typed page.

        Exists so a query client can see the whole brain in one read instead of
        walking the tree. It is derived state and safe to delete.
        """
        entries: list[dict[str, Any]] = []
        for page_type, sub in SUBDIRS.items():
            directory = self.root / sub
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.md")):
                page = self._read_page(path)
                if page is None:
                    continue
                meta, _ = page
                entries.append(
                    {
                        "type": meta.get("type", page_type),
                        "id": meta.get("id", path.stem),
                        "title": meta.get("title", path.stem),
                        "path": str(path.relative_to(self.root)),
                        "tags": meta.get("tags", []),
                        "updated_at": meta.get("updated_at", ""),
                    }
                )
        index: dict[str, Any] = {
            "generated_at": utcnow().isoformat(),
            "count": len(entries),
            "pages": entries,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
        return index


def source_id_for(published: str, show: str, title: str) -> str:
    """Stable, human-scannable Source id: date + show + episode title."""
    return f"{published}-{slugify(show, 24)}-{slugify(title, 40)}"
