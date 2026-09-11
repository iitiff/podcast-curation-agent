"""Durable brain page schema: frontmatter, page models, and section editing.

Every brain page is a markdown file with YAML frontmatter:

    ---
    type: Thesis
    id: thesis-personalization-as-decision-system
    ...
    ---

    ## Reasoning
    ...

The frontmatter carries the structured fields skills query on; the body carries
"compiled truth at the top, timeline below". Markdown stays the source of
truth so the brain outlives whatever indexes it.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime
from typing import Any

import yaml
from pydantic import BaseModel, Field

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)
_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    """Current UTC time in the same 'Z' form pydantic emits.

    Kept identical to the model serialization so a page edited in place does
    not end up with `updated_at: ...+00:00` next to `created_at: ...Z`, which
    makes every diff look like a format change.
    """
    return utcnow().isoformat().replace("+00:00", "Z")


def slugify(value: str, max_length: int = 60) -> str:
    """Lowercase ASCII slug safe for use as a filename and a stable page id."""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_STRIP_RE.sub("-", ascii_only.lower()).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug or "untitled"


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown page into (frontmatter dict, body).

    A page with no frontmatter returns an empty dict and the whole text, so
    hand-written notes dropped into the brain are never destroyed by a rewrite.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw = yaml.safe_load(match.group(1))
    meta: dict[str, Any] = raw if isinstance(raw, dict) else {}
    return meta, match.group(2)


def join_frontmatter(meta: dict[str, Any], body: str) -> str:
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).rstrip("\n")
    return f"---\n{front}\n---\n\n{body.lstrip('\n')}"


def _heading_matches(line: str, heading: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return False
    return stripped.lstrip("#").strip().lower() == heading.strip().lower()


def append_to_section(body: str, heading: str, line: str) -> str:
    """Append `line` to the end of the `## heading` section.

    The section is created at the end of the page if it does not exist. The
    line is inserted after the section's last non-blank line rather than at the
    very end of the section, so accumulated evidence stays inside its own
    heading instead of drifting into the following one.
    """
    lines = body.splitlines()
    start: int | None = None
    for i, current in enumerate(lines):
        if _heading_matches(current, heading):
            start = i
            break

    if start is None:
        prefix = body.rstrip("\n")
        separator = "\n\n" if prefix else ""
        return f"{prefix}{separator}## {heading}\n\n{line}\n"

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip().startswith("#"):
            end = i
            break

    insert_at = end
    while insert_at > start + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1

    # An empty section is just "## Heading" followed by blank lines; keep the
    # blank line after the heading so the first entry doesn't get glued to it.
    if insert_at == start + 1:
        lines.insert(insert_at, "")
        insert_at += 1

    lines.insert(insert_at, line)
    return "\n".join(lines) + ("\n" if body.endswith("\n") else "")


class BrainPage(BaseModel):
    """Fields common to every durable entity page."""

    type: str
    id: str
    title: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    tags: list[str] = Field(default_factory=list)
    # Governs the security boundary. Only `public` and `personal` pages belong
    # in the portable brain; `company-private` marks material that must not
    # travel with the brain to another employer.
    visibility: str = "personal"
    body: str = ""

    def to_markdown(self) -> str:
        meta = self.model_dump(exclude={"body"}, mode="json")
        return join_frontmatter(meta, self.body)


class ConfidenceEntry(BaseModel):
    date: datetime = Field(default_factory=utcnow)
    confidence: str
    note: str = ""


class Thesis(BrainPage):
    """A durable belief you are willing to defend, and what would break it."""

    type: str = "Thesis"
    statement: str = ""
    confidence: str = "Low"  # Low | Medium | High
    # The single most important field in the brain: without a falsifier a
    # thesis can only ever accumulate confirmation.
    falsifier: str = ""
    status: str = "active"  # active | retired
    confidence_history: list[ConfidenceEntry] = Field(default_factory=list)


class Source(BrainPage):
    """A piece of evidence that entered the brain, with its bias declared."""

    type: str = "Source"
    source_type: str = "podcast"
    origin: str = ""
    credibility: str = "medium"  # high | medium | low
    bias_notes: str = ""
    summary: str = ""


class Question(BrainPage):
    """An open question you are tracking.

    A lighter primitive than Thesis, and for most people a more honest one. A
    thesis asks you to commit to a belief and author a falsifier for it; a
    question asks only what you want to know. Both direct the watch at
    something, but a question carries no maintenance debt -- and an unmaintained
    thesis is worse than none, because it accumulates supporting evidence and
    starts to look like rigor.
    """

    type: str = "Question"
    question: str = ""
    why: str = ""
    status: str = "open"  # open | answered | parked


class Pattern(BrainPage):
    type: str = "Pattern"
    domain: str = ""
    description: str = ""


class Company(BrainPage):
    type: str = "Company"
    strategic_position: str = ""
