"""Falsifier watch: check incoming signals against what would change your mind.

Every active Thesis declares a `falsifier`. This module asks the LLM, in one
batched call, whether any of today's signals actually bears on those theses --
and specifically whether anything trips a falsifier.

This is the one part of the pipeline that exists to *disagree* with the reader.
Ranking optimises for relevance, which by construction surfaces material that
matches existing interests; without an explicit check against falsifiers, a
brain accumulates confirmation and calls it learning.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from ..providers.base import BaseLLMProvider, LLMMessage
from .schema import Thesis

log = logging.getLogger(__name__)

# Batched in one call rather than theses x signals calls. A brain with 8 active
# theses and 20 signals would otherwise be 160 LLM round-trips per run.
MAX_SIGNALS_PER_CHECK = 25


class FalsifierHit(BaseModel):
    thesis_id: str
    thesis_title: str = ""
    signal_id: str
    signal_title: str = ""
    direction: str  # supports | challenges
    strength: str = "moderate"  # strong | moderate | weak
    reasoning: str = ""

    @property
    def is_challenge(self) -> bool:
        return self.direction == "challenges"


class SignalInput(BaseModel):
    """A minimal, source-agnostic view of something that entered the brain."""

    id: str
    title: str
    summary: str = ""
    origin: str = ""


def _parse_json_array(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("Falsifier check returned unparseable JSON: %s", exc)
        return []
    return parsed if isinstance(parsed, list) else []


def _build_prompt(theses: list[Thesis], signals: list[SignalInput]) -> str:
    thesis_block = "\n".join(
        f'- id: {t.id}\n  statement: "{t.statement or t.title}"\n  '
        f'falsifier: "{t.falsifier or "(none stated)"}"\n  confidence: {t.confidence}'
        for t in theses
    )
    signal_block = "\n".join(
        f'- id: {s.id}\n  title: "{s.title}"\n  summary: "{s.summary[:400]}"'
        for s in signals
    )
    return f"""You are testing a product leader's stated beliefs against new evidence.

ACTIVE THESES (each has a falsifier -- the evidence its holder said would change their mind):
{thesis_block}

NEW SIGNALS:
{signal_block}

For each signal that genuinely bears on a thesis, emit one object:
  thesis_id      - the thesis id
  signal_id      - the signal id
  direction      - "supports" or "challenges"
  strength       - "strong" | "moderate" | "weak"
  reasoning      - one sentence, concrete, citing what in the signal does the work

Rules:
- Be strict. Most signals bear on no thesis. Returning [] is the correct and
  expected answer on most runs.
- Only mark "challenges" when the signal genuinely cuts against the thesis or
  moves toward its stated falsifier. Do not manufacture disagreement.
- Topical overlap is NOT support. A signal that merely mentions the same
  subject as a thesis, without bearing on whether it is true, is not a hit.
- Never mark "strong" unless the evidence would move a reasonable person's
  confidence by a full level.

Return ONLY a raw JSON array. No markdown, no prose."""


async def check_falsifiers(
    theses: list[Thesis],
    signals: list[SignalInput],
    llm: BaseLLMProvider | None,
) -> list[FalsifierHit]:
    """Return the signals that bear on active theses. Never raises."""
    if not theses or not signals or llm is None:
        return []

    capped = signals[:MAX_SIGNALS_PER_CHECK]
    by_id = {t.id: t for t in theses}
    signal_titles = {s.id: s.title for s in capped}

    try:
        resp = await llm.complete(
            messages=[LLMMessage(role="user", content=_build_prompt(theses, capped))],
            max_tokens=1500,
        )
    except Exception as exc:
        # A brain that cannot run the check must still produce a brief; the
        # absence of hits is logged rather than mistaken for "nothing bears on
        # your beliefs today".
        log.warning("Falsifier check failed, skipping: %s", exc)
        return []

    hits: list[FalsifierHit] = []
    for raw in _parse_json_array(resp.content):
        if not isinstance(raw, dict):
            continue
        thesis = by_id.get(str(raw.get("thesis_id", "")))
        signal_id = str(raw.get("signal_id", ""))
        if thesis is None or signal_id not in signal_titles:
            # The model invented an id; drop it rather than writing a dangling
            # evidence line into the brain.
            continue
        direction = str(raw.get("direction", "")).lower()
        if direction not in {"supports", "challenges"}:
            continue
        hits.append(
            FalsifierHit(
                thesis_id=thesis.id,
                thesis_title=thesis.title,
                signal_id=signal_id,
                signal_title=signal_titles[signal_id],
                direction=direction,
                strength=str(raw.get("strength", "moderate")).lower(),
                reasoning=str(raw.get("reasoning", "")).strip(),
            )
        )

    # Challenges first, then by strength: the brief should lead with whatever
    # argues against the reader, not with comfortable agreement.
    order = {"strong": 0, "moderate": 1, "weak": 2}
    hits.sort(key=lambda h: (not h.is_challenge, order.get(h.strength, 3)))
    return hits
