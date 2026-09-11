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
from .schema import Question, Thesis

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


# ---------------------------------------------------------------------------
# Questions. The lighter half of the watch.
#
# A thesis asks its holder to commit to a belief and author a falsifier. A
# question asks only what they want to know, which is a far lower bar to clear
# honestly -- and an unmaintained thesis is worse than none, because it
# accumulates supporting evidence and starts to look like rigor.
# ---------------------------------------------------------------------------


class QuestionHit(BaseModel):
    question_id: str
    question_title: str = ""
    signal_id: str
    signal_title: str = ""
    # answers   -- moves toward an answer
    # complicates -- cuts against the obvious answer, or shows it is harder
    # extends   -- reframes or widens the question itself
    relation: str = "answers"
    strength: str = "moderate"  # strong | moderate | weak
    takeaway: str = ""

    @property
    def is_notable(self) -> bool:
        """Worth leading the brief with.

        `complicates` earns attention regardless of strength: a signal that
        makes a question harder is the one most easily skimmed past, and the
        one most likely to matter.
        """
        return self.relation == "complicates" or self.strength == "strong"


def _build_question_prompt(questions: list[Question], signals: list[SignalInput]) -> str:
    question_block = "\n".join(
        f'- id: {q.id}\n  question: "{q.question or q.title}"\n'
        f'  why_it_matters: "{q.why or "(not stated)"}"'
        for q in questions
    )
    signal_block = "\n".join(
        f'- id: {s.id}\n  title: "{s.title}"\n  summary: "{s.summary[:400]}"'
        for s in signals
    )
    return f"""You are triaging new material against a product leader's open questions.

OPEN QUESTIONS:
{question_block}

NEW SIGNALS:
{signal_block}

For each signal that genuinely bears on a question, emit one object:
  question_id - the question id
  signal_id   - the signal id
  relation    - "answers" | "complicates" | "extends"
  strength    - "strong" | "moderate" | "weak"
  takeaway    - one sentence on what it actually adds, concrete and specific

Rules:
- Be strict. Most signals bear on no question. Returning [] is the correct and
  expected answer on most runs.
- Topical overlap is NOT relevance. A signal that merely mentions the same
  subject, without adding anything to the question, is not a hit.
- "complicates" means it cuts against the obvious answer or shows the question
  is harder than it looked. Prefer it over "answers" when both could apply --
  an answer that arrives too easily is usually the reader's prior reflected
  back at them.
- Never mark "strong" unless someone tracking this question would change what
  they read or do next.

Return ONLY a raw JSON array. No markdown, no prose."""


async def check_questions(
    questions: list[Question],
    signals: list[SignalInput],
    llm: BaseLLMProvider | None,
) -> list[QuestionHit]:
    """Return the signals that bear on open questions. Never raises."""
    if not questions or not signals or llm is None:
        return []

    capped = signals[:MAX_SIGNALS_PER_CHECK]
    by_id = {q.id: q for q in questions}
    signal_titles = {s.id: s.title for s in capped}

    try:
        resp = await llm.complete(
            messages=[LLMMessage(role="user", content=_build_question_prompt(questions, capped))],
            max_tokens=1500,
        )
    except Exception as exc:
        log.warning("Question check failed, skipping: %s", exc)
        return []

    hits: list[QuestionHit] = []
    for raw in _parse_json_array(resp.content):
        if not isinstance(raw, dict):
            continue
        question = by_id.get(str(raw.get("question_id", "")))
        signal_id = str(raw.get("signal_id", ""))
        if question is None or signal_id not in signal_titles:
            # Invented id; drop it rather than writing a dangling finding.
            continue
        relation = str(raw.get("relation", "")).lower()
        if relation not in {"answers", "complicates", "extends"}:
            continue
        hits.append(
            QuestionHit(
                question_id=question.id,
                question_title=question.title,
                signal_id=signal_id,
                signal_title=signal_titles[signal_id],
                relation=relation,
                strength=str(raw.get("strength", "moderate")).lower(),
                takeaway=str(raw.get("takeaway", "")).strip(),
            )
        )

    order = {"strong": 0, "moderate": 1, "weak": 2}
    # Complications first: they are the least likely to be sought out and the
    # most likely to be worth the reader's attention.
    hits.sort(key=lambda h: (h.relation != "complicates", order.get(h.strength, 3)))
    return hits
