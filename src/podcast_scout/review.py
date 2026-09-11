"""Monthly "State of My Thinking" — a review of the brain, not of a week's feed.

The weekly synthesis summarises what arrived. This asks a different question:
given everything accumulated, what has actually changed in what the reader
believes? It therefore reads the brain (open questions and the findings
attached to them, plus the month's Source pages) rather than the ranked queue,
and it is the only artifact that can say "this question is answered" or "the
evidence here has stopped moving".
"""
from __future__ import annotations

import json
import logging
import re
from datetime import timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from .brain.schema import Source, utcnow
from .brain.store import BrainStore
from .providers.base import BaseLLMProvider, LLMMessage

log = logging.getLogger(__name__)

# Enough to characterise a month without blowing a free-tier request budget on
# a single call. Sources are the long tail; questions are the point.
_MAX_SOURCES = 40
_MAX_FINDINGS_PER_QUESTION = 12


class QuestionState(BaseModel):
    question_id: str = ""
    question: str = ""
    verdict: str = "still open"   # still open | converging | answered | should be retired
    what_changed: str = ""
    next_probe: str = ""


class MonthlyReview(BaseModel):
    period: str = ""
    headline: str = ""
    question_states: list[QuestionState] = Field(default_factory=list)
    changed_my_mind: str = ""
    stopped_believing: str = ""
    blind_spot: str = ""
    source_diet: str = ""
    confidence: str = "low"

    def to_markdown(self) -> str:
        lines = [
            f"# State of My Thinking — {self.period}\n",
            f"**{self.headline}**\n" if self.headline else "",
            "## Where each open question stands\n",
        ]
        for qs in self.question_states:
            lines.append(f"### {qs.question or qs.question_id}")
            lines.append(f"- **Verdict:** {qs.verdict}")
            if qs.what_changed:
                lines.append(f"- **What changed:** {qs.what_changed}")
            if qs.next_probe:
                lines.append(f"- **Next probe:** {qs.next_probe}")
            lines.append("")
        if self.changed_my_mind:
            lines.append(f"## What changed my mind\n\n{self.changed_my_mind}\n")
        if self.stopped_believing:
            lines.append(f"## What I stopped believing\n\n{self.stopped_believing}\n")
        if self.blind_spot:
            lines.append(f"## Blind spot\n\n{self.blind_spot}\n")
        if self.source_diet:
            lines.append(f"## Source diet\n\n{self.source_diet}\n")
        lines.append(f"\n_Confidence: {self.confidence}._")
        return "\n".join(line for line in lines if line is not None)


def _recent_sources(store: BrainStore, lookback_days: int) -> list[Source]:
    """Source pages published within the window, newest first.

    Source ids begin with the publish date (see source_id_for), so the window
    is applied to the id rather than to file mtime -- a re-cloned repo has
    today's mtime on every file.
    """
    cutoff = (utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    directory = store.dir_for("Source")
    if not directory.exists():
        return []
    sources: list[Source] = []
    for path in sorted(directory.glob("*.md"), reverse=True):
        if path.stem[:10] < cutoff:
            continue
        page = store.load_page("Source", path.stem)
        if isinstance(page, Source):
            sources.append(page)
        if len(sources) >= _MAX_SOURCES:
            break
    return sources


def _findings_for(body: str) -> list[str]:
    """Pull the bullet lines out of a question page's Findings section."""
    match = re.search(
        r"^##+\s*Findings\s*$(.*?)(?=^##+\s|\Z)", body, re.MULTILINE | re.DOTALL
    )
    if not match:
        return []
    lines = [
        line.strip()
        for line in match.group(1).splitlines()
        if line.strip().startswith("-")
    ]
    return lines[-_MAX_FINDINGS_PER_QUESTION:]


def build_review_prompt(store: BrainStore, lookback_days: int, period: str) -> str:
    questions = store.load_questions(open_only=True)
    sources = _recent_sources(store, lookback_days)

    question_blocks = []
    for q in questions:
        findings = _findings_for(q.body)
        findings_text = "\n".join(f"    {f}" for f in findings) or "    (no findings yet)"
        question_blocks.append(
            f"- id: {q.id}\n"
            f"  question: {q.question or q.title}\n"
            f"  why it matters: {q.why}\n"
            f"  findings attached this period:\n{findings_text}"
        )
    questions_text = "\n".join(question_blocks) or "(no open questions)"

    # Credibility and source_type are carried into the prompt deliberately: the
    # review is supposed to notice when a month's evidence is mostly vendor
    # material, which is exactly the failure the source classes exist to catch.
    source_lines = [
        f"- [{s.source_type}/{s.credibility}] {s.title}: {(s.summary or '')[:200]}"
        for s in sources
    ]
    sources_text = "\n".join(source_lines) or "(no sources recorded)"

    return f"""You are reviewing a product leader's accumulated research for {period}.

Their open questions and what has attached to each:
{questions_text}

Sources recorded this period ({len(sources)} shown, tagged with type/credibility):
{sources_text}

Be blunt and specific. Say "the evidence did not move" when it did not; do not
manufacture progress. Judge each question only on the findings listed under it.
Note when the evidence is dominated by low-credibility vendor material.

Return a JSON object with these keys:
- headline: string, one sentence on what actually changed this period
- question_states: list of objects with keys question_id, question, verdict
  ("still open" | "converging" | "answered" | "should be retired"),
  what_changed, next_probe (one concrete thing to look for next)
- changed_my_mind: string
- stopped_believing: string
- blind_spot: string (what this source diet cannot see)
- source_diet: string (one line on the mix and skew of source types)
- confidence: "high" | "medium" | "low"

Return ONLY raw JSON, no markdown."""


async def generate_monthly_review(
    store: BrainStore,
    llm: BaseLLMProvider,
    lookback_days: int = 30,
    period: str = "",
) -> MonthlyReview | None:
    """Generate the review. Returns None on any failure; never raises."""
    period = period or utcnow().strftime("%Y-%m")
    questions = store.load_questions(open_only=True)
    if not questions and not _recent_sources(store, lookback_days):
        log.info("Nothing in the brain to review for %s.", period)
        return None

    prompt = build_review_prompt(store, lookback_days, period)
    try:
        resp = await llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            max_tokens=1600,
        )
        text = resp.content.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        data = json.loads(text)
        return MonthlyReview(period=period, **data)
    except Exception as exc:
        log.warning("Monthly review generation failed: %s", exc)
        return None


def write_review(store: BrainStore, review: MonthlyReview) -> Path:
    """Write the review into the brain's syntheses directory."""
    directory = store.root / "syntheses"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{review.period}-state-of-my-thinking.md"
    path.write_text(review.to_markdown(), encoding="utf-8")
    return path
