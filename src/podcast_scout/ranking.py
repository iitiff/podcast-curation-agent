"""Two-stage episode ranking engine."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from .config import Preferences
from .normalize import NormalizedEpisode
from .providers.base import BaseLLMProvider, LLMMessage, TranscriptResult

log = logging.getLogger(__name__)


class RubricScore(BaseModel):
    relevance: float = 0        # 0-30
    novelty: float = 0          # 0-15
    guest_authority: float = 0  # 0-15
    actionability: float = 0    # 0-15
    evidence: float = 0         # 0-10
    strategic_importance: float = 0  # 0-10
    learning_per_minute: float = 0   # 0-5
    # Penalties (negative values)
    repetition_penalty: float = 0    # up to -15
    generic_penalty: float = 0       # up to -15
    weak_evidence_penalty: float = 0 # up to -10
    confidence_penalty: float = 0    # up to -15
    motivational_penalty: float = 0  # up to -10
    relevance_penalty: float = 0     # up to -20

    @property
    def total(self) -> float:
        base = (
            self.relevance + self.novelty + self.guest_authority
            + self.actionability + self.evidence
            + self.strategic_importance + self.learning_per_minute
        )
        penalties = (
            self.repetition_penalty + self.generic_penalty
            + self.weak_evidence_penalty + self.confidence_penalty
            + self.motivational_penalty + self.relevance_penalty
        )
        return max(0.0, min(100.0, base + penalties))


class Stage1Result(BaseModel):
    guid: str
    score: float
    reason: str
    should_deep_process: bool


class RankedEpisode(BaseModel):
    episode: NormalizedEpisode
    score: float
    rubric: RubricScore
    classification: str  # Listen Fully | Read Summary Only | Skip
    classification_reason: str = ""
    evidence_confidence: str = "low"  # high | medium | low
    summary: str = ""
    key_ideas: list[str] = Field(default_factory=list)
    implications: str = ""
    who_should_listen: str = ""
    summary_captures_value: str = ""
    listen_nuance: str = ""
    transcript_source: str = "none"
    tokens_used: int = 0


class _Stage1LLMOutput(BaseModel):
    score: float
    reason: str
    should_deep_process: bool


class _Stage2LLMOutput(BaseModel):
    rubric: dict[str, float]
    classification: str
    classification_reason: str
    summary: str
    key_ideas: list[str]
    implications: str
    who_should_listen: str
    summary_captures_value: str
    listen_nuance: str


def _show_prior(show_title: str, prefs: Preferences) -> float:
    for name, prior in prefs.show_priors.items():
        if name.lower() in show_title.lower():
            return prior
    return 0.5


def _is_acquired(show_title: str) -> bool:
    return "acquired" in show_title.lower()


def _parse_llm_json(raw: str) -> dict:
    """Robustly parse JSON from LLM output.

    Handles two common failure modes:
    1. The model wraps output in a ```json ... ``` markdown fence.
    2. The model produces slightly malformed JSON (unterminated strings,
       trailing commas).  We try stdlib json first; if that fails we
       attempt a manual fence-strip + second parse before giving up.
    """
    # Strip leading/trailing whitespace
    text = raw.strip()

    # Remove markdown code fence if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # First attempt: standard json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second attempt: truncate at last complete top-level value.
    # Find the last } that could close the root object and try up to that.
    last_brace = text.rfind("}")
    if last_brace != -1:
        try:
            return json.loads(text[: last_brace + 1])
        except json.JSONDecodeError:
            pass

    # Third attempt: use json-repair if available
    try:
        import json_repair  # type: ignore
        return json_repair.loads(text)  # type: ignore[return-value]
    except Exception:
        pass

    raise ValueError(f"Could not parse LLM JSON output (length={len(raw)})")


def stage1_metadata_score(ep: NormalizedEpisode, prefs: Preferences) -> Stage1Result:
    """Fast metadata-only score. No LLM call."""
    score = 0.0
    reasons = []

    # Interest keyword matching
    text = f"{ep.show_title} {ep.episode_title} {ep.description}".lower()
    interest_hits = sum(
        weight for topic, weight in prefs.interests.items()
        if topic.replace("_", " ") in text
    )
    relevance = min(30.0, interest_hits * 8)
    score += relevance

    # Show prior boost
    prior = _show_prior(ep.show_title, prefs)
    score += prior * 15
    reasons.append(f"show_prior={prior:.2f}")

    # Guest watchlist boost
    for guest in prefs.guest_watchlist:
        if guest.lower() in text:
            score += 8
            reasons.append(f"guest:{guest}")
            break

    # Competitor watchlist boost
    competitor_hits = sum(1 for c in prefs.competitor_watchlist if c.lower() in text)
    score += min(10.0, competitor_hits * 3)

    # Duration penalty
    dur = ep.duration_minutes
    if dur > 0:
        if dur < prefs.length.preferred_min_minutes:
            score -= 5
        elif dur > prefs.length.hard_max_minutes and not _is_acquired(ep.show_title):
            score -= 10

    # Topic exclusion penalty
    for excl in prefs.topic_exclusions:
        if excl.lower() in text:
            score -= 20
            reasons.append(f"excluded_topic:{excl}")

    score = max(0.0, min(100.0, score))
    should_deep = score >= 35 or any(g.lower() in text for g in prefs.guest_watchlist)

    return Stage1Result(
        guid=ep.guid,
        score=score,
        reason=", ".join(reasons) or "metadata_baseline",
        should_deep_process=should_deep,
    )


async def stage2_deep_rank(
    ep: NormalizedEpisode,
    transcript: TranscriptResult,
    prefs: Preferences,
    llm: BaseLLMProvider,
    token_budget: int = 3000,
) -> RankedEpisode:
    """Full LLM-powered ranking with rubric scoring."""
    persona_ctx = (
        f"You are ranking podcasts for a {prefs.persona.seniority} {prefs.persona.role} "
        f"whose focus is: {prefs.persona.focus}. "
        f"Preferred depth: {prefs.persona.preferred_depth}."
    )

    source_text = transcript.text[:6000] if transcript.text else (
        f"{ep.episode_title}\n\n{ep.description}"
    )
    confidence = transcript.confidence

    system_prompt = f"""{persona_ctx}

Score this podcast episode on a 100-point rubric and produce a structured JSON output.

RUBRIC (base points):
- relevance: 0-30 (how closely does it match the user's interest areas)
- novelty: 0-15 (new insight, not a rehash)
- guest_authority: 0-15 (firsthand expertise)
- actionability: 0-15 (can the user act on this)
- evidence: 0-10 (data, case studies, concrete examples)
- strategic_importance: 0-10 (timeliness, competitive relevance)
- learning_per_minute: 0-5 (density of value)

PENALTIES (negative):
- repetition_penalty: 0 to -15
- generic_penalty: 0 to -15
- weak_evidence_penalty: 0 to -10
- confidence_penalty: 0 to -15 (use -{15*(1 if confidence=='low' else 5 if confidence=='medium' else 0)} as baseline)
- motivational_penalty: 0 to -10
- relevance_penalty: 0 to -20

CLASSIFICATION (based on total score):
- "Listen Fully" if score >= 75
- "Read Summary Only" if score >= 50
- "Skip" if below 50
You may adjust boundary by up to 5 points with written justification.

NEVER invent specific claims not present in the source text when confidence is low.
Source confidence: {confidence}

IMPORTANT: Return ONLY a raw JSON object. Do NOT wrap in markdown code fences.
All string values must be properly escaped. Do NOT use boolean values for string fields.
"""

    user_msg = f"""SHOW: {ep.show_title}
EPISODE: {ep.episode_title}
GUESTS: {', '.join(ep.guests) or 'unknown'}
DURATION: {ep.duration_minutes:.0f} min
SOURCE TEXT ({confidence} confidence):
{source_text}

Return JSON with keys: rubric (dict of all rubric+penalty fields), classification, classification_reason,
summary (150-300 words), key_ideas (list of 2-3 strings), implications, who_should_listen,
summary_captures_value (string: "yes" | "partial" | "no"), listen_nuance (string)."""

    tokens_used = 0
    try:
        resp = await llm.complete(
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_msg),
            ],
            max_tokens=token_budget,
        )
        tokens_used = resp.input_tokens + resp.output_tokens
        data = _parse_llm_json(resp.content)
        rubric_data = data.get("rubric", {})
        rubric = RubricScore(**{k: float(v) for k, v in rubric_data.items() if k in RubricScore.model_fields})
        score = rubric.total

        # Coerce potentially-boolean fields to str before Pydantic validation
        def _str(val: Any, fallback: str = "") -> str:
            if val is None:
                return fallback
            if isinstance(val, bool):
                return "yes" if val else "no"
            return str(val)

        return RankedEpisode(
            episode=ep,
            score=score,
            rubric=rubric,
            classification=data.get("classification", _classify(score, prefs)),
            classification_reason=_str(data.get("classification_reason")),
            evidence_confidence=confidence,
            summary=_str(data.get("summary")),
            key_ideas=data.get("key_ideas", []),
            implications=_str(data.get("implications")),
            who_should_listen=_str(data.get("who_should_listen")),
            summary_captures_value=_str(data.get("summary_captures_value")),
            listen_nuance=_str(data.get("listen_nuance")),
            transcript_source=transcript.source,
            tokens_used=tokens_used,
        )
    except Exception as exc:
        log.warning("Stage 2 ranking failed for %s: %s", ep.episode_title, exc)
        # Fallback: use metadata score
        s1 = stage1_metadata_score(ep, prefs)
        rubric = RubricScore(relevance=min(30, s1.score * 0.4))
        return RankedEpisode(
            episode=ep,
            score=s1.score,
            rubric=rubric,
            classification=_classify(s1.score, prefs),
            classification_reason="LLM unavailable; metadata fallback",
            evidence_confidence="low",
            summary=ep.description[:300] if ep.description else "Summary unavailable.",
            transcript_source="none",
            tokens_used=tokens_used,
        )


def _classify(score: float, prefs: Preferences) -> str:
    if score >= prefs.classification.listen_fully_min_score:
        return "Listen Fully"
    if score >= prefs.classification.read_summary_min_score:
        return "Read Summary Only"
    return "Skip"


def build_daily_queue(
    ranked: list[RankedEpisode],
    max_minutes: float,
    max_listen_fully: int,
    max_read_summary: int,
    max_outside: int,
) -> tuple[list[RankedEpisode], list[RankedEpisode]]:
    """Split ranked episodes into (queued_for_rss, email_only).
    Acquired episodes are always queued regardless of time budget.
    Returns (rss_queue, email_only).
    """
    sorted_eps = sorted(ranked, key=lambda r: r.score, reverse=True)

    rss_queue: list[RankedEpisode] = []
    email_only: list[RankedEpisode] = []
    total_minutes = 0.0
    listen_count = 0
    summary_count = 0
    outside_count = 0

    for r in sorted_eps:
        if r.classification == "Skip":
            continue  # silently dropped

        is_acquired = _is_acquired(r.episode.show_title)
        dur = r.episode.duration_minutes or 30  # default 30 min if unknown
        is_outside = r.episode.is_outside_feed

        # Check caps
        if r.classification == "Listen Fully" and listen_count >= max_listen_fully:
            email_only.append(r)
            continue
        if r.classification == "Read Summary Only" and summary_count >= max_read_summary:
            email_only.append(r)
            continue
        if is_outside and outside_count >= max_outside:
            email_only.append(r)
            continue

        # Time budget (Acquired bypasses)
        if not is_acquired and total_minutes + dur > max_minutes:
            email_only.append(r)
            continue

        rss_queue.append(r)
        total_minutes += dur if not is_acquired else 0  # don't count Acquired against budget
        if r.classification == "Listen Fully":
            listen_count += 1
        elif r.classification == "Read Summary Only":
            summary_count += 1
        if is_outside:
            outside_count += 1

    return rss_queue, email_only
