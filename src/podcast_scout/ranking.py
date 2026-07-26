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


def _is_followed(show_title: str, prefs: Preferences) -> bool:
    title_lower = show_title.lower()
    return any(name.lower() in title_lower for name in prefs.show_priors)


def _is_acquired(show_title: str) -> bool:
    return "acquired" in show_title.lower()


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """Robustly parse JSON from LLM output."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        result: dict[str, Any] = json.loads(text)
        return result
    except json.JSONDecodeError:
        pass
    last_brace = text.rfind("}")
    if last_brace != -1:
        try:
            result = json.loads(text[: last_brace + 1])
            return result
        except json.JSONDecodeError:
            pass
    try:
        import json_repair  # type: ignore
        repaired: dict[str, Any] = json_repair.loads(text)
        return repaired
    except Exception:
        pass
    raise ValueError(f"Could not parse LLM JSON output (length={len(raw)})")


def _parse_llm_json_array(raw: str) -> list[Any]:
    """Parse a JSON array from LLM output, handling markdown fences."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result  # type: ignore[return-value]
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v  # type: ignore[return-value]
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result  # type: ignore[return-value]
        except json.JSONDecodeError:
            pass
    try:
        import json_repair  # type: ignore
        repaired = json_repair.loads(text)
        if isinstance(repaired, list):
            return repaired  # type: ignore[return-value]
        if isinstance(repaired, dict):
            for v in repaired.values():
                if isinstance(v, list):
                    return v  # type: ignore[return-value]
    except Exception:
        pass
    raise ValueError(f"Could not parse LLM JSON array (length={len(raw)})")


def stage1_metadata_score(ep: NormalizedEpisode, prefs: Preferences) -> Stage1Result:
    """Fast metadata-only pre-filter score. No LLM call.

    When running without a Gemini key, this score is the final score.
    Followed shows are guaranteed a minimum score of 50 (Read Summary)
    so they always surface rather than being silently dropped.
    Scoring is otherwise based on show priors, guest watchlist,
    competitor watchlist, duration, and topic exclusions.
    """
    score = 0.0
    reasons: list[str] = []

    text = f"{ep.show_title} {ep.episode_title} {ep.description}".lower()

    # Show prior boost — primary signal
    prior = _show_prior(ep.show_title, prefs)
    score += prior * 40  # scaled to give 0.95-prior shows a score of 38
    reasons.append(f"show_prior={prior:.2f}")

    # Guest watchlist boost
    for guest in prefs.guest_watchlist:
        if guest.lower() in text:
            score += 15
            reasons.append(f"guest:{guest}")
            break

    # Competitor / entity watchlist boost
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

    # Floor: any followed show always surfaces as at least Read Summary
    # so the daily queue is never empty just because no LLM key is set.
    if _is_followed(ep.show_title, prefs) and score < 50:
        score = 50.0
        reasons.append("followed_show_floor")

    should_deep = score >= 35 or any(g.lower() in text for g in prefs.guest_watchlist)

    return Stage1Result(
        guid=ep.guid,
        score=score,
        reason=", ".join(reasons) or "metadata_baseline",
        should_deep_process=should_deep,
    )


def _classify(score: float, prefs: Preferences) -> str:
    """Assign an episode action from its score and configured thresholds."""
    if score >= prefs.classification.listen_fully_min_score:
        return "Listen Fully"
    if score >= prefs.classification.read_summary_min_score:
        return "Read Summary Only"
    return "Skip"


def _build_episode_block(idx: int, ep: NormalizedEpisode, transcript: TranscriptResult) -> str:
    source_text = transcript.text[:2000] if transcript.text else (
        f"{ep.episode_title}\n\n{ep.description}"
    )
    return (
        f"--- EPISODE {idx} ---\n"
        f"SHOW: {ep.show_title}\n"
        f"EPISODE: {ep.episode_title}\n"
        f"GUESTS: {', '.join(ep.guests) or 'unknown'}\n"
        f"DURATION: {ep.duration_minutes:.0f} min\n"
        f"TRANSCRIPT CONFIDENCE: {transcript.confidence}\n"
        f"TEXT:\n{source_text}\n"
    )


async def stage2_batch_rank(
    items: list[tuple[NormalizedEpisode, TranscriptResult]],
    prefs: Preferences,
    llm: BaseLLMProvider,
    token_budget: int = 8000,
) -> list[RankedEpisode]:
    """Rank multiple episodes in a SINGLE LLM call to conserve API quota."""
    if not items:
        return []

    persona_ctx = (
        f"You are ranking podcasts for a {prefs.persona.seniority} {prefs.persona.role} "
        f"whose focus is: {prefs.persona.focus}. "
        f"Preferred depth: {prefs.persona.preferred_depth}."
    )

    episode_blocks = "\n".join(
        _build_episode_block(i, ep, tr) for i, (ep, tr) in enumerate(items)
    )

    system_prompt = f"""{persona_ctx}

You will receive {len(items)} podcast episode(s). Score EACH on a 100-point rubric and return a
JSON ARRAY (one object per episode, in the same order). Do NOT wrap in markdown fences.

RUBRIC (base points):
- relevance: 0-30
- novelty: 0-15
- guest_authority: 0-15
- actionability: 0-15
- evidence: 0-10
- strategic_importance: 0-10
- learning_per_minute: 0-5

PENALTIES (negative):
- repetition_penalty: 0 to -15
- generic_penalty: 0 to -15
- weak_evidence_penalty: 0 to -10
- confidence_penalty: 0 to -15
- motivational_penalty: 0 to -10
- relevance_penalty: 0 to -20

CLASSIFICATION:
- "Listen Fully" if total >= 75
- "Read Summary Only" if total >= 50
- "Skip" otherwise

For EACH episode return an object with keys:
  rubric (dict), classification, classification_reason,
  summary (100-200 words), key_ideas (list of 2-3 strings),
  implications, who_should_listen, summary_captures_value ("yes"|"partial"|"no"), listen_nuance

Return ONLY a raw JSON array of {len(items)} objects. No prose, no markdown."""

    user_msg = f"Rank these {len(items)} episode(s):\n\n{episode_blocks}"

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
        entries = _parse_llm_json_array(resp.content)
    except Exception as exc:
        log.warning("Batch Stage 2 LLM call failed: %s — falling back to metadata for all", exc)
        entries = []

    results: list[RankedEpisode] = []
    for i, (ep, transcript) in enumerate(items):
        data = entries[i] if i < len(entries) and isinstance(entries[i], dict) else {}
        if not data:
            log.warning(
                "No batch result for episode %d (%s) — using metadata fallback",
                i,
                ep.episode_title,
            )
            s1 = stage1_metadata_score(ep, prefs)
            results.append(
                RankedEpisode(
                    episode=ep,
                    score=s1.score,
                    rubric=RubricScore(relevance=min(30, s1.score * 0.4)),
                    classification=_classify(s1.score, prefs),
                    classification_reason="LLM batch entry missing; metadata fallback",
                    evidence_confidence="low",
                    summary=ep.description[:300] or "Summary unavailable.",
                    transcript_source=transcript.source,
                    tokens_used=0,
                )
            )
            continue

        try:
            rubric_data = data.get("rubric", {})
            rubric = RubricScore(
                **{
                    key: float(value)
                    for key, value in rubric_data.items()
                    if key in RubricScore.model_fields
                }
            )
            score = rubric.total

            def _str(value: Any, fallback: str = "") -> str:
                if value is None:
                    return fallback
                if isinstance(value, bool):
                    return "yes" if value else "no"
                return str(value)

            key_ideas = data.get("key_ideas", [])
            if not isinstance(key_ideas, list):
                key_ideas = []

            results.append(
                RankedEpisode(
                    episode=ep,
                    score=score,
                    rubric=rubric,
                    classification=_str(
                        data.get("classification"),
                        _classify(score, prefs),
                    ),
                    classification_reason=_str(
                        data.get("classification_reason")
                    ),
                    evidence_confidence=transcript.confidence,
                    summary=_str(data.get("summary")),
                    key_ideas=[_str(idea) for idea in key_ideas],
                    implications=_str(data.get("implications")),
                    who_should_listen=_str(data.get("who_should_listen")),
                    summary_captures_value=_str(
                        data.get("summary_captures_value")
                    ),
                    listen_nuance=_str(data.get("listen_nuance")),
                    transcript_source=transcript.source,
                    tokens_used=tokens_used // len(items),
                )
            )
        except Exception as exc:
            log.warning(
                "Could not parse batch entry %d for %s: %s",
                i,
                ep.episode_title,
                exc,
            )
            s1 = stage1_metadata_score(ep, prefs)
            results.append(
                RankedEpisode(
                    episode=ep,
                    score=s1.score,
                    rubric=RubricScore(relevance=min(30, s1.score * 0.4)),
                    classification=_classify(s1.score, prefs),
                    classification_reason="batch parse error; metadata fallback",
                    evidence_confidence="low",
                    summary=ep.description[:300] or "Summary unavailable.",
                    transcript_source=transcript.source,
                    tokens_used=0,
                )
            )

    return results
