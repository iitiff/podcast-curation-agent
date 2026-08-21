"""Two-stage episode ranking engine."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field

from .config import Preferences
from .normalize import NormalizedEpisode, clean_snippet
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


def _normalize_apostrophes(s: str) -> str:
    """Normalize curly/fancy apostrophes to straight ASCII so prior lookups
    match regardless of whether the RSS feed uses ’ vs '.
    """
    return unicodedata.normalize("NFKD", s).replace("’", "'").replace("‘", "'")


def _show_prior(show_title: str, prefs: Preferences) -> float:
    title = _normalize_apostrophes(show_title).lower()
    for name, prior in prefs.show_priors.items():
        if _normalize_apostrophes(name).lower() in title:
            return prior
    return 0.5


def _is_followed(show_title: str, prefs: Preferences) -> bool:
    title = _normalize_apostrophes(show_title).lower()
    return any(_normalize_apostrophes(name).lower() in title for name in prefs.show_priors)


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
        result: list[Any] = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    try:
        import json_repair  # type: ignore
        repaired: list[Any] = json_repair.loads(text)
        if isinstance(repaired, list):
            return repaired
        if isinstance(repaired, dict):
            for v in repaired.values():
                if isinstance(v, list):
                    return v
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

    # Lower threshold: send any followed show or score >= 20 to Stage 2
    # so that shows whose RSS title doesn't perfectly match the prior key
    # still get LLM evaluation rather than being silently dropped.
    should_deep = (
        score >= 20
        or _is_followed(ep.show_title, prefs)
        or any(g.lower() in text for g in prefs.guest_watchlist)
    )

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


def build_daily_queue(
    ranked: list[RankedEpisode],
    max_minutes: float = 480.0,
    max_listen_fully: int = 3,
    max_read_summary: int = 5,
    max_outside: int = 3,
    max_email_only: int = 10,
) -> tuple[list[RankedEpisode], list[RankedEpisode]]:
    """Split ranked episodes into RSS queue and email-only overflow.

    Returns (rss_queue, email_only).
    Only "Listen Fully" episodes go into the RSS/Pocket Casts feed.
    "Read Summary Only" episodes are always routed to email_only so the
    listener's podcast app queue stays clean and playable.
    rss_queue honours max_listen_fully, max_outside caps, and the total
    listen-time budget (max_minutes). All remaining surfaced episodes go
    into email_only, capped at max_email_only total entries.
    """
    rss: list[RankedEpisode] = []
    email_only: list[RankedEpisode] = []

    listen_count = 0
    outside_count = 0
    minutes_used = 0.0

    for r in ranked:
        if r.classification == "Skip":
            continue

        # Read Summary Only → always email, never RSS
        if r.classification == "Read Summary Only":
            if len(email_only) < max_email_only:
                email_only.append(r)
            continue

        # Listen Fully below
        is_outside = getattr(r.episode, "is_outside_feed", False)

        if listen_count >= max_listen_fully:
            if len(email_only) < max_email_only:
                email_only.append(r)
            continue
        if is_outside and outside_count >= max_outside:
            if len(email_only) < max_email_only:
                email_only.append(r)
            continue
        if minutes_used + r.episode.duration_minutes > max_minutes and rss:
            if len(email_only) < max_email_only:
                email_only.append(r)
            continue

        rss.append(r)
        listen_count += 1
        minutes_used += r.episode.duration_minutes
        if is_outside:
            outside_count += 1

    return rss, email_only


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

SCORING PHILOSOPHY for this persona:
- Prioritise episodes with concrete strategic insight, real business cases, or named expert guests.
- Penalise heavily: generic communication/soft-skills content (e.g. "how to give feedback",
  "speak with confidence"), motivational fluff, and episodes that could apply to anyone at any
  level rather than a senior executive making product and business decisions.
- Reward: AI product strategy, eCommerce and retail industry dynamics, founder/operator
  stories with transferable lessons, product craft at scale, market structure analysis.
- A 17-minute episode scored 79 is almost certainly wrong — short soft-skills episodes should
  score below 60 for this persona unless the guest is a top-tier authority.

RUBRIC (base points):
- relevance: 0-30  (is the topic directly useful to this persona's strategic focus?)
- novelty: 0-15  (does it surface new frameworks, data, or perspectives?)
- guest_authority: 0-15  (is the guest a genuine expert or operator, not just a coach?)
- actionability: 0-15  (does it produce decisions or strategies the listener can act on?)
- evidence: 0-10  (are claims backed by data, case studies, or first-hand experience?)
- strategic_importance: 0-10  (does it cover trends or dynamics that matter at the director+ level?)
- learning_per_minute: 0-5  (signal density relative to episode length)

PENALTIES (negative):
- repetition_penalty: 0 to -15  (topic covered in recent episodes of same show)
- generic_penalty: 0 to -15  (content applies to anyone, not specifically to this persona)
- weak_evidence_penalty: 0 to -10  (opinion without data or real examples)
- confidence_penalty: 0 to -15  (scoring based on description only with no transcript)
- motivational_penalty: 0 to -10  (inspirational/feel-good without strategic substance)
- relevance_penalty: 0 to -20  (off-topic relative to AI, retail, eCommerce, product strategy)

CLASSIFICATION:
- "Listen Fully" if total >= 75  (goes into Pocket Casts RSS queue)
- "Read Summary Only" if total >= 50  (email digest only, NOT in RSS feed)
- "Skip" otherwise

KEY IDEAS — what counts as an insight vs. a topic label:
A "key idea" is NOT a restatement of the episode's topic, title, or theme. It is a specific,
non-obvious claim, framework, number, or contrarian take that a {prefs.persona.seniority}
{prefs.persona.role} — someone who already knows the basics of {prefs.persona.focus} — would
find genuinely new. Ground every key idea in something the TEXT actually says (a claim, a
number, a named example) rather than a category label for what the episode is "about".

  BAD (topic label — do not produce this style):
    "AI agents as both attackers and defenders in cybersecurity."
    "Strategic implications for AI product security and risk management."

  GOOD (specific, sourced, persona-relevant):
    "The guest argues patch-cycle security becomes obsolete once attackers automate
    exploit discovery, forcing a shift to continuous agent-vs-agent defense within 2 years —
    a budget line most CISOs haven't created yet."
    "Cites a case where an AI red-team found a zero-day in 40 minutes that a human pentest
    team missed for 6 months, used to argue AI-assisted offense now outpaces AI-assisted
    defense by default."

Each key_idea must pass this test: could this sentence be copy-pasted onto a DIFFERENT
episode about the same broad topic without becoming false? If yes, it's a topic label, not
an insight — rewrite it or drop it.

If TRANSCRIPT CONFIDENCE is low or the source text is description-only, do not fabricate
specificity that isn't in the text — return fewer key_ideas (even an empty list) rather than
disguising a topic label as an insight.

For EACH episode return an object with keys:
  rubric (dict), classification, classification_reason,
  summary (100-200 words — what the episode covers, for orientation),
  key_ideas (list of 0-3 strings — specific, sourced, persona-relevant insights per the
    definition above; this is NOT a compressed restatement of summary),
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
        # A short array means the response was truncated mid-JSON. Every
        # unmatched episode below silently degrades to the metadata floor
        # (50.0, no summary, no key ideas), so make the cause explicit rather
        # than emitting N separate "batch entry missing" lines that look like
        # N unrelated problems.
        if len(entries) < len(items):
            log.warning(
                "Stage 2 returned %d entries for %d episodes — response was "
                "TRUNCATED. %d episode(s) will fall back to metadata-only "
                "scoring. Consider lowering _BATCH_SIZE or raising "
                "token_budget_per_episode.",
                len(entries), len(items), len(items) - len(entries),
            )
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
                    summary=clean_snippet(ep.description, 300) or "Summary unavailable — no AI analysis for this episode.",
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
                    summary=clean_snippet(ep.description, 300) or "Summary unavailable — no AI analysis for this episode.",
                    transcript_source=transcript.source,
                    tokens_used=0,
                )
            )

    return results
