"""Two-stage episode ranking engine."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .normalize import NormalizedEpisode

if TYPE_CHECKING:
    from .config import Preferences
    from .providers.llm import OpenAIProvider

logger = logging.getLogger(__name__)


class RubricScores(BaseModel):
    relevance: float = Field(0, ge=0, le=30)
    novelty: float = Field(0, ge=0, le=15)
    guest_authority: float = Field(0, ge=0, le=15)
    actionability: float = Field(0, ge=0, le=15)
    evidence: float = Field(0, ge=0, le=10)
    strategic_importance: float = Field(0, ge=0, le=10)
    learning_value_per_minute: float = Field(0, ge=0, le=5)
    # Penalties (stored as positive numbers, subtracted)
    penalty_repetition: float = Field(0, ge=0, le=15)
    penalty_promotional: float = Field(0, ge=0, le=15)
    penalty_weak_evidence: float = Field(0, ge=0, le=10)
    penalty_low_confidence: float = Field(0, ge=0, le=15)
    penalty_motivational: float = Field(0, ge=0, le=10)
    penalty_low_relevance: float = Field(0, ge=0, le=20)
    boundary_override: float = Field(0, ge=-5, le=5)
    override_justification: str = ""

    @property
    def total(self) -> float:
        raw = (
            self.relevance
            + self.novelty
            + self.guest_authority
            + self.actionability
            + self.evidence
            + self.strategic_importance
            + self.learning_value_per_minute
            - self.penalty_repetition
            - self.penalty_promotional
            - self.penalty_weak_evidence
            - self.penalty_low_confidence
            - self.penalty_motivational
            - self.penalty_low_relevance
        )
        return max(0.0, min(100.0, raw + self.boundary_override))


class RankedEpisode(BaseModel):
    episode: NormalizedEpisode
    stage1_score: float = 0.0
    scores: RubricScores | None = None
    final_score: float = 0.0
    classification: str = "Skip"  # Listen Fully | Read Summary Only | Skip
    confidence: str = "low"  # high | medium | low
    why_ranked: str = ""
    executive_summary: str = ""
    key_ideas: list[str] = Field(default_factory=list)
    implications: str = ""
    who_should_listen: str = ""
    summary_captures: str = ""
    listen_nuance: str = ""
    skip_reason: str = ""


# ───────────────────────── Stage 1: Metadata ranking ─────────────────────────

def stage1_score(ep: NormalizedEpisode, prefs: "Preferences") -> float:
    """Fast metadata-based scoring. No LLM calls."""
    score = 0.0

    # Interest keyword matching
    text = f"{ep.show_title} {ep.episode_title} {ep.description}".lower()
    interest_hits = sum(
        weight
        for topic, weight in prefs.interests.items()
        if topic.replace("_", " ") in text or topic.replace("_", "") in text
    )
    score += min(30.0, interest_hits * 8)

    # Guest watchlist boost
    for guest in prefs.guest_watchlist:
        if guest.lower() in text:
            score += 10
            break

    # Competitor watchlist boost
    competitor_hits = sum(1 for c in prefs.competitor_watchlist if c.lower() in text)
    score += min(10.0, competitor_hits * 3)

    # Show prior
    show_prior = _lookup_show_prior(ep.show_title, prefs.show_priors)
    score += show_prior * 15

    # Topic exclusion penalty
    for excl in prefs.topic_exclusions:
        if excl.lower() in text:
            score -= 20
            break

    # Duration fit
    mins = ep.duration_minutes
    if mins > 0:
        if prefs.length.preferred_min_minutes <= mins <= prefs.length.preferred_max_minutes:
            score += 5
        elif mins > prefs.length.hard_max_minutes:
            score -= 10

    return max(0.0, min(100.0, score))


def _lookup_show_prior(show_title: str, priors: dict[str, float]) -> float:
    title_lower = show_title.lower()
    for key, val in priors.items():
        if key.lower() in title_lower or title_lower in key.lower():
            return val
    return 0.5


# ───────────────────────── Stage 2: LLM deep ranking ─────────────────────────

STAGE2_SYSTEM_PROMPT = """You are a podcast ranking assistant for a {role} focused on {focus}.
Your job is to score a podcast episode using the rubric below and return valid JSON only.

RUBRIC (score each dimension):
- relevance (0-30): How directly relevant is this to the listener's stated interests?
- novelty (0-15): Does it contain non-obvious insights or new information?
- guest_authority (0-15): Does the guest have firsthand authority and direct experience?
- actionability (0-15): Can a product or retail leader act on what they learn?
- evidence (0-10): Are claims backed by data, case studies, or concrete examples?
- strategic_importance (0-10): Is this timely and strategically important right now?
- learning_value_per_minute (0-5): High value per minute of listening time?

PENALTIES (score each as a positive number to be subtracted):
- penalty_repetition (0-15): Repeats ideas covered better elsewhere
- penalty_promotional (0-15): Generic commentary or promotional interview
- penalty_weak_evidence (0-10): Vague predictions, no supporting evidence
- penalty_low_confidence (0-15): Based only on title/short description
- penalty_motivational (0-10): Primarily motivational or personal-development content
- penalty_low_relevance (0-20): Poor relevance to current priorities

OPTIONAL:
- boundary_override (-5 to +5): Override score boundary if justified
- override_justification: Required if boundary_override != 0

Return ONLY this JSON structure, no markdown:
{{
  "relevance": 0,
  "novelty": 0,
  "guest_authority": 0,
  "actionability": 0,
  "evidence": 0,
  "strategic_importance": 0,
  "learning_value_per_minute": 0,
  "penalty_repetition": 0,
  "penalty_promotional": 0,
  "penalty_weak_evidence": 0,
  "penalty_low_confidence": 0,
  "penalty_motivational": 0,
  "penalty_low_relevance": 0,
  "boundary_override": 0,
  "override_justification": "",
  "why_ranked": "One sentence.",
  "executive_summary": "150-300 word summary.",
  "key_ideas": ["idea 1", "idea 2", "idea 3"],
  "implications": "Implications for retail/product leaders.",
  "who_should_listen": "Who benefits most from this episode.",
  "summary_captures": "What the summary captures vs listening.",
  "listen_nuance": "What nuance is lost by not listening (for Listen Fully only).",
  "skip_reason": "One sentence reason if skipped."
}}"""


async def stage2_rank(
    ep: NormalizedEpisode,
    source_text: str,
    confidence: str,
    prefs: "Preferences",
    llm: "OpenAIProvider",
) -> RankedEpisode:
    """Deep LLM-based ranking for top-candidate episodes."""
    system = STAGE2_SYSTEM_PROMPT.format(
        role=prefs.persona.role,
        focus=prefs.persona.focus,
    )
    user_msg = f"""Show: {ep.show_title}
Title: {ep.episode_title}
Published: {ep.published.date()}
Duration: {ep.duration_minutes:.0f} min
Guests: {', '.join(ep.guests) or 'Unknown'}
Evidence confidence: {confidence}

Source text (may be truncated):
{source_text[:6000]}"""

    try:
        raw = await llm.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            model=llm.stage2_model,
            max_tokens=1200,
        )
        data = json.loads(raw)
        scores = RubricScores.model_validate(data)
    except Exception as exc:
        logger.warning("Stage 2 ranking failed for '%s': %s", ep.episode_title, exc)
        scores = RubricScores(penalty_low_confidence=15)

    final = scores.total
    cfg = prefs.classification
    if final >= cfg.listen_fully_min_score:
        classification = "Listen Fully"
    elif final >= cfg.read_summary_min_score:
        classification = "Read Summary Only"
    else:
        classification = "Skip"

    return RankedEpisode(
        episode=ep,
        scores=scores,
        final_score=final,
        classification=classification,
        confidence=confidence,
        why_ranked=data.get("why_ranked", "") if isinstance(data, dict) else "",
        executive_summary=data.get("executive_summary", "") if isinstance(data, dict) else "",
        key_ideas=data.get("key_ideas", []) if isinstance(data, dict) else [],
        implications=data.get("implications", "") if isinstance(data, dict) else "",
        who_should_listen=data.get("who_should_listen", "") if isinstance(data, dict) else "",
        summary_captures=data.get("summary_captures", "") if isinstance(data, dict) else "",
        listen_nuance=data.get("listen_nuance", "") if isinstance(data, dict) else "",
        skip_reason=data.get("skip_reason", "") if isinstance(data, dict) else "",
    )
