"""Two-stage episode ranking engine."""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from .config import Preferences
from .normalize import NormalizedEpisode

log = logging.getLogger(__name__)

CLASSIFICATION_LISTEN = "Listen Fully"
CLASSIFICATION_SUMMARY = "Read Summary Only"
CLASSIFICATION_SKIP = "Skip"

STAGE1_SYSTEM = """
You are a podcast ranking assistant for a senior retail and eCommerce product leader.
Your job is to score episodes based ONLY on their metadata (title, description, guests, keywords).
You must be strict. Generic interviews and motivational content score low.
Return JSON only.
"""

STAGE2_SYSTEM = """
You are a podcast intelligence analyst for a senior retail/eCommerce product leader.
Analyze the episode content deeply and produce a detailed ranking and executive summary.
Be rigorous. Penalize vague predictions, generic advice, and promotional content.
Return JSON only.
"""


class RubricScore(BaseModel):
    relevance: int = Field(0, ge=0, le=30, description="Relevance to interests 0-30")
    novelty: int = Field(0, ge=0, le=15, description="Novelty/non-obvious insight 0-15")
    guest_authority: int = Field(0, ge=0, le=15, description="Guest authority 0-15")
    actionability: int = Field(0, ge=0, le=15, description="Actionability 0-15")
    evidence: int = Field(0, ge=0, le=10, description="Evidence/data/case studies 0-10")
    strategic_importance: int = Field(0, ge=0, le=10, description="Strategic importance 0-10")
    learning_value_per_min: int = Field(0, ge=0, le=5, description="Learning value per minute 0-5")

    # Penalties (stored as negative values)
    penalty_repetition: int = Field(0, ge=-15, le=0)
    penalty_generic: int = Field(0, ge=-15, le=0)
    penalty_weak_evidence: int = Field(0, ge=-10, le=0)
    penalty_low_confidence: int = Field(0, ge=-15, le=0)
    penalty_motivational: int = Field(0, ge=-10, le=0)
    penalty_poor_relevance: int = Field(0, ge=-20, le=0)

    @property
    def total(self) -> int:
        base = (
            self.relevance + self.novelty + self.guest_authority
            + self.actionability + self.evidence + self.strategic_importance
            + self.learning_value_per_min
        )
        penalties = (
            self.penalty_repetition + self.penalty_generic
            + self.penalty_weak_evidence + self.penalty_low_confidence
            + self.penalty_motivational + self.penalty_poor_relevance
        )
        return max(0, min(100, base + penalties))


class Stage1Result(BaseModel):
    guid: str
    score: int
    rubric: RubricScore
    classification: str
    reason: str
    advance_to_stage2: bool


class Stage2Result(BaseModel):
    guid: str
    score: int
    rubric: RubricScore
    classification: str
    classification_override_justification: str = ""
    evidence_confidence: str  # High | Medium | Low
    executive_summary: str
    key_ideas: list[str]
    implications: str = ""
    who_should_listen: str = ""
    summary_captures_value: str = ""
    listen_nuance: str = ""
    skip_reason: str = ""
    guests_identified: list[str] = Field(default_factory=list)


def _classify(score: int, prefs: Preferences, override_justification: str = "") -> str:
    cfg = prefs.classification
    if score >= cfg.listen_fully_min_score:
        return CLASSIFICATION_LISTEN
    elif score >= cfg.read_summary_min_score:
        return CLASSIFICATION_SUMMARY
    else:
        return CLASSIFICATION_SKIP


def stage1_rank(
    episodes: list[NormalizedEpisode],
    prefs: Preferences,
    llm_client: Any,
    model: str,
) -> list[Stage1Result]:
    """Inexpensive metadata-based first pass. Returns all episodes with scores."""
    results: list[Stage1Result] = []

    interest_str = ", ".join(
        f"{k} ({v})" for k, v in sorted(prefs.interests.items(), key=lambda x: -x[1])
    )
    persona_str = f"{prefs.persona.role} focused on {prefs.persona.focus}"
    competitors = ", ".join(prefs.competitor_watchlist[:10])
    watchlist = ", ".join(prefs.guest_watchlist[:10])
    exclusions = ", ".join(prefs.topic_exclusions)

    for ep in episodes:
        show_prior = prefs.show_priors.get(ep.show_title, 0.5)
        prompt = f"""Score this podcast episode for: {persona_str}

Interest weights: {interest_str}
Guest watchlist (boost): {watchlist}
Competitor watchlist (boost): {competitors}
Topic exclusions (penalise): {exclusions}
Show prior weight: {show_prior}

Episode metadata:
- Show: {ep.show_title}
- Title: {ep.episode_title}
- Published: {ep.published.date()}
- Duration: {ep.duration_minutes:.0f} min
- Description: {ep.description[:800]}
- Guests: {', '.join(ep.guests) or 'unknown'}
- Keywords: {', '.join(ep.keywords[:10])}

Return JSON with fields:
  guid (string, use "{ep.guid}"),
  score (int 0-100),
  rubric (object with all rubric fields),
  classification ("Listen Fully" | "Read Summary Only" | "Skip"),
  reason (string, 1-2 sentences),
  advance_to_stage2 (bool, true if score >= 50)
"""
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": STAGE1_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = json.loads(response.choices[0].message.content)
            rubric = RubricScore.model_validate(raw.get("rubric", {}))
            score = rubric.total
            result = Stage1Result(
                guid=ep.guid,
                score=score,
                rubric=rubric,
                classification=_classify(score, prefs),
                reason=raw.get("reason", ""),
                advance_to_stage2=score >= prefs.classification.read_summary_min_score,
            )
            results.append(result)
            log.info("Stage1 %s — %s: score=%d", ep.show_title, ep.episode_title[:50], score)
        except Exception as exc:
            log.warning("Stage1 failed for %s: %s", ep.guid, exc)
            results.append(
                Stage1Result(
                    guid=ep.guid,
                    score=0,
                    rubric=RubricScore(),
                    classification=CLASSIFICATION_SKIP,
                    reason=f"Ranking failed: {exc}",
                    advance_to_stage2=False,
                )
            )

    return results


def stage2_rank(
    episodes: list[NormalizedEpisode],
    stage1_results: dict[str, Stage1Result],
    prefs: Preferences,
    llm_client: Any,
    model: str,
) -> list[Stage2Result]:
    """Deep content-aware ranking for top candidates."""
    results: list[Stage2Result] = []
    persona_str = f"{prefs.persona.role} focused on {prefs.persona.focus}"
    competitors = ", ".join(prefs.competitor_watchlist[:10])

    for ep in episodes:
        s1 = stage1_results.get(ep.guid)
        confidence = "Low"
        content = ep.description[:3000]

        if ep.transcript_url:
            confidence = "High"
        elif len(ep.description) > 500:
            confidence = "Medium"

        prompt = f"""Deep-rank this podcast episode for: {persona_str}
Strategically relevant companies: {competitors}

Episode:
- Show: {ep.show_title}
- Title: {ep.episode_title}
- Published: {ep.published.date()}
- Duration: {ep.duration_minutes:.0f} min
- Guests: {', '.join(ep.guests) or 'unknown'}
- Stage 1 score: {s1.score if s1 else 'N/A'}
- Source confidence: {confidence}

Content ({confidence} confidence):
{content}

Return JSON:
  guid: "{ep.guid}"
  score: int 0-100
  rubric: {{all rubric fields}}
  classification: "Listen Fully" | "Read Summary Only" | "Skip"
  classification_override_justification: string (if overriding stage1, explain why)
  evidence_confidence: "{confidence}"
  executive_summary: string 150-300 words
  key_ideas: [string, string, string]
  implications: string (Walmart/retail/CX/product implications)
  who_should_listen: string
  summary_captures_value: string (for Read Summary Only)
  listen_nuance: string (for Listen Fully — what summary misses)
  skip_reason: string (for Skip)
  guests_identified: [string]
"""
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": STAGE2_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            raw = json.loads(response.choices[0].message.content)
            rubric = RubricScore.model_validate(raw.get("rubric", {}))
            score = rubric.total
            result = Stage2Result(
                guid=ep.guid,
                score=score,
                rubric=rubric,
                classification=_classify(score, prefs),
                classification_override_justification=raw.get("classification_override_justification", ""),
                evidence_confidence=raw.get("evidence_confidence", confidence),
                executive_summary=raw.get("executive_summary", ""),
                key_ideas=raw.get("key_ideas", []),
                implications=raw.get("implications", ""),
                who_should_listen=raw.get("who_should_listen", ""),
                summary_captures_value=raw.get("summary_captures_value", ""),
                listen_nuance=raw.get("listen_nuance", ""),
                skip_reason=raw.get("skip_reason", ""),
                guests_identified=raw.get("guests_identified", []),
            )
            results.append(result)
            log.info("Stage2 %s — %s: score=%d cls=%s", ep.show_title, ep.episode_title[:50], score, result.classification)
        except Exception as exc:
            log.warning("Stage2 failed for %s: %s", ep.guid, exc)

    return results
