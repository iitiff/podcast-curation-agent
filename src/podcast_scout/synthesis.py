"""Weekly cross-episode synthesis (runs on Fridays)."""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from .config import Preferences
from .providers.base import BaseLLMProvider, LLMMessage
from .ranking import RankedEpisode

log = logging.getLogger(__name__)


class WeeklySynthesis(BaseModel):
    major_themes: list[str] = Field(default_factory=list)
    areas_of_agreement: str = ""
    areas_of_disagreement: str = ""
    weak_signal: str = ""
    overhyped_belief: str = ""
    retailer_implications: str = ""
    product_ideas: list[str] = Field(default_factory=list)
    synthesis_note: str = ""


async def generate_synthesis(
    episodes: list[RankedEpisode],
    prefs: Preferences,
    llm: BaseLLMProvider,
) -> WeeklySynthesis:
    if not episodes:
        return WeeklySynthesis(synthesis_note="No episodes processed this week.")

    summaries = []
    for r in episodes[:10]:  # cap context
        summaries.append(
            f"SHOW: {r.episode.show_title}\n"
            f"EPISODE: {r.episode.episode_title}\n"
            f"SCORE: {r.score:.0f}\n"
            f"SUMMARY: {r.summary[:400]}\n"
        )

    context = "\n---\n".join(summaries)
    system = f"""You are a strategic analyst for a {prefs.persona.seniority} {prefs.persona.role}.
Focus: {prefs.persona.focus}.
Analyse the week's top podcast episodes and produce a cross-episode synthesis.
Return valid JSON only."""

    user = f"""EPISODES THIS WEEK:
{context}

Return JSON with keys:
- major_themes: list of 3 strings
- areas_of_agreement: string
- areas_of_disagreement: string  
- weak_signal: string (one emerging signal, do not force if evidence is weak)
- overhyped_belief: string
- retailer_implications: string (implications for a large omnichannel retailer)
- product_ideas: list of 3-5 strings
- synthesis_note: string (any caveats about thin evidence)"""

    try:
        resp = await llm.complete(
            messages=[
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            max_tokens=1500,
        )
        data = json.loads(resp.content)
        return WeeklySynthesis(**{k: v for k, v in data.items() if k in WeeklySynthesis.model_fields})
    except Exception as exc:
        log.warning("Synthesis generation failed: %s", exc)
        return WeeklySynthesis(synthesis_note=f"Synthesis unavailable: {exc}")
