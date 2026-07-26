"""Weekly cross-episode synthesis using LLM."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from .providers.base import BaseLLMProvider, LLMMessage
from .ranking import RankedEpisode

log = logging.getLogger(__name__)


class WeeklySynthesis(BaseModel):
    major_themes: list[str] = []
    areas_of_agreement: str = ""
    areas_of_disagreement: str = ""
    weak_signal: str = ""
    overhyped_belief: str = ""
    recommended_action: str = ""
    confidence: str = "low"


def _build_synthesis_prompt(episodes: list[RankedEpisode]) -> str:
    summaries = []
    for r in episodes[:15]:
        if r.summary:
            summaries.append(f"- [{r.episode.show_title}] {r.episode.episode_title}: {r.summary[:300]}")
    episode_block = "\n".join(summaries) or "No episode summaries available."

    return f"""You are a strategic intelligence analyst reviewing this week's podcast content.

Episodes:
{episode_block}

Return a JSON object with these keys:
- major_themes: list of 3 strings
- areas_of_agreement: string
- areas_of_disagreement: string
- weak_signal: string (one emerging signal, do not force if evidence is weak)
- overhyped_belief: string
- recommended_action: string (one concrete action based on this week's insights)
- confidence: "high" | "medium" | "low"

Return ONLY raw JSON, no markdown."""


async def generate_synthesis(
    episodes: list[RankedEpisode],
    prefs: Any,
    llm: BaseLLMProvider,
) -> WeeklySynthesis | None:
    prompt = _build_synthesis_prompt(episodes)
    try:
        resp = await llm.complete(
            messages=[LLMMessage(role="user", content=prompt)],
            max_tokens=800,
        )
        import json
        import re
        text = resp.content.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        data = json.loads(text)
        return WeeklySynthesis(**data)
    except Exception as exc:
        log.warning("Synthesis generation failed: %s", exc)
        return None
