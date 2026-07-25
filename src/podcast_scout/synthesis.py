"""Weekly cross-episode synthesis using LLM."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from .ranking import RankedEpisode

if TYPE_CHECKING:
    from .config import Preferences
    from .providers.llm import OpenAIProvider

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """You are a strategic analyst for a senior retail and eCommerce product leader.

Given the following podcast episode summaries from this week, produce a cross-episode synthesis.

Return ONLY valid JSON:
{{
  "themes": ["theme 1", "theme 2", "theme 3"],
  "agreements": "Where credible speakers agree this week.",
  "disagreements": "Where credible speakers disagree.",
  "weak_signal": "One emerging weak signal worth watching.",
  "overhyped": "One potentially overhyped industry belief.",
  "retailer_implications": "Implications for a large omnichannel retailer like Walmart.",
  "product_ideas": [
    "Product idea or experiment 1",
    "Product idea or experiment 2",
    "Product idea or experiment 3"
  ]
}}"""


async def generate_synthesis(
    ranked: list[RankedEpisode],
    prefs: "Preferences",
    llm: "OpenAIProvider",
) -> dict[str, Any]:
    surfaced = [
        ep for ep in ranked
        if ep.classification in ("Listen Fully", "Read Summary Only")
    ]
    if not surfaced:
        return {"themes": [], "agreements": "", "disagreements": "",
                "weak_signal": "", "overhyped": "", "retailer_implications": "",
                "product_ideas": []}

    episode_text = "\n\n".join(
        f"Show: {ep.episode.show_title}\n"
        f"Title: {ep.episode.episode_title}\n"
        f"Score: {ep.final_score:.0f}\n"
        f"Summary: {ep.executive_summary[:500]}"
        for ep in surfaced[:8]
    )

    try:
        raw = await llm.complete(
            messages=[
                {"role": "system", "content": SYNTHESIS_PROMPT},
                {"role": "user", "content": episode_text},
            ],
            model=llm.stage2_model,
            max_tokens=1000,
        )
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Synthesis generation failed: %s", exc)
        return {"themes": ["Synthesis unavailable this week."],
                "agreements": "", "disagreements": "",
                "weak_signal": "", "overhyped": "",
                "retailer_implications": "", "product_ideas": []}
