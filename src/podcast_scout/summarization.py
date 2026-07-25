"""Executive summary generation and synthesis."""
from __future__ import annotations

import logging
from typing import Any

from .config import Preferences
from .normalize import NormalizedEpisode
from .ranking import Stage2Result

log = logging.getLogger(__name__)

SYNTHESIS_SYSTEM = """
You are a strategic analyst for a senior retail and eCommerce product leader.
You will receive a set of podcast episode summaries from this week.
Your job is to synthesise themes, agreements, disagreements, weak signals,
overhyped beliefs, and actionable implications.
Be concise, specific, and rigorous. Return JSON only.
"""


def synthesize_week(
    episodes: list[NormalizedEpisode],
    results: list[Stage2Result],
    prefs: Preferences,
    llm_client: Any,
    model: str,
) -> dict:
    """Generate weekly cross-episode synthesis."""
    if not results:
        return {"error": "No episodes to synthesize"}

    ep_map = {ep.guid: ep for ep in episodes}
    summaries = []
    for r in results[:10]:  # cap context
        ep = ep_map.get(r.guid)
        if not ep:
            continue
        summaries.append(
            f"Show: {ep.show_title}\n"
            f"Episode: {ep.episode_title}\n"
            f"Classification: {r.classification}\n"
            f"Summary: {r.executive_summary[:400]}\n"
            f"Key ideas: {'; '.join(r.key_ideas)}\n"
        )

    persona = f"{prefs.persona.role} focused on {prefs.persona.focus}"
    prompt = f"""Synthesize these {len(summaries)} podcast episodes for: {persona}

{'---'.join(summaries)}

Return JSON:
  major_themes: [{{theme: str, evidence: str}}, ...] (3 items)
  agreements: [str] (2-3 points where speakers agree)
  disagreements: [str] (credible speakers who disagree)
  weak_signal: str (one emerging trend not yet mainstream)
  overhyped_belief: str (one popular claim that may be wrong)
  retailer_implications: str (implications for a large omnichannel retailer)
  product_ideas: [str] (3-5 experiments or questions worth exploring)
"""
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as exc:
        log.warning("Synthesis failed: %s", exc)
        return {"error": str(exc)}
