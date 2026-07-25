"""Episode summarization orchestrator: obtains best transcript then deep-ranks."""
from __future__ import annotations

import asyncio
import logging

from .config import Preferences
from .normalize import NormalizedEpisode
from .providers.base import BaseLLMProvider, BaseTranscriptionProvider
from .ranking import RankedEpisode, Stage1Result, stage2_deep_rank, stage1_metadata_score

log = logging.getLogger(__name__)


async def process_episodes(
    episodes: list[NormalizedEpisode],
    prefs: Preferences,
    llm: BaseLLMProvider,
    transcription: BaseTranscriptionProvider,
    max_deep_process: int = 15,
    token_budget_per_episode: int = 3000,
    total_token_budget: int = 400_000,
) -> list[RankedEpisode]:
    """Stage 1 filter then Stage 2 deep-rank top candidates."""
    # Stage 1: metadata rank all
    s1_results: list[tuple[NormalizedEpisode, Stage1Result]] = []
    for ep in episodes:
        s1 = stage1_metadata_score(ep, prefs)
        s1_results.append((ep, s1))

    # Sort by Stage 1 score, take top candidates for deep processing
    s1_results.sort(key=lambda x: x[1].score, reverse=True)
    deep_candidates = [
        ep for ep, s1 in s1_results
        if s1.should_deep_process
    ][:max_deep_process]

    # For episodes not deep-processed, build a lightweight RankedEpisode from S1
    deep_guids = {ep.guid for ep in deep_candidates}
    ranked: list[RankedEpisode] = []
    tokens_used = 0

    async def process_one(ep: NormalizedEpisode) -> RankedEpisode:
        nonlocal tokens_used
        if tokens_used >= total_token_budget:
            log.warning("Token budget exhausted, falling back to metadata for %s", ep.episode_title)
            s1 = stage1_metadata_score(ep, prefs)
            from .ranking import RubricScore, _classify
            return RankedEpisode(
                episode=ep,
                score=s1.score,
                rubric=RubricScore(),
                classification=_classify(s1.score, prefs),
                classification_reason="token budget exhausted",
                evidence_confidence="low",
                summary=ep.description[:300] or "No summary available.",
            )
        transcript = await transcription.get_transcript(
            episode_url=ep.episode_url,
            audio_url=ep.enclosure.url if ep.enclosure else "",
        )
        result = await stage2_deep_rank(ep, transcript, prefs, llm, token_budget_per_episode)
        tokens_used += result.tokens_used
        return result

    deep_results = await asyncio.gather(*[process_one(ep) for ep in deep_candidates])
    ranked.extend(deep_results)

    # Add S1-only episodes that weren't deep processed
    from .ranking import RubricScore, _classify
    for ep, s1 in s1_results:
        if ep.guid not in deep_guids:
            ranked.append(RankedEpisode(
                episode=ep,
                score=s1.score,
                rubric=RubricScore(),
                classification=_classify(s1.score, prefs),
                classification_reason="stage1 only",
                evidence_confidence="low",
                summary=ep.description[:300] or "No summary available.",
            ))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
