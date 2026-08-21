"""Episode summarization orchestrator: obtains best transcript then deep-ranks."""
from __future__ import annotations

import logging

from .config import Preferences
from .normalize import NormalizedEpisode
from .providers.base import BaseLLMProvider, BaseTranscriptionProvider, TranscriptResult
from .ranking import (
    RankedEpisode,
    RubricScore,
    Stage1Result,
    _classify,
    stage1_metadata_score,
    stage2_batch_rank,
)

log = logging.getLogger(__name__)

# Maximum episodes per batch LLM call.
#
# Lowered 5 -> 3 after a production run where 11 of 14 episodes fell back to
# metadata-only scoring: batches of 5 produced JSON arrays that exceeded the
# output budget and came back truncated (or empty), so every unmatched episode
# lost its summary and key ideas. A smaller batch means a shorter response, and
# a truncation costs 3 episodes instead of 5.
_BATCH_SIZE = 3


async def process_episodes(
    episodes: list[NormalizedEpisode],
    prefs: Preferences,
    llm: BaseLLMProvider,
    transcription: BaseTranscriptionProvider,
    max_deep_process: int = 15,
    # Raised 3000 -> 5000. Each episode's object carries a 13-field rubric, a
    # 100-200 word summary, 3 key ideas, implications and several prose fields;
    # 3000 left no headroom once Gemini's overhead was included.
    token_budget_per_episode: int = 5000,
    total_token_budget: int = 400_000,
) -> list[RankedEpisode]:
    """Stage 1 filter then Stage 2 deep-rank top candidates.

    Stage 2 calls are batched (_BATCH_SIZE episodes per LLM request) to
    minimise API quota consumption on the free Gemini tier.
    """
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

    deep_guids = {ep.guid for ep in deep_candidates}
    ranked: list[RankedEpisode] = []
    tokens_used = 0

    # Fetch transcripts for all deep candidates first (these are cheap/free)
    transcript_map: dict[str, TranscriptResult] = {}
    for ep in deep_candidates:
        transcript = await transcription.transcribe(
            episode_url=ep.episode_url,
            description=ep.description,
        )
        transcript_map[ep.guid] = transcript

    # Batch Stage 2 LLM calls: process _BATCH_SIZE episodes per API call
    for batch_start in range(0, len(deep_candidates), _BATCH_SIZE):
        if tokens_used >= total_token_budget:
            log.warning("Token budget exhausted at batch %d", batch_start // _BATCH_SIZE)
            # Remaining deep candidates fall through to metadata fallback below
            break

        batch = deep_candidates[batch_start: batch_start + _BATCH_SIZE]
        items = [(ep, transcript_map[ep.guid]) for ep in batch]

        # token_budget passed is per-episode * batch size so the model has
        # enough room to write all summaries
        batch_token_budget = token_budget_per_episode * len(batch)
        batch_results = await stage2_batch_rank(items, prefs, llm, token_budget=batch_token_budget)

        for r in batch_results:
            tokens_used += r.tokens_used
        ranked.extend(batch_results)

    # Add S1-only episodes (not deep-processed, or budget exhausted)
    processed_guids = {r.episode.guid for r in ranked}
    for ep, s1 in s1_results:
        if ep.guid not in processed_guids:
            ranked.append(RankedEpisode(
                episode=ep,
                score=s1.score,
                rubric=RubricScore(),
                classification=_classify(s1.score, prefs),
                classification_reason="stage1 only" if ep.guid not in deep_guids else "token budget exhausted",
                evidence_confidence="low",
                summary=ep.description[:300] or "No summary available.",
            ))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
