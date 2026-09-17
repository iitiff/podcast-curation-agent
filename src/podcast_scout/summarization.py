"""Episode summarization orchestrator: obtains best transcript then deep-ranks."""
from __future__ import annotations

import logging
import os

from .config import Preferences
from .normalize import NormalizedEpisode
from .providers.base import BaseLLMProvider, BaseTranscriptionProvider, TranscriptResult
from .ranking import (
    BUDGET_EXHAUSTED_REASON,
    STAGE1_ONLY_REASON,
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
#
# That truncation was driven by thinking tokens being billed against
# maxOutputTokens, which no longer applies on a model that omits thinkingConfig
# -- so the 3 is conservative rather than load-bearing on newer models. It is
# also the main lever on REQUEST COUNT, which is what free-tier Gemini actually
# limits (20 requests/minute): 24 episodes is 8 calls at 3 per batch, but only 3
# calls at 8. Raise it to trade truncation risk for rate-limit headroom.
_BATCH_SIZE = int(os.getenv("STAGE2_BATCH_SIZE") or 3)


def _allocate_deep_slots(
    candidates: list[NormalizedEpisode],
    limit: int,
    s1_by_guid: dict[str, Stage1Result],
    prefs: Preferences,
) -> list[NormalizedEpisode]:
    """Pick which candidates get a Stage 2 call, honouring per-lane reservations.

    `candidates` must already be sorted best-first. Any lane configured with
    `min_deep_slots` takes its best few first; everything left is filled purely
    by Stage 1 score, exactly as before.

    Reservations are applied per track (podcasts and articles are allocated
    separately), so a lane configured for 2 slots can take 2 of each. That is
    intentional: the two tracks have independent allowances precisely because
    one must not starve the other.

    A reservation is a floor, not a quota. A lane with nothing to offer on a
    given day reserves nothing, and its slots are filled globally.
    """
    if limit <= 0:
        return []
    chosen: list[NormalizedEpisode] = []
    taken: set[str] = set()

    for name, cfg in sorted(prefs.categories.items()):
        if cfg.min_deep_slots <= 0:
            continue
        lane = [
            ep for ep in candidates
            if ep.guid not in taken
            and s1_by_guid[ep.guid].predicted_category == name
        ]
        for ep in lane[:cfg.min_deep_slots]:
            if len(chosen) >= limit:
                break
            chosen.append(ep)
            taken.add(ep.guid)
        if lane:
            log.info(
                "Reserved %d/%d Stage 2 slot(s) for lane %r",
                min(len(lane), cfg.min_deep_slots), cfg.min_deep_slots, name,
            )

    for ep in candidates:
        if len(chosen) >= limit:
            break
        if ep.guid in taken:
            continue
        chosen.append(ep)
        taken.add(ep.guid)

    return chosen


async def process_episodes(
    episodes: list[NormalizedEpisode],
    prefs: Preferences,
    llm: BaseLLMProvider,
    transcription: BaseTranscriptionProvider,
    max_deep_process: int = 15,
    max_deep_articles: int = 10,
    # Raised 3000 -> 5000. Each episode's object carries a 13-field rubric, a
    # 100-200 word summary, 3 key ideas, implications and several prose fields;
    # 3000 left no headroom once Gemini's overhead was included.
    token_budget_per_episode: int = 5000,
    total_token_budget: int = 400_000,
    persona_emphasis: str = "",
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

    # Sort by Stage 1 score, take top candidates for deep processing.
    #
    # Podcasts and articles draw from separate allowances. Stage 2 is the
    # scarce resource (free-tier Gemini is capped per day, not per token), and
    # a radar run returning 18 papers and trade-press items would otherwise
    # outrank and starve the podcast feed the tool exists to produce.
    s1_results.sort(key=lambda x: x[1].score, reverse=True)
    eligible = [ep for ep, s1 in s1_results if s1.should_deep_process]
    podcasts = [ep for ep in eligible if ep.source_type == "podcast"]
    articles = [ep for ep in eligible if ep.source_type != "podcast"]
    # Re-sorted into one list so batching still groups the highest scorers
    # together; the caps above are what keeps the two tracks independent.
    s1_by_guid = {s1.guid: s1 for _, s1 in s1_results}
    chosen = {
        ep.guid
        for ep in (
            _allocate_deep_slots(podcasts, max_deep_process, s1_by_guid, prefs)
            + _allocate_deep_slots(articles, max_deep_articles, s1_by_guid, prefs)
        )
    }
    deep_candidates = [ep for ep in eligible if ep.guid in chosen]

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
        batch_results = await stage2_batch_rank(
            items, prefs, llm,
            token_budget=batch_token_budget,
            persona_emphasis=persona_emphasis,
        )

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
                classification_reason=(
                    STAGE1_ONLY_REASON if ep.guid not in deep_guids
                    else BUDGET_EXHAUSTED_REASON
                ),
                evidence_confidence="low",
                summary=ep.description[:300] or "No summary available.",
            ))

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked
