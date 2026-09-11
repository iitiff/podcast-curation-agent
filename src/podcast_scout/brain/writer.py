"""Turn ranked episodes into brain pages and thesis evidence."""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from ..providers.base import BaseLLMProvider
from ..ranking import RankedEpisode
from .falsifier import (
    FalsifierHit,
    QuestionHit,
    SignalInput,
    check_falsifiers,
    check_questions,
)
from .schema import Source
from .store import BrainStore, source_id_for

log = logging.getLogger(__name__)

# Vendor and podcast material is a pattern source, not neutral evidence.
# Podcasts are promotional-adjacent by default, so credibility starts at medium
# and is only raised by corroboration, never by the pipeline itself.
DEFAULT_PODCAST_BIAS = (
    "Podcast interview: guest is usually promoting their own work. "
    "Treat claims as hypotheses pending independent corroboration."
)


class BrainWriteResult(BaseModel):
    sources_written: list[str] = Field(default_factory=list)
    hits: list[FalsifierHit] = Field(default_factory=list)
    question_hits: list[QuestionHit] = Field(default_factory=list)
    evidence_appended: int = 0
    findings_appended: int = 0

    @property
    def challenges(self) -> list[FalsifierHit]:
        return [h for h in self.hits if h.is_challenge]

    @property
    def notable_questions(self) -> list[QuestionHit]:
        return [h for h in self.question_hits if h.is_notable]


def _to_source(ranked: RankedEpisode) -> Source:
    ep = ranked.episode
    published = ep.published.strftime("%Y-%m-%d")
    source_id = source_id_for(published, ep.show_title, ep.episode_title)
    body_parts = [
        "## Compiled truth\n",
        ranked.summary or ep.description[:500] or "_No summary available._",
        "",
    ]
    if ranked.key_ideas:
        body_parts.append("### Key ideas\n")
        body_parts.extend(f"- {idea}" for idea in ranked.key_ideas)
        body_parts.append("")
    if ranked.implications:
        body_parts.append(f"### Implications\n\n{ranked.implications}\n")
    body_parts.append("## Timeline\n")
    body_parts.append(f"- {published} — episode published.")

    return Source(
        id=source_id,
        title=f"{ep.show_title}: {ep.episode_title}",
        source_type="podcast",
        origin=ep.episode_url or ep.source_feed_url,
        credibility="medium",
        bias_notes=DEFAULT_PODCAST_BIAS,
        summary=ranked.summary,
        tags=[t for t in [ep.category] if t],
        visibility="personal",
        body="\n".join(body_parts),
    )


async def write_to_brain(
    ranked: list[RankedEpisode],
    brain_dir: Path,
    llm: BaseLLMProvider | None,
) -> BrainWriteResult:
    """Write admitted signals as Source pages and link them to thesis evidence.

    Never raises: a brain failure must not take down the daily brief.
    """
    result = BrainWriteResult()
    admitted = [r for r in ranked if r.classification != "Skip"]
    if not admitted:
        return result

    store = BrainStore(brain_dir)
    store.ensure_dirs()

    signals: list[SignalInput] = []
    source_paths: dict[str, Path] = {}

    for item in admitted:
        source = _to_source(item)
        try:
            written = store.write_source(source)
        except OSError as exc:
            log.warning("Could not write source %s: %s", source.id, exc)
            continue
        if written is not None:
            result.sources_written.append(source.id)
        source_paths[source.id] = store.path_for("Source", source.id)
        signals.append(
            SignalInput(
                id=source.id,
                title=source.title,
                summary=source.summary or item.episode.description,
                origin=source.origin,
            )
        )

    # Questions and theses are independent: a brain may run on either, both, or
    # neither. Neither is required for Sources to be written.
    questions = store.load_questions(open_only=True)
    if questions:
        result.question_hits = await check_questions(questions, signals, llm)
        for qhit in result.question_hits:
            if qhit.strength == "weak":
                continue
            rel = source_paths.get(qhit.signal_id)
            link = f"../sources/{rel.name}" if rel else ""
            line = (
                f"- {qhit.signal_id[:10]} — [{qhit.signal_title}]({link}) — "
                f"{qhit.takeaway} _({qhit.relation}, {qhit.strength})_"
            )
            if store.append_finding(qhit.question_id, line):
                result.findings_appended += 1

    theses = store.load_theses(active_only=True)
    if not theses:
        if not questions:
            log.info("No open questions or active theses; skipping the watch.")
        store.build_index()
        return result

    result.hits = await check_falsifiers(theses, signals, llm)

    for hit in result.hits:
        # Weak hits are surfaced in the brief but never written to a thesis:
        # the evidence trail should stay defensible, not exhaustive.
        if hit.strength == "weak":
            continue
        rel = source_paths.get(hit.signal_id)
        link = f"../sources/{rel.name}" if rel else ""
        # Source ids are built as "<YYYY-MM-DD>-<show>-<title>", so the leading
        # 10 characters are the publish date the evidence line is dated with.
        date = hit.signal_id[:10]
        line = (
            f"- {date} — [{hit.signal_title}]({link}) — {hit.reasoning} "
            f"_({hit.strength}, auto-linked)_"
        )
        if store.append_evidence(hit.thesis_id, line, supports=not hit.is_challenge):
            result.evidence_appended += 1

    store.build_index()
    return result
