"""Unit tests for the durable brain: schema, store, and falsifier watch."""
import json
from datetime import UTC, datetime

import pytest

from podcast_scout.brain import (
    BrainStore,
    Source,
    Thesis,
    append_to_section,
    check_falsifiers,
    join_frontmatter,
    slugify,
    split_frontmatter,
    write_to_brain,
)
from podcast_scout.brain.falsifier import SignalInput
from podcast_scout.normalize import NormalizedEpisode
from podcast_scout.ranking import RankedEpisode, RubricScore

BODY = """## Reasoning

Seed reasoning.

## Evidence

- 2026-09-01 — [Older](../sources/a.md)

## Counter-evidence

## Implications

Downstream effects.
"""


# ---------------------------------------------------------------------------
# frontmatter + section editing
# ---------------------------------------------------------------------------

def test_frontmatter_roundtrip():
    meta = {"type": "Thesis", "id": "t-1", "title": "T"}
    text = join_frontmatter(meta, "## Reasoning\n\nBody.\n")
    parsed_meta, body = split_frontmatter(text)
    assert parsed_meta == meta
    assert body.strip().startswith("## Reasoning")


def test_page_without_frontmatter_is_preserved():
    """A hand-written note must come back whole, not be coerced into a page."""
    raw = "# Just a note\n\nNo frontmatter here.\n"
    meta, body = split_frontmatter(raw)
    assert meta == {}
    assert body == raw


def test_append_lands_inside_the_named_section():
    out = append_to_section(BODY, "Evidence", "- 2026-09-11 — [New](../sources/b.md)")
    evidence = out.split("## Evidence")[1].split("##")[0]
    assert "[New]" in evidence
    # The critical failure mode: drifting past the heading into the next section.
    assert "[New]" not in out.split("## Counter-evidence")[1]


def test_append_to_empty_section_keeps_blank_line_after_heading():
    out = append_to_section(BODY, "Counter-evidence", "- contra")
    assert "## Counter-evidence\n\n- contra" in out


def test_append_creates_a_missing_section():
    out = append_to_section("## Reasoning\n\nX\n", "Evidence", "- new")
    assert "## Evidence" in out and "- new" in out


def test_slugify_strips_accents_and_punctuation():
    assert slugify("Personalization: a Decision System!") == "personalization-a-decision-system"
    assert slugify("") == "untitled"


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

def _seed_thesis(store: BrainStore, thesis_id: str = "t-decision-system") -> Thesis:
    thesis = Thesis(
        id=thesis_id,
        title="Personalization is a decision system",
        statement="Personalization is becoming a decision system.",
        falsifier="Segment-based systems matching decision layers at scale.",
        confidence="Medium",
        body=BODY,
    )
    store.save(thesis)
    return thesis


def test_load_theses_filters_retired(tmp_path):
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    _seed_thesis(store)
    store.save(Thesis(id="t-old", title="Retired", status="retired", body=BODY))

    assert {t.id for t in store.load_theses(active_only=True)} == {"t-decision-system"}
    assert len(store.load_theses(active_only=False)) == 2


def test_append_evidence_routes_supports_and_challenges(tmp_path):
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    _seed_thesis(store)

    assert store.append_evidence("t-decision-system", "- supporting line", supports=True)
    assert store.append_evidence("t-decision-system", "- contradicting line", supports=False)

    text = store.path_for("Thesis", "t-decision-system").read_text()
    evidence, counter = text.split("## Evidence")[1].split("## Counter-evidence")
    assert "supporting line" in evidence
    assert "contradicting line" in counter
    assert "contradicting line" not in evidence


def test_append_evidence_is_idempotent(tmp_path):
    """Re-running over the same signal must not inflate the evidence trail."""
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    _seed_thesis(store)

    assert store.append_evidence("t-decision-system", "- same line") is True
    assert store.append_evidence("t-decision-system", "- same line") is False
    assert store.path_for("Thesis", "t-decision-system").read_text().count("- same line") == 1


def test_append_evidence_to_unknown_thesis_returns_false(tmp_path):
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    assert store.append_evidence("does-not-exist", "- line") is False


def test_write_source_does_not_overwrite_existing(tmp_path):
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    source = Source(id="s-1", title="A", summary="first")
    assert store.write_source(source) is not None

    again = Source(id="s-1", title="A", summary="rewritten")
    assert store.write_source(again) is None, "sources are immutable evidence"
    assert "first" in store.path_for("Source", "s-1").read_text()


def test_build_index_lists_typed_pages_only(tmp_path):
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    _seed_thesis(store)
    store.write_source(Source(id="s-1", title="A"))
    (tmp_path / "theses" / "freeform.md").write_text("# no frontmatter\n")

    index = store.build_index()
    ids = {p["id"] for p in index["pages"]}
    assert ids == {"t-decision-system", "s-1"}
    assert json.loads((tmp_path / "index.json").read_text())["count"] == 2


def test_ensure_dirs_writes_gitkeep(tmp_path):
    """The brain lives in git; empty directories would not survive a clone."""
    BrainStore(tmp_path).ensure_dirs()
    assert (tmp_path / "theses" / ".gitkeep").exists()
    assert (tmp_path / "sources" / ".gitkeep").exists()


# ---------------------------------------------------------------------------
# falsifier watch
# ---------------------------------------------------------------------------

class _StubLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def complete(self, messages, max_tokens=None, **kwargs):
        self.calls += 1

        class _Resp:
            content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)

        return _Resp()


def _signal(sid="sig-1"):
    return SignalInput(id=sid, title="Rollouts underperforming", summary="Data.")


@pytest.mark.asyncio
async def test_falsifier_hit_is_returned(tmp_path):
    theses = [Thesis(id="t-1", title="T", statement="S", falsifier="F")]
    llm = _StubLLM([
        {"thesis_id": "t-1", "signal_id": "sig-1", "direction": "challenges",
         "strength": "strong", "reasoning": "Cuts against it."}
    ])
    hits = await check_falsifiers(theses, [_signal()], llm)
    assert len(hits) == 1
    assert hits[0].is_challenge and hits[0].strength == "strong"


@pytest.mark.asyncio
async def test_hallucinated_ids_are_dropped():
    """A model-invented id must never become a dangling evidence line."""
    theses = [Thesis(id="t-1", title="T", statement="S", falsifier="F")]
    llm = _StubLLM([
        {"thesis_id": "NOPE", "signal_id": "sig-1", "direction": "supports"},
        {"thesis_id": "t-1", "signal_id": "NOPE", "direction": "supports"},
        {"thesis_id": "t-1", "signal_id": "sig-1", "direction": "sideways"},
    ])
    assert await check_falsifiers(theses, [_signal()], llm) == []


@pytest.mark.asyncio
async def test_challenges_sort_above_support():
    theses = [Thesis(id="t-1", title="T", falsifier="F"), Thesis(id="t-2", title="U", falsifier="F")]
    llm = _StubLLM([
        {"thesis_id": "t-1", "signal_id": "sig-1", "direction": "supports", "strength": "strong"},
        {"thesis_id": "t-2", "signal_id": "sig-1", "direction": "challenges", "strength": "weak"},
    ])
    hits = await check_falsifiers(theses, [_signal()], llm)
    assert [h.direction for h in hits] == ["challenges", "supports"]


@pytest.mark.asyncio
async def test_llm_failure_degrades_quietly():
    class _Boom:
        async def complete(self, *a, **kw):
            raise RuntimeError("provider down")

    theses = [Thesis(id="t-1", title="T", falsifier="F")]
    assert await check_falsifiers(theses, [_signal()], _Boom()) == []


@pytest.mark.asyncio
async def test_no_llm_or_no_theses_skips_the_call():
    assert await check_falsifiers([], [_signal()], _StubLLM([])) == []
    assert await check_falsifiers([Thesis(id="t", title="T")], [], _StubLLM([])) == []
    assert await check_falsifiers([Thesis(id="t", title="T")], [_signal()], None) == []


@pytest.mark.asyncio
async def test_unparseable_output_is_not_fatal():
    theses = [Thesis(id="t-1", title="T", falsifier="F")]
    assert await check_falsifiers(theses, [_signal()], _StubLLM("not json at all")) == []


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

def _ranked(classification="Listen Fully"):
    ep = NormalizedEpisode(
        guid="g1",
        source_feed_url="https://f.example/rss",
        show_title="No Priors",
        episode_title="Agentic checkout",
        description="Rollout data.",
        published=datetime(2026, 9, 11, tzinfo=UTC),
        duration_seconds=2400,
        episode_url="https://ex.com/1",
        category="ai_retail",
    )
    return RankedEpisode(
        episode=ep, score=82.0, rubric=RubricScore(), classification=classification,
        summary="Decision-layer personalization underperformed.",
        key_ideas=["Static segmentation held."],
    )


@pytest.mark.asyncio
async def test_write_to_brain_links_evidence_to_thesis(tmp_path):
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    _seed_thesis(store)
    source_id = "2026-09-11-no-priors-agentic-checkout"
    llm = _StubLLM([
        {"thesis_id": "t-decision-system", "signal_id": source_id,
         "direction": "challenges", "strength": "strong", "reasoning": "Contradicts."}
    ])

    result = await write_to_brain([_ranked()], tmp_path, llm)

    assert result.sources_written == [source_id]
    assert result.evidence_appended == 1
    assert len(result.challenges) == 1
    assert "Contradicts." in store.path_for("Thesis", "t-decision-system").read_text()


@pytest.mark.asyncio
async def test_weak_hits_surface_but_are_not_written(tmp_path):
    """Weak evidence belongs in the brief, not in the thesis's evidence trail."""
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    _seed_thesis(store)
    llm = _StubLLM([
        {"thesis_id": "t-decision-system", "signal_id": "2026-09-11-no-priors-agentic-checkout",
         "direction": "challenges", "strength": "weak", "reasoning": "Tangential."}
    ])

    result = await write_to_brain([_ranked()], tmp_path, llm)

    assert len(result.hits) == 1
    assert result.evidence_appended == 0
    assert "Tangential." not in store.path_for("Thesis", "t-decision-system").read_text()


@pytest.mark.asyncio
async def test_skipped_episodes_never_enter_the_brain(tmp_path):
    result = await write_to_brain([_ranked(classification="Skip")], tmp_path, _StubLLM([]))
    assert result.sources_written == []
    assert not (tmp_path / "sources").exists()


@pytest.mark.asyncio
async def test_no_theses_still_writes_sources(tmp_path):
    result = await write_to_brain([_ranked()], tmp_path, _StubLLM([]))
    assert len(result.sources_written) == 1
    assert result.hits == []


# ---------------------------------------------------------------------------
# Questions. A lighter primitive than Thesis: no belief to defend, no
# falsifier to author, so it carries no maintenance debt.
# ---------------------------------------------------------------------------

from podcast_scout.brain import Question  # noqa: E402
from podcast_scout.brain.falsifier import check_questions  # noqa: E402


def _question(store, qid="q-nba-next", status="open"):
    q = Question(
        id=qid, title="What comes after next-best-action?",
        question="What comes after next-best-action?",
        why="NBA is the default decisioning frame; I want to see it superseded.",
        status=status,
        body="## Why I'm tracking this\n\nX\n\n## Findings\n\n## Current answer\n",
    )
    store.save(q)
    return q


def test_open_questions_load_and_closed_ones_do_not(tmp_path):
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    _question(store, "q-open")
    _question(store, "q-answered", status="answered")
    assert {q.id for q in store.load_questions(open_only=True)} == {"q-open"}
    assert len(store.load_questions(open_only=False)) == 2


def test_findings_append_and_are_idempotent(tmp_path):
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    _question(store)
    assert store.append_finding("q-nba-next", "- a finding") is True
    assert store.append_finding("q-nba-next", "- a finding") is False
    text = store.path_for("Question", "q-nba-next").read_text()
    assert text.count("- a finding") == 1
    assert "- a finding" in text.split("## Findings")[1]


def test_finding_on_unknown_question_returns_false(tmp_path):
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    assert store.append_finding("nope", "- x") is False


@pytest.mark.asyncio
async def test_question_hit_is_returned():
    qs = [Question(id="q-1", title="Q", question="Q?")]
    llm = _StubLLM([
        {"question_id": "q-1", "signal_id": "sig-1", "relation": "complicates",
         "strength": "moderate", "takeaway": "Shows the obvious answer fails at scale."}
    ])
    hits = await check_questions(qs, [_signal()], llm)
    assert len(hits) == 1 and hits[0].relation == "complicates"


@pytest.mark.asyncio
async def test_complications_sort_first_and_are_always_notable():
    """A signal that makes a question harder is the easiest to skim past."""
    qs = [Question(id="q-1", title="Q"), Question(id="q-2", title="R")]
    llm = _StubLLM([
        {"question_id": "q-1", "signal_id": "sig-1", "relation": "answers", "strength": "strong"},
        {"question_id": "q-2", "signal_id": "sig-1", "relation": "complicates", "strength": "weak"},
    ])
    hits = await check_questions(qs, [_signal()], llm)
    assert [h.relation for h in hits] == ["complicates", "answers"]
    # A weak complication still earns a place in the brief.
    assert hits[0].is_notable is True


@pytest.mark.asyncio
async def test_invented_question_ids_are_dropped():
    qs = [Question(id="q-1", title="Q")]
    llm = _StubLLM([
        {"question_id": "NOPE", "signal_id": "sig-1", "relation": "answers"},
        {"question_id": "q-1", "signal_id": "sig-1", "relation": "sideways"},
    ])
    assert await check_questions(qs, [_signal()], llm) == []


@pytest.mark.asyncio
async def test_questions_work_without_any_thesis(tmp_path):
    """The two mechanisms are independent; either alone must function."""
    store = BrainStore(tmp_path)
    store.ensure_dirs()
    _question(store)
    source_id = "2026-09-11-no-priors-agentic-checkout"
    llm = _StubLLM([
        {"question_id": "q-nba-next", "signal_id": source_id, "relation": "extends",
         "strength": "strong", "takeaway": "Reframes NBA as an arbitration problem."}
    ])
    result = await write_to_brain([_ranked()], tmp_path, llm)
    assert result.findings_appended == 1
    assert result.hits == [], "no theses configured"
    assert "arbitration" in store.path_for("Question", "q-nba-next").read_text()
