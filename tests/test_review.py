"""Tests for the monthly "State of My Thinking" review."""
from datetime import timedelta

from podcast_scout.brain.schema import Question, Source, utcnow
from podcast_scout.brain.store import BrainStore
from podcast_scout.review import (
    MonthlyReview,
    QuestionState,
    _findings_for,
    _recent_sources,
    build_review_prompt,
    write_review,
)


def _store(tmp_path) -> BrainStore:
    store = BrainStore(tmp_path / "brain")
    store.ensure_dirs()
    return store


def _source(store: BrainStore, days_ago: int, **kwargs) -> Source:
    date = (utcnow() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    source = Source(
        id=f"{date}-show-item-{days_ago}",
        title=kwargs.pop("title", f"Item from {days_ago}d ago"),
        **kwargs,
    )
    store.save(source)
    return source


def test_findings_are_parsed_out_of_the_question_body():
    body = (
        "## Why this matters\n\nBecause.\n\n"
        "## Findings\n\n"
        "- 2026-09-01 — [A](../sources/a.md) — first _(complicates, strong)_\n"
        "- 2026-09-02 — [B](../sources/b.md) — second _(answers, moderate)_\n\n"
        "## Notes\n\n- not a finding\n"
    )
    findings = _findings_for(body)

    assert len(findings) == 2
    assert "first" in findings[0] and "second" in findings[1]
    # The Notes bullet sits in a later section and must not be swept in.
    assert not any("not a finding" in f for f in findings)


def test_findings_empty_when_section_absent():
    assert _findings_for("## Why this matters\n\nBecause.\n") == []


def test_recent_sources_windows_on_the_id_not_the_mtime(tmp_path):
    """Every file has today's mtime in a fresh clone, so the date must come
    from the Source id, which begins with the publish date."""
    store = _store(tmp_path)
    _source(store, days_ago=3, title="In window")
    _source(store, days_ago=90, title="Out of window")

    titles = [s.title for s in _recent_sources(store, lookback_days=30)]

    assert titles == ["In window"]


def test_prompt_carries_questions_findings_and_source_credibility(tmp_path):
    store = _store(tmp_path)
    store.save(
        Question(
            id="q-next-best-action",
            title="What comes after next-best-action?",
            question="What comes after next-best-action?",
            why="It decides where the roadmap goes.",
            body="## Findings\n\n- 2026-09-01 — [X](../sources/x.md) — arbitration matters\n",
        )
    )
    _source(store, days_ago=2, title="Vendor claims 40% lift",
            source_type="vendor", credibility="low", summary="A case study.")

    prompt = build_review_prompt(store, lookback_days=30, period="2026-09")

    assert "What comes after next-best-action?" in prompt
    assert "arbitration matters" in prompt
    # Credibility must reach the model: noticing a vendor-skewed month is the
    # whole point of tagging sources by class.
    assert "[vendor/low]" in prompt
    assert "2026-09" in prompt


def test_prompt_survives_a_question_with_no_findings(tmp_path):
    store = _store(tmp_path)
    store.save(Question(id="q1", title="Goal arbitration?", question="Goal arbitration?"))

    prompt = build_review_prompt(store, lookback_days=30, period="2026-09")

    assert "(no findings yet)" in prompt
    assert "(no sources recorded)" in prompt


def test_review_renders_and_writes_into_syntheses(tmp_path):
    store = _store(tmp_path)
    review = MonthlyReview(
        period="2026-09",
        headline="Arbitration moved; personalisation did not.",
        question_states=[
            QuestionState(
                question_id="q1",
                question="What comes after next-best-action?",
                verdict="converging",
                what_changed="Two papers formalised objective arbitration.",
                next_probe="Look for a production deployment, not a benchmark.",
            )
        ],
        changed_my_mind="Arbitration is a scheduling problem.",
        source_diet="Heavy on vendor material; thin on earnings calls.",
        confidence="medium",
    )

    md = review.to_markdown()
    assert "State of My Thinking — 2026-09" in md
    assert "converging" in md
    assert "Source diet" in md

    path = write_review(store, review)
    assert path.name == "2026-09-state-of-my-thinking.md"
    assert path.parent.name == "syntheses"
    assert "Arbitration is a scheduling problem." in path.read_text()
