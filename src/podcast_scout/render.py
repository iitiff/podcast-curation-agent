"""HTML and Markdown briefing renderer using Jinja2 templates."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .brain import FalsifierHit, QuestionHit
from .ranking import RankedEpisode
from .synthesis import WeeklySynthesis


def _build_env(templates_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def render_briefing(
    templates_dir: Path,
    output_path: Path,
    queued: list[RankedEpisode],
    email_only: list[RankedEpisode],
    synthesis: WeeklySynthesis | None,
    run_date: str,
    feed_url: str = "",
    all_feed_url: str = "",
    hits: list[FalsifierHit] | None = None,
    question_hits: list[QuestionHit] | None = None,
) -> None:
    env = _build_env(templates_dir)
    tmpl = env.get_template("index.html.j2")
    html = tmpl.render(
        queued=queued,
        email_only=email_only,
        synthesis=synthesis,
        run_date=run_date,
        feed_url=feed_url,
        all_feed_url=all_feed_url,
        challenges=[h for h in (hits or []) if h.is_challenge],
        question_hits=[h for h in (question_hits or []) if h.is_notable],
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def render_markdown(
    queued: list[RankedEpisode],
    email_only: list[RankedEpisode],
    synthesis: WeeklySynthesis | None,
    run_date: str,
    hits: list[FalsifierHit] | None = None,
    question_hits: list[QuestionHit] | None = None,
) -> str:
    lines = [f"# Podcast Scout — {run_date}\n"]

    # Challenges lead the brief. Ranking optimises for relevance, so the queue
    # is by construction full of things that agree with the reader; anything
    # that cuts against a stated belief is the most valuable item of the day
    # and must not be buried under it.
    challenges = [h for h in (hits or []) if h.is_challenge]
    if challenges:
        lines.append("## ⚡ Challenges to your active theses\n")
        for hit in challenges:
            lines.append(f"- **{hit.thesis_title}** — {hit.reasoning}")
            lines.append(f"  ↳ via _{hit.signal_title}_ ({hit.strength})")
        lines.append("")

    notable = [h for h in (question_hits or []) if h.is_notable]
    if notable:
        lines.append("## ❓ Bearing on your open questions\n")
        for qhit in notable:
            # Distinct name from the falsifier loop above: the two carry
            # different types, and reusing `hit` makes the second one
            # unverifiable.
            marker = "⚠︎" if qhit.relation == "complicates" else "→"
            lines.append(f"- {marker} **{qhit.question_title}** — {qhit.takeaway}")
            lines.append(f"  ↳ via _{qhit.signal_title}_ ({qhit.relation}, {qhit.strength})")
        lines.append("")

    if queued:
        lines.append("## 🎧 In Your Queue Today\n")
        for i, r in enumerate(queued, 1):
            ep = r.episode
            lines.append(f"### {i}. {ep.show_title}: {ep.episode_title}")
            lines.append(f"**Score:** {r.score:.0f}/100 | **{r.classification}** | {r.evidence_confidence} confidence")
            if ep.duration_seconds:
                lines.append(f"**Duration:** {ep.duration_minutes:.0f} min")
            if r.summary:
                lines.append(f"\n{r.summary}\n")
            if r.key_ideas:
                lines.append("**Key Ideas:**")
                for idea in r.key_ideas:
                    lines.append(f"- {idea}")
            if ep.episode_url:
                lines.append(f"\n[Listen →]({ep.episode_url})\n")
            lines.append("")

    if email_only:
        lines.append("## 📌 Also Good — Didn't Fit Today's Queue\n")
        for r in email_only:
            ep = r.episode
            lines.append(f"- **{ep.show_title}: {ep.episode_title}** (Score {r.score:.0f}) — {(r.summary or ep.description)[:150]}...")
            if ep.episode_url:
                lines.append(f"  [Link]({ep.episode_url})")
        lines.append("")

    if synthesis:
        lines.append("## 📊 Weekly Synthesis\n")
        if synthesis.major_themes:
            lines.append("**Major themes:**")
            for t in synthesis.major_themes:
                lines.append(f"- {t}")
        if synthesis.weak_signal:
            lines.append(f"\n**Weak signal:** {synthesis.weak_signal}")
        if synthesis.overhyped_belief:
            lines.append(f"\n**Overhyped:** {synthesis.overhyped_belief}")
        if synthesis.retailer_implications:
            lines.append(f"\n**Retailer implications:** {synthesis.retailer_implications}")
        if synthesis.product_ideas:
            lines.append("\n**Product ideas:**")
            for idea in synthesis.product_ideas:
                lines.append(f"- {idea}")

    return "\n".join(lines)
