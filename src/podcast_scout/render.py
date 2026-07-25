"""HTML and Markdown briefing generation using Jinja2 templates."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .normalize import utcnow
from .ranking import RankedEpisode

logger = logging.getLogger(__name__)


def _load_env(templates_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html_briefing(
    ranked: list[RankedEpisode],
    synthesis: dict[str, Any],
    run_date: str,
    diagnostics: dict[str, Any],
    prefs_dict: dict[str, Any],
    templates_dir: Path,
    public_dir: Path,
) -> None:
    env = _load_env(templates_dir)
    try:
        tmpl = env.get_template("index.html.j2")
    except Exception:
        logger.warning("Template index.html.j2 not found; skipping HTML briefing.")
        return

    listen = [e for e in ranked if e.classification == "Listen Fully"]
    summary_only = [e for e in ranked if e.classification == "Read Summary Only"]
    outside = [e for e in ranked if e.episode.is_outside_feed]
    skipped = [e for e in ranked if e.classification == "Skip"]

    html = tmpl.render(
        run_date=run_date,
        now=utcnow().isoformat(),
        listen=listen,
        summary_only=summary_only,
        outside=outside,
        skipped=skipped,
        all_ranked=ranked,
        synthesis=synthesis,
        diagnostics=diagnostics,
        feed_title=prefs_dict.get("feed", {}).get("title", "Podcast Scout"),
    )
    public_dir.mkdir(parents=True, exist_ok=True)
    (public_dir / "index.html").write_text(html, encoding="utf-8")
    logger.info("HTML briefing written.")


def render_markdown_briefing(
    ranked: list[RankedEpisode],
    synthesis: dict[str, Any],
    run_date: str,
    public_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append(f"# Podcast Scout Briefing — {run_date}\n")

    listen = [e for e in ranked if e.classification == "Listen Fully"]
    if listen:
        lines.append("## \U0001f3a7 Listen Fully\n")
        for i, ep in enumerate(listen, 1):
            lines.append(f"### {i}. {ep.episode.show_title}: {ep.episode.episode_title}")
            lines.append(f"**Score:** {ep.final_score:.0f} | **Duration:** {ep.episode.duration_minutes:.0f} min")
            lines.append(f"\n{ep.executive_summary}\n")
            if ep.episode.episode_url:
                lines.append(f"[Listen]({ep.episode.episode_url})\n")

    summary_only = [e for e in ranked if e.classification == "Read Summary Only"]
    if summary_only:
        lines.append("## \U0001f4d6 Read Summary Only\n")
        for ep in summary_only:
            lines.append(f"### {ep.episode.show_title}: {ep.episode.episode_title}")
            lines.append(f"**Score:** {ep.final_score:.0f} | **Duration:** {ep.episode.duration_minutes:.0f} min")
            lines.append(f"\n{ep.executive_summary}\n")
            if ep.episode.episode_url:
                lines.append(f"[Link]({ep.episode.episode_url})\n")

    if synthesis:
        lines.append("## \U0001f9e0 Synthesis\n")
        for theme in synthesis.get("themes", []):
            lines.append(f"- {theme}")

    public_dir.mkdir(parents=True, exist_ok=True)
    (public_dir / "latest.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown briefing written.")


def render_json_output(
    ranked: list[RankedEpisode],
    synthesis: dict[str, Any],
    run_date: str,
    public_dir: Path,
) -> None:
    data: dict[str, Any] = {
        "run_date": run_date,
        "episodes": [
            {
                "guid": ep.episode.guid,
                "show": ep.episode.show_title,
                "title": ep.episode.episode_title,
                "score": ep.final_score,
                "classification": ep.classification,
                "confidence": ep.confidence,
                "duration_minutes": round(ep.episode.duration_minutes, 1),
                "published": ep.episode.published.isoformat(),
                "episode_url": ep.episode.episode_url,
                "is_outside_feed": ep.episode.is_outside_feed,
                "executive_summary": ep.executive_summary,
                "key_ideas": ep.key_ideas,
                "implications": ep.implications,
            }
            for ep in ranked
        ],
        "synthesis": synthesis,
    }
    data_dir = public_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "latest.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8"
    )
    logger.info("JSON output written.")
