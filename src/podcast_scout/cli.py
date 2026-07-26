"""CLI entry point and main pipeline orchestrator."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from .config import Settings, load_discovery, load_preferences, load_show_config
from .discovery import discover_episodes
from .email_digest import SMTPConfig, build_email_html, send_digest
from .normalize import dedup_episodes
from .providers.llm import GeminiProvider
from .providers.podcast_search import ITunesSearchProvider, PodcastIndexProvider
from .providers.transcription import CascadeTranscriptionProvider
from .providers.web_search import BraveSearchProvider, NullWebSearchProvider, SerperSearchProvider
from .ranking import RankedEpisode, RubricScore, _classify, build_daily_queue, stage1_metadata_score
from .render import render_briefing, render_markdown
from .rss import build_category_feed, build_feed
from .state import EpisodeRecord, StateManager
from .summarization import process_episodes
from .synthesis import generate_synthesis

console = Console()
log = logging.getLogger(__name__)

# Default category if a show has no category set
_DEFAULT_CATEGORY = "ai_retail"

# Top-level keys recognised in preferences.yaml — anything else triggers a warning.
_KNOWN_PREFS_KEYS = frozenset({
    "feed", "categories", "persona", "geography", "length",
    "classification", "output_caps", "show_priors",
    "topic_exclusions", "guest_watchlist", "competitor_watchlist",
})


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _make_podcast_search(settings: Settings):
    if settings.podcast_index_key and settings.podcast_index_secret:
        return PodcastIndexProvider(settings.podcast_index_key, settings.podcast_index_secret)
    return ITunesSearchProvider()


def _make_web_search(settings: Settings):
    if settings.web_search_api_key:
        if settings.web_search_provider == "serper":
            return SerperSearchProvider(settings.web_search_api_key)
        return BraveSearchProvider(settings.web_search_api_key)
    return NullWebSearchProvider()


def _smtp_from_env() -> SMTPConfig | None:
    host = os.getenv("SMTP_HOST", "")
    if not host:
        return None
    return SMTPConfig(
        host=host,
        port=int(os.getenv("SMTP_PORT", "587")),
        user=os.getenv("SMTP_USER", ""),
        password=os.getenv("SMTP_PASSWORD", ""),
        to=os.getenv("SMTP_TO", os.getenv("SMTP_USER", "")),
        from_addr=os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "")),
        use_tls=os.getenv("SMTP_USE_TLS", "true").lower() != "false",
    )


def _build_category_map(config_dir) -> dict[str, str]:
    """Build a mapping of show display_name -> category from shows.yaml."""
    shows = load_show_config(config_dir)
    mapping: dict[str, str] = {}
    for show in shows.shows:
        name = show.display_name or show.match
        cat = show.category or _DEFAULT_CATEGORY
        mapping[name.lower()] = cat
        # Also store by match string for fuzzy lookup
        mapping[show.match.lower()] = cat
    return mapping


def _resolve_category(show_title: str, category_map: dict[str, str]) -> str:
    """Return category for a show title via case-insensitive partial match."""
    title_lower = show_title.lower()
    # Exact match first
    if title_lower in category_map:
        return category_map[title_lower]
    # Partial match
    for key, cat in category_map.items():
        if key and (key in title_lower or title_lower in key):
            return cat
    return _DEFAULT_CATEGORY


def _tag_episodes_with_category(
    episodes: list[RankedEpisode],
    category_map: dict[str, str],
) -> None:
    """Stamp each episode.category in-place based on its show title."""
    for r in episodes:
        cat = _resolve_category(r.episode.show_title, category_map)
        r.episode.category = cat  # type: ignore[attr-defined]


async def _run_pipeline(
    settings: Settings,
    run_synthesis: bool = False,
    dry_run: bool = False,
) -> dict:
    prefs = load_preferences(settings.config_dir)
    discovery_cfg = load_discovery(settings.config_dir)
    category_map = _build_category_map(settings.config_dir)

    if settings.pages_base_url:
        prefs.feed.base_url = settings.pages_base_url

    state = StateManager(settings.data_dir)
    run_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # 1. Discover
    podcast_search = _make_podcast_search(settings)
    web_search = _make_web_search(settings)

    console.print(f"[cyan]Discovering episodes (lookback {settings.lookback_days} days)...[/cyan]")
    all_candidates = await discover_episodes(
        prefs=prefs,
        cfg=discovery_cfg,
        podcast_search=podcast_search,
        web_search=web_search,
        lookback_days=settings.lookback_days,
    )

    # 2. Dedup
    new_episodes, _ = dedup_episodes(all_candidates, state.seen_guids())
    console.print(f"Found {len(new_episodes)} new candidates (after dedup from {len(all_candidates)} discovered)")

    if not new_episodes:
        console.print("[yellow]No new episodes found. Exiting.[/yellow]")
        return {"queued": 0, "email_only": 0, "errors": {}}

    # 3. Categorise candidates before ranking so categories never compete.
    active_categories = list(prefs.categories.keys()) if prefs.categories else [_DEFAULT_CATEGORY]
    episodes_by_category: dict[str, list] = {category: [] for category in active_categories}
    for episode in new_episodes:
        category = _resolve_category(episode.show_title, category_map)
        episode.category = category
        episodes_by_category.setdefault(category, []).append(episode)

    # 4. LLM
    llm = None
    if settings.gemini_api_key:
        llm = GeminiProvider(settings.gemini_api_key, settings.gemini_stage2_model)
    else:
        console.print("[yellow]WARNING: No GEMINI_API_KEY — running metadata-only ranking.[/yellow]")

    # 5. Rank and select independently within each category.
    transcription = CascadeTranscriptionProvider(
        openai_api_key=None,
        enable_whisper=settings.enable_audio_transcription,
    )
    non_empty_categories = [category for category in active_categories if episodes_by_category.get(category)]
    token_budget_per_category = settings.max_llm_tokens_per_run // max(1, len(non_empty_categories))
    ranked: list[RankedEpisode] = []
    rss_queue: list[RankedEpisode] = []
    email_only: list[RankedEpisode] = []

    for category in active_categories:
        candidates = episodes_by_category.get(category, [])
        if not candidates:
            continue
        if llm:
            category_ranked = await process_episodes(
                candidates, prefs=prefs, llm=llm, transcription=transcription,
                max_deep_process=15, total_token_budget=token_budget_per_category,
            )
        else:
            category_ranked = [
                RankedEpisode(episode=episode, score=s1.score, rubric=RubricScore(),
                    classification=_classify(s1.score, prefs),
                    classification_reason="metadata only (no LLM key)",
                    evidence_confidence="low", summary=episode.description[:300] or "No summary.")
                for episode in candidates
                for s1 in [stage1_metadata_score(episode, prefs)]
            ]
            category_ranked.sort(key=lambda item: item.score, reverse=True)

        category_cfg = prefs.categories.get(category)
        category_rss, category_email = build_daily_queue(
            category_ranked,
            max_minutes=prefs.length.max_weekly_listen_hours * 60,
            max_listen_fully=category_cfg.max_listen_fully if category_cfg else prefs.output_caps.max_listen_fully,
            max_read_summary=category_cfg.max_read_summary if category_cfg else prefs.output_caps.max_read_summary,
            max_outside=prefs.output_caps.max_outside_feed,
        )
        ranked.extend(category_ranked)
        rss_queue.extend(category_rss)
        email_only.extend(category_email)

    ranked.sort(key=lambda item: item.score, reverse=True)
    rss_queue.sort(key=lambda item: item.score, reverse=True)
    email_only.sort(key=lambda item: item.score, reverse=True)
    all_surfaced = rss_queue + email_only

    console.print(f"Queue: {len(rss_queue)} episodes across {len(active_categories)} categories | Email-only: {len(email_only)}")

    # 7. Weekly synthesis
    synthesis = None
    if run_synthesis and llm:
        synthesis = await generate_synthesis(ranked, prefs, llm)

    if dry_run:
        console.print("[yellow]Dry run — skipping writes and email.[/yellow]")
        _print_summary_table(rss_queue, email_only)
        return {"queued": len(rss_queue), "email_only": len(email_only), "errors": {}}

    # 8. Write outputs
    settings.public_dir.mkdir(parents=True, exist_ok=True)
    base_url = prefs.feed.base_url or settings.pages_base_url

    # Per-category feeds
    for cat_key in active_categories:
        cat_cfg = prefs.categories.get(cat_key)
        slug = cat_cfg.slug if cat_cfg else cat_key.replace("_", "-")
        xml = build_category_feed(
            episodes=all_surfaced,
            category=cat_key,
            prefs=prefs,
            base_url=base_url,
            state=state,
        )
        (settings.public_dir / f"{slug}.xml").write_text(xml, encoding="utf-8")
        console.print(f"  [green]Wrote public/{slug}.xml[/green]")

    # Also write legacy combined feeds for backward compatibility
    listen_xml = build_feed(rss_queue, prefs, "listen", base_url, state)
    all_xml = build_feed(all_surfaced, prefs, "all", base_url, state)
    (settings.public_dir / "listen.xml").write_text(listen_xml, encoding="utf-8")
    (settings.public_dir / "all.xml").write_text(all_xml, encoding="utf-8")

    # data/latest.json
    data_dir = settings.public_dir / "data"
    data_dir.mkdir(exist_ok=True)
    latest_json = {
        "run_date": run_date,
        "queued": [_ep_to_dict(r) for r in rss_queue],
        "email_only": [_ep_to_dict(r) for r in email_only],
        "synthesis": synthesis.model_dump() if synthesis else None,
    }
    (data_dir / "latest.json").write_text(json.dumps(latest_json, indent=2, default=str))

    if settings.templates_dir.exists():
        render_briefing(
            templates_dir=settings.templates_dir,
            output_path=settings.public_dir / "index.html",
            queued=rss_queue,
            email_only=email_only,
            synthesis=synthesis,
            run_date=run_date,
            feed_url=f"{base_url}/listen.xml" if base_url else "",
            all_feed_url=f"{base_url}/all.xml" if base_url else "",
        )

    md = render_markdown(rss_queue, email_only, synthesis, run_date)
    (settings.public_dir / "latest.md").write_text(md, encoding="utf-8")

    # 9. Update state
    from .normalize import utcnow
    for r in ranked:
        state.mark_processed(EpisodeRecord(
            guid=r.episode.guid,
            show_title=r.episode.show_title,
            episode_title=r.episode.episode_title,
            published=r.episode.published,
            processed_at=utcnow(),
            score=r.score,
            classification=r.classification,
            is_outside_feed=False,
            source_feed_url=r.episode.source_feed_url,
        ))
    state.prune_old()
    state.update_last_run()
    state.snapshot_history(run_date, latest_json)
    state.save()

    # 10. Email digest
    smtp = _smtp_from_env()
    if smtp:
        feed_url = f"{base_url}/listen.xml" if base_url else ""
        html_body = build_email_html(rss_queue, email_only, run_date, feed_url)
        subject = f"Your Podcast Scout — {run_date} ({len(rss_queue)} queued)"
        try:
            send_digest(smtp, subject, html_body)
        except Exception as exc:
            console.print(f"[red]Email failed: {exc}[/red]")
    else:
        console.print("[yellow]SMTP not configured — skipping email.[/yellow]")

    _print_summary_table(rss_queue, email_only)
    return latest_json


def _ep_to_dict(r: RankedEpisode) -> dict:
    ep = r.episode
    return {
        "guid": ep.guid,
        "show": ep.show_title,
        "title": ep.episode_title,
        "score": round(r.score, 1),
        "classification": r.classification,
        "duration_min": round(ep.duration_minutes, 0),
        "url": ep.episode_url,
        "summary": r.summary,
        "key_ideas": r.key_ideas,
        "confidence": r.evidence_confidence,
        "category": getattr(ep, "category", _DEFAULT_CATEGORY),
    }


def _print_summary_table(queued: list[RankedEpisode], email_only: list[RankedEpisode]) -> None:
    table = Table(title="Daily Queue", show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Cat", width=10)
    table.add_column("Show")
    table.add_column("Episode", max_width=45)
    table.add_column("Score", justify="right")
    table.add_column("Class")
    table.add_column("Min", justify="right")
    for i, r in enumerate(queued, 1):
        cat = getattr(r.episode, "category", "?")
        table.add_row(
            str(i), cat[:10], r.episode.show_title[:28], r.episode.episode_title[:45],
            f"{r.score:.0f}", r.classification, f"{r.episode.duration_minutes:.0f}",
        )
    console.print(table)
    if email_only:
        console.print(f"[dim]+{len(email_only)} additional episodes in email only[/dim]")


@click.group()
@click.option("--verbose", "-v", is_flag=True)
def main(verbose: bool) -> None:
    """Podcast Scout — personal podcast intelligence agent."""
    _setup_logging(verbose)


@main.command()
@click.option("--synthesis", is_flag=True, help="Run weekly cross-episode synthesis")
@click.option("--dry-run", is_flag=True, help="Rank and print without writing files or sending email")
@click.option("--lookback", default=None, type=int, help="Override lookback window in days")
def run(synthesis: bool, dry_run: bool, lookback: int | None) -> None:
    """Run the full daily pipeline."""
    settings = Settings()
    if lookback:
        settings.lookback_days = lookback
    result = asyncio.run(_run_pipeline(settings, run_synthesis=synthesis, dry_run=dry_run))
    queued = result.get("queued", 0) if isinstance(result, dict) else len(result)
    console.print(f"[bold green]Done. {queued} episodes queued.[/bold green]")


@main.command()
def validate() -> None:
    """Validate config files without running the pipeline."""
    settings = Settings()
    prefs = load_preferences(settings.config_dir)
    discovery_cfg = load_discovery(settings.config_dir)
    from .discovery import _build_queries
    queries = _build_queries(prefs, discovery_cfg)
    cats = list(prefs.categories.keys())

    # Warn on any unrecognised top-level keys in preferences.yaml
    prefs_path = settings.config_dir / "preferences.yaml"
    raw_keys = set(yaml.safe_load(prefs_path.read_text()) or {})
    unknown_keys = raw_keys - _KNOWN_PREFS_KEYS
    if unknown_keys:
        for key in sorted(unknown_keys):
            console.print(f"[yellow]WARNING: preferences.yaml contains unknown key '{key}' — it will be ignored.[/yellow]")

    console.print(
        f"[green]preferences.yaml OK — "
        f"{len(prefs.show_priors)} show priors | "
        f"persona: {prefs.persona.focus[:60]}[/green]"
    )
    console.print(f"[green]Categories: {', '.join(cats) or 'none (using legacy caps)'}[/green]")
    console.print(f"[green]discovery.yaml OK — {len(queries)} queries will run[/green]")
    for q in queries:
        console.print(f"  [dim]· {q}[/dim]")
