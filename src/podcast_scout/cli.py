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
from rich.console import Console
from rich.table import Table

from .config import Settings, load_discovery, load_preferences, load_shows
from .discovery import discover_outside_episodes
from .email_digest import SMTPConfig, build_email_html, send_digest
from .feeds import fetch_all_feeds
from .normalize import dedup_episodes
from .opml import parse_opml
from .providers.llm import OpenAIProvider
from .providers.podcast_search import ITunesSearchProvider, NullPodcastSearchProvider, PodcastIndexProvider
from .providers.transcription import CascadeTranscriptionProvider
from .providers.web_search import BraveSearchProvider, NullWebSearchProvider, SerperSearchProvider
from .ranking import build_daily_queue
from .render import render_briefing, render_markdown
from .rss import build_feed
from .state import EpisodeRecord, StateManager
from .summarization import process_episodes
from .synthesis import generate_synthesis

console = Console()
log = logging.getLogger(__name__)


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


async def _run_pipeline(
    settings: Settings,
    run_synthesis: bool = False,
    dry_run: bool = False,
) -> dict:
    prefs = load_preferences(settings.config_dir)
    shows_cfg = load_shows(settings.config_dir)
    discovery_cfg = load_discovery(settings.config_dir)

    # Override feed base_url from settings/env
    if settings.pages_base_url:
        prefs.feed.base_url = settings.pages_base_url

    state = StateManager(settings.data_dir)
    run_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # 1. Parse OPML
    opml_path = settings.config_dir / "subscriptions.opml"
    if not opml_path.exists():
        console.print("[red]ERROR: config/subscriptions.opml not found. Copy the example and add your feeds.[/red]")
        sys.exit(1)

    feeds = parse_opml(opml_path)
    active_feeds = [f for f in feeds if not f.is_private]
    private_count = len(feeds) - len(active_feeds)
    if private_count:
        console.print(f"[yellow]Skipping {private_count} private/authenticated feed(s)[/yellow]")

    # Apply show overrides
    show_max_map: dict[str, int] = {}
    disabled_titles: set[str] = set()
    for override in shows_cfg.shows:
        for feed in active_feeds:
            if override.match.lower() in feed.title.lower():
                if not override.enabled:
                    disabled_titles.add(feed.title)
                if override.canonical_feed_url:
                    feed.xml_url = override.canonical_feed_url
                show_max_map[feed.title] = override.max_episodes_per_run

    active_feeds = [f for f in active_feeds if f.title not in disabled_titles]
    followed_urls = {f.xml_url for f in active_feeds}

    console.print(f"[cyan]Scanning {len(active_feeds)} feeds (lookback {settings.lookback_days} days)...[/cyan]")

    # 2. Fetch episodes
    all_episodes, feed_errors = await fetch_all_feeds(
        active_feeds,
        lookback_days=settings.lookback_days,
        max_episodes_per_feed=3,
    )

    # 3. Deduplicate against state
    new_episodes, _ = dedup_episodes(all_episodes, state.seen_guids())
    console.print(f"Found {len(new_episodes)} new episodes (after dedup)")

    # 4. Outside-feed discovery
    llm_stage1 = None
    llm_stage2 = None
    if settings.openai_api_key:
        llm_stage1 = OpenAIProvider(settings.openai_api_key, settings.openai_stage1_model, settings.openai_base_url)
        llm_stage2 = OpenAIProvider(settings.openai_api_key, settings.openai_stage2_model, settings.openai_base_url)
    else:
        console.print("[yellow]WARNING: No OPENAI_API_KEY set. Running metadata-only ranking.[/yellow]")

    outside_episodes = []
    if llm_stage1:
        podcast_search = _make_podcast_search(settings)
        web_search = _make_web_search(settings)
        outside_episodes = await discover_outside_episodes(
            cfg=discovery_cfg,
            prefs=prefs,
            llm=llm_stage1,
            podcast_search=podcast_search,
            web_search=web_search,
            followed_feed_urls=followed_urls,
            weekly_topics=[],
            lookback_days=settings.lookback_days,
        )
        console.print(f"Outside discovery: {len(outside_episodes)} candidates")

    all_candidates = new_episodes + outside_episodes

    # 5. Summarize and rank
    transcription = CascadeTranscriptionProvider(
        openai_api_key=settings.openai_api_key,
        enable_whisper=settings.enable_audio_transcription,
    )
    active_llm = llm_stage2 or llm_stage1

    if active_llm:
        ranked = await process_episodes(
            all_candidates,
            prefs=prefs,
            llm=active_llm,
            transcription=transcription,
            max_deep_process=15,
            total_token_budget=settings.max_llm_tokens_per_run,
        )
    else:
        # Metadata-only fallback
        from .ranking import RubricScore, _classify, stage1_metadata_score
        ranked = []
        for ep in all_candidates:
            s1 = stage1_metadata_score(ep, prefs)
            ranked.append(__import__('podcast_scout.ranking', fromlist=['RankedEpisode']).RankedEpisode(
                episode=ep, score=s1.score, rubric=RubricScore(),
                classification=_classify(s1.score, prefs),
                classification_reason="metadata only (no LLM key)",
                evidence_confidence="low",
                summary=ep.description[:300] or "No summary.",
            ))
        ranked.sort(key=lambda r: r.score, reverse=True)

    # 6. Build daily queue (2hr budget)
    max_minutes = prefs.length.max_weekly_listen_hours * 60  # daily: treat setting as daily cap
    rss_queue, email_only = build_daily_queue(
        ranked,
        max_minutes=max_minutes,
        max_listen_fully=prefs.output_caps.max_listen_fully,
        max_read_summary=prefs.output_caps.max_read_summary,
        max_outside=prefs.output_caps.max_outside_feed,
    )

    console.print(f"Queue: {len(rss_queue)} episodes | Email-only: {len(email_only)} episodes")

    # 7. Weekly synthesis (Fridays or forced)
    synthesis = None
    if run_synthesis and active_llm:
        synthesis = await generate_synthesis(ranked, prefs, active_llm)

    if dry_run:
        console.print("[yellow]Dry run — skipping file writes and email.[/yellow]")
        _print_summary_table(rss_queue, email_only)
        return {"queued": len(rss_queue), "email_only": len(email_only), "errors": feed_errors}

    # 8. Generate outputs
    settings.public_dir.mkdir(parents=True, exist_ok=True)
    base_url = prefs.feed.base_url or settings.pages_base_url

    # RSS feeds
    listen_xml = build_feed(rss_queue, prefs, "listen", base_url, state)
    all_xml = build_feed(rss_queue + email_only, prefs, "all", base_url, state)
    (settings.public_dir / "listen.xml").write_text(listen_xml, encoding="utf-8")
    (settings.public_dir / "all.xml").write_text(all_xml, encoding="utf-8")
    console.print(f"[green]RSS feeds written to {settings.public_dir}[/green]")

    # JSON data
    latest_json = {
        "run_date": run_date,
        "queued": [_ep_to_dict(r) for r in rss_queue],
        "email_only": [_ep_to_dict(r) for r in email_only],
        "synthesis": synthesis.model_dump() if synthesis else None,
        "feed_errors": feed_errors,
    }
    data_dir = settings.public_dir / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "latest.json").write_text(json.dumps(latest_json, indent=2, default=str))

    # HTML briefing
    templates_dir = settings.templates_dir
    if templates_dir.exists():
        render_briefing(
            templates_dir=templates_dir,
            output_path=settings.public_dir / "index.html",
            queued=rss_queue,
            email_only=email_only,
            synthesis=synthesis,
            run_date=run_date,
            feed_url=f"{base_url}/listen.xml" if base_url else "",
            all_feed_url=f"{base_url}/all.xml" if base_url else "",
        )

    # Markdown
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
            is_outside_feed=r.episode.is_outside_feed,
            source_feed_url=r.episode.source_feed_url,
        ))
    state.prune_old()
    state.update_last_run()
    state.snapshot_history(run_date, latest_json)
    state.save()

    # 10. Send email digest
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
        console.print("[yellow]SMTP not configured — skipping email digest.[/yellow]")

    _print_summary_table(rss_queue, email_only)
    return latest_json


def _ep_to_dict(r) -> dict:
    ep = r.episode
    return {
        "guid": ep.guid,
        "show": ep.show_title,
        "title": ep.episode_title,
        "score": round(r.score, 1),
        "classification": r.classification,
        "duration_min": round(ep.duration_minutes, 0),
        "url": ep.episode_url,
        "is_outside": ep.is_outside_feed,
        "summary": r.summary,
        "key_ideas": r.key_ideas,
        "confidence": r.evidence_confidence,
    }


def _print_summary_table(queued, email_only) -> None:
    table = Table(title="Daily Queue", show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Show")
    table.add_column("Episode", max_width=50)
    table.add_column("Score", justify="right")
    table.add_column("Class")
    table.add_column("Min", justify="right")
    for i, r in enumerate(queued, 1):
        table.add_row(
            str(i), r.episode.show_title[:30], r.episode.episode_title[:50],
            f"{r.score:.0f}", r.classification, f"{r.episode.duration_minutes:.0f}"
        )
    console.print(table)
    if email_only:
        console.print(f"[dim]+{len(email_only)} additional episodes sent to email only[/dim]")


@click.group()
@click.option("--verbose", "-v", is_flag=True)
def main(verbose: bool) -> None:
    """Podcast Scout — personal podcast intelligence agent."""
    _setup_logging(verbose)


@main.command()
@click.option("--synthesis", is_flag=True, help="Run weekly cross-episode synthesis (use on Fridays)")
@click.option("--dry-run", is_flag=True, help="Rank and print results without writing files or sending email")
@click.option("--lookback", default=None, type=int, help="Override lookback window in days")
def run(
    synthesis: bool,
    dry_run: bool,
    lookback: int | None,
) -> None:
    """Run the full daily pipeline."""
    settings = Settings()
    if lookback:
        settings.lookback_days = lookback
    result = asyncio.run(_run_pipeline(settings, run_synthesis=synthesis, dry_run=dry_run))
    queued = result.get("queued", 0) if isinstance(result, dict) else len(result)
    console.print(f"[bold green]Done. {queued} episodes queued.[/bold green]")


@main.command()
def validate() -> None:
    """Validate config files and OPML without running the pipeline."""
    settings = Settings()
    prefs = load_preferences(settings.config_dir)
    shows = load_shows(settings.config_dir)
    console.print(f"[green]preferences.yaml OK — {len(prefs.show_priors)} show priors loaded[/green]")
    console.print(f"[green]shows.yaml OK — {len(shows.shows)} overrides loaded[/green]")
    opml_path = settings.config_dir / "subscriptions.opml"
    if opml_path.exists():
        feeds = parse_opml(opml_path)
        private = sum(1 for f in feeds if f.is_private)
        console.print(f"[green]subscriptions.opml OK — {len(feeds)} feeds ({private} private)[/green]")
    else:
        console.print("[yellow]subscriptions.opml not found (copy from subscriptions.opml.example)[/yellow]")
