"""CLI entry point and main pipeline orchestrator."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from .config import Settings, load_discovery, load_preferences, load_show_config
from .discovery import discover_episodes
from .email_digest import SMTPConfig, build_email_html, send_digest
from .normalize import NormalizedEpisode, clean_snippet, dedup_episodes
from .providers.base import BaseLLMProvider
from .providers.llm import FallbackLLMProvider, GeminiProvider, GitHubModelsProvider
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

_DEFAULT_CATEGORY = "ai_retail"

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


def _make_podcast_search(
    settings: Settings,
) -> PodcastIndexProvider | ITunesSearchProvider:
    if settings.podcast_index_key and settings.podcast_index_secret:
        return PodcastIndexProvider(settings.podcast_index_key, settings.podcast_index_secret)
    return ITunesSearchProvider()


def _make_web_search(
    settings: Settings,
) -> BraveSearchProvider | SerperSearchProvider | NullWebSearchProvider:
    if settings.web_search_api_key:
        if settings.web_search_provider == "serper":
            return SerperSearchProvider(settings.web_search_api_key)
        return BraveSearchProvider(settings.web_search_api_key)
    return NullWebSearchProvider()


def _make_llm(settings: Settings) -> BaseLLMProvider | None:
    """Build the LLM provider: GitHub Models primary, Gemini as a true runtime
    fallback when both credentials are available.

    IMPORTANT — regression history: this function was previously restored to
    a single-provider version by a hotfix (PR#10) that was accidentally based
    on a stale local file predating this fallback wiring. That silently
    reverted true runtime fallback for several days while looking unrelated
    to the change actually being shipped. If touching this function again,
    always diff against the live file on GitHub first, never a cached copy.

    Since GITHUB_TOKEN is always present in GitHub Actions, a naive
    "GitHub Models if token else Gemini" check always picks GitHub Models and
    NEVER tries Gemini even when GEMINI_API_KEY is configured. Wrapping both
    providers in FallbackLLMProvider makes the fallback happen on every LLM
    call (per-batch), not just once at startup.
    """
    primary: BaseLLMProvider | None = None
    secondary: BaseLLMProvider | None = None
    primary_name = secondary_name = ""

    if settings.github_token:
        primary = GitHubModelsProvider(settings.github_token, settings.github_models_model)
        primary_name = f"GitHub Models ({settings.github_models_model})"

    if settings.gemini_api_key:
        gemini = GeminiProvider(settings.gemini_api_key, settings.gemini_stage2_model)
        if primary is None:
            primary = gemini
            primary_name = f"Gemini ({settings.gemini_stage2_model})"
        else:
            secondary = gemini
            secondary_name = f"Gemini ({settings.gemini_stage2_model})"

    if primary is None:
        console.print(
            "[red]WARNING: No GITHUB_TOKEN or GEMINI_API_KEY — running metadata-only ranking.[/red]"
        )
        return None

    if secondary is not None:
        console.print(f"[cyan]LLM: {primary_name} -> runtime fallback {secondary_name}[/cyan]")
        return FallbackLLMProvider(primary, secondary, primary_name, secondary_name)

    console.print(f"[cyan]LLM: {primary_name} (no fallback configured)[/cyan]")
    return primary


def _ascii_clean(value: str) -> str:
    """Strip non-ASCII characters (e.g. a pasted U+00A0 non-breaking space)
    and surrounding whitespace from an env-derived identifier.

    Root cause history: smtplib's AUTH PLAIN/LOGIN mechanism calls
    `.encode("ascii")` directly on the username AND password inside
    `server.login()` — a code path entirely separate from the email message
    headers. Sanitizing only `From`/`To` in email_digest.py did NOT fix this,
    because the crash happens during authentication, before any message is
    even built. Cleaning every SMTP identifier at the single source (here)
    closes off all of those call sites at once.
    """
    return value.encode("ascii", "ignore").decode("ascii").strip()


def _smtp_from_env() -> SMTPConfig | None:
    # Use `or` fallback instead of the default= arg so that empty-string env
    # vars injected by GitHub Actions for unset secrets are treated as absent.
    host = _ascii_clean(os.getenv("SMTP_HOST") or "")
    if not host:
        return None
    user = _ascii_clean(os.getenv("SMTP_USER") or "")
    if not user:
        log.warning("SMTP_HOST is set but SMTP_USER is empty — skipping email.")
        return None
    return SMTPConfig(
        host=host,
        port=int((os.getenv("SMTP_PORT") or "587").strip()),
        user=user,
        # smtplib's AUTH PLAIN/LOGIN mechanisms call .encode("ascii") on BOTH
        # the username and the password. Real SMTP passwords (app passwords,
        # etc.) are always plain ASCII, so it's safe to strip any stray
        # non-ASCII byte here too.
        password=_ascii_clean(os.getenv("SMTP_PASSWORD") or ""),
        to=_ascii_clean(os.getenv("SMTP_TO") or user),
        from_addr=_ascii_clean(os.getenv("SMTP_FROM") or user),
        use_tls=(os.getenv("SMTP_USE_TLS") or "true").strip().lower() != "false",
    )


def _build_category_map(config_dir: Path) -> dict[str, str]:
    shows = load_show_config(config_dir)
    mapping: dict[str, str] = {}
    for show in shows.shows:
        name = show.display_name or show.match
        cat = show.category or _DEFAULT_CATEGORY
        mapping[name.lower()] = cat
        mapping[show.match.lower()] = cat
    return mapping


def _resolve_category(show_title: str, category_map: dict[str, str]) -> str:
    title_lower = show_title.lower()
    if title_lower in category_map:
        return category_map[title_lower]
    for key, cat in category_map.items():
        if key and (key in title_lower or title_lower in key):
            return cat
    return _DEFAULT_CATEGORY


def _tag_episodes_with_category(
    episodes: list[RankedEpisode],
    category_map: dict[str, str],
) -> None:
    for r in episodes:
        cat = _resolve_category(r.episode.show_title, category_map)
        r.episode.category = cat


def _load_carryover_candidates(
    state: StateManager,
    category_map: dict[str, str],
    lookback_days: int,
    already_scored_this_run: set[str],
) -> dict[str, list[RankedEpisode]]:
    """Rebuild RankedEpisode stubs for previously scored episodes that have not
    yet won a category-feed (playlist) slot.

    Carryover pool includes:
    - Episodes scored in previous runs, within the lookback window
    - Episodes that appeared in email digest / all.xml only (not playlisted)

    Excluded:
    - Episodes that have already won a category-feed slot (playlist_guids)
    - Skip / unclassified episodes
    - Episodes scored in the current run (already in newly_ranked)

    Using playlist_guids (not published_guids) means email-only episodes remain
    in the pool until they win a Pocket Casts category slot.
    """
    from .normalize import utcnow
    cutoff = utcnow() - timedelta(days=lookback_days)
    carryover: dict[str, list[RankedEpisode]] = {}

    processed = state._state.get("processed", {})
    already_playlisted = state.playlist_guids()

    for guid, rec_data in processed.items():
        # Don't double-count episodes scored in this run
        if guid in already_scored_this_run:
            continue
        # Only exclude episodes that won an actual category-feed playlist slot.
        # Episodes published to all.xml / email digest only remain eligible.
        if guid in already_playlisted:
            continue
        try:
            rec = EpisodeRecord(**rec_data)
        except Exception:
            continue
        # Only carry forward within the lookback window
        if rec.processed_at and rec.processed_at < cutoff:
            continue
        # Skip low-value classifications
        if rec.classification in ("Skip", None, ""):
            continue

        ep = NormalizedEpisode(
            guid=rec.guid,
            show_title=rec.show_title,
            episode_title=rec.episode_title,
            description="",
            published=rec.published or utcnow(),
            duration_seconds=0,
            episode_url="",
            source_feed_url=rec.source_feed_url,
        )
        cat = _resolve_category(rec.show_title, category_map)
        ep.category = cat

        ranked = RankedEpisode(
            episode=ep,
            score=rec.score,
            rubric=RubricScore(),
            classification=rec.classification,
            classification_reason="carried over from previous run",
            evidence_confidence="low",
            summary="",
        )
        carryover.setdefault(cat, []).append(ranked)

    return carryover


def _load_accumulated_this_week(
    state: StateManager,
    category_map: dict[str, str],
    lookback_days: int = 7,
) -> list[RankedEpisode]:
    """Return all scored-but-not-yet-playlisted episodes from the past week.

    These are episodes that scored well enough to surface (not Skip) but never
    won a daily category-feed slot. They are surfaced in the Friday weekly digest
    email so good episodes don't disappear unseen.

    Unlike the carryover pool (which uses settings.lookback_days), this always
    looks back a full 7 days to give the complete weekly picture.
    """
    from .normalize import utcnow
    cutoff = utcnow() - timedelta(days=lookback_days)
    playlisted = state.playlist_guids()
    accumulated: list[RankedEpisode] = []

    processed = state._state.get("processed", {})
    for guid, rec_data in processed.items():
        if guid in playlisted:
            continue
        try:
            rec = EpisodeRecord(**rec_data)
        except Exception:
            continue
        if rec.processed_at and rec.processed_at < cutoff:
            continue
        if rec.classification in ("Skip", None, ""):
            continue

        ep = NormalizedEpisode(
            guid=rec.guid,
            show_title=rec.show_title,
            episode_title=rec.episode_title,
            description="",
            published=rec.published or utcnow(),
            duration_seconds=0,
            episode_url="",
            source_feed_url=rec.source_feed_url,
        )
        cat = _resolve_category(rec.show_title, category_map)
        ep.category = cat

        ranked = RankedEpisode(
            episode=ep,
            score=rec.score,
            rubric=RubricScore(),
            classification=rec.classification,
            classification_reason="accumulated — did not win a daily playlist slot this week",
            evidence_confidence="low",
            summary="",
        )
        accumulated.append(ranked)

    accumulated.sort(key=lambda r: r.score, reverse=True)
    return accumulated


async def _run_pipeline(
    settings: Settings,
    run_synthesis: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    prefs = load_preferences(settings.config_dir)
    discovery_cfg = load_discovery(settings.config_dir)
    shows_cfg = load_show_config(settings.config_dir)
    category_map = _build_category_map(settings.config_dir)

    if settings.pages_base_url:
        prefs.feed.base_url = settings.pages_base_url

    state = StateManager(settings.data_dir)
    run_date = datetime.now(tz=UTC).strftime("%Y-%m-%d")

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
        shows_cfg=shows_cfg,
    )

    # 2. Dedup — only truly new episodes get LLM scoring
    new_episodes, _ = dedup_episodes(all_candidates, state.seen_guids())
    console.print(f"Found {len(new_episodes)} new candidates (after dedup from {len(all_candidates)} discovered)")

    # 3. Categorise new episodes
    active_categories = list(prefs.categories.keys()) if prefs.categories else [_DEFAULT_CATEGORY]
    episodes_by_category: dict[str, list[NormalizedEpisode]] = {category: [] for category in active_categories}
    for episode in new_episodes:
        category = _resolve_category(episode.show_title, category_map)
        episode.category = category
        episodes_by_category.setdefault(category, []).append(episode)

    # 4. LLM — GitHub Models (primary) → Gemini (runtime fallback) → metadata-only
    llm = _make_llm(settings)

    # 5. Rank new episodes
    transcription = CascadeTranscriptionProvider(
        openai_api_key=None,
        enable_whisper=settings.enable_audio_transcription,
    )
    non_empty_categories = [category for category in active_categories if episodes_by_category.get(category)]
    token_budget_per_category = settings.max_llm_tokens_per_run // max(1, len(non_empty_categories))
    newly_ranked: dict[str, list[RankedEpisode]] = {}

    minutes_budget = max(480.0, prefs.length.max_weekly_listen_hours * 60)

    for category in active_categories:
        candidates = episodes_by_category.get(category, [])
        if not candidates:
            newly_ranked[category] = []
            continue
        if llm:
            category_ranked = await process_episodes(
                candidates,
                prefs=prefs, llm=llm, transcription=transcription,
                max_deep_process=15, total_token_budget=token_budget_per_category,
            )
        else:
            category_ranked = [
                RankedEpisode(episode=episode, score=s1.score, rubric=RubricScore(),
                    classification=_classify(s1.score, prefs),
                    classification_reason="metadata only (no LLM key)",
                    evidence_confidence="low",
                    summary=clean_snippet(episode.description, 300) or "No summary available.")
                for episode in candidates
                for s1 in [stage1_metadata_score(episode, prefs)]
            ]
            category_ranked.sort(key=lambda item: item.score, reverse=True)
        newly_ranked[category] = category_ranked

    # 6. Persist new scores to state BEFORE carry-over so we don't re-LLM them tomorrow
    from .normalize import utcnow
    for category, cat_ranked in newly_ranked.items():
        for r in cat_ranked:
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

    # 7. Merge carryover: episodes scored in previous runs that have not yet won
    # a category-feed slot compete alongside today's new episodes. Episodes that
    # appeared in email digest / all.xml only are still eligible (we use
    # playlist_guids, not published_guids, as the exclusion set).
    current_run_guids: set[str] = {
        r.episode.guid
        for cat_ranked in newly_ranked.values()
        for r in cat_ranked
    }
    carryover = _load_carryover_candidates(
        state=state,
        category_map=category_map,
        lookback_days=settings.lookback_days,
        already_scored_this_run=current_run_guids,
    )
    if carryover:
        total_carryover = sum(len(v) for v in carryover.values())
        console.print(f"[dim]Carrying over {total_carryover} previously scored episode(s)[/dim]")

    ranked: list[RankedEpisode] = []
    rss_queue: list[RankedEpisode] = []
    email_only: list[RankedEpisode] = []

    for category in active_categories:
        fresh = newly_ranked.get(category, [])
        carried = carryover.get(category, [])
        # Merge fresh + carryover, sort by score descending
        combined = sorted(fresh + carried, key=lambda r: r.score, reverse=True)

        category_cfg = prefs.categories.get(category)
        category_rss, category_email = build_daily_queue(
            combined,
            max_minutes=minutes_budget,
            max_listen_fully=category_cfg.max_listen_fully if category_cfg else prefs.output_caps.max_listen_fully,
            max_read_summary=category_cfg.max_read_summary if category_cfg else prefs.output_caps.max_read_summary,
            max_outside=prefs.output_caps.max_outside_feed,
        )
        ranked.extend(combined)
        rss_queue.extend(category_rss)
        email_only.extend(category_email)

    ranked.sort(key=lambda item: item.score, reverse=True)
    rss_queue.sort(key=lambda item: item.score, reverse=True)
    email_only.sort(key=lambda item: item.score, reverse=True)
    all_surfaced = rss_queue + email_only

    console.print(f"Queue: {len(rss_queue)} episodes across {len(active_categories)} categories | Email-only: {len(email_only)}")

    # 8. Weekly synthesis + accumulated digest
    # When --synthesis is passed (Friday runs), load all episodes from the past
    # week that scored well but never won a category-feed playlist slot.
    # These are surfaced in the Friday email as a "didn't make the cut" digest.
    synthesis = None
    accumulated_this_week: list[RankedEpisode] = []
    if run_synthesis:
        accumulated_this_week = _load_accumulated_this_week(
            state=state,
            category_map=category_map,
            lookback_days=7,
        )
        if accumulated_this_week:
            console.print(
                f"[dim]Weekly digest: {len(accumulated_this_week)} accumulated episode(s) "
                f"that didn't win a playlist slot this week[/dim]"
            )
        if llm:
            synthesis = await generate_synthesis(ranked + accumulated_this_week, prefs, llm)

    if dry_run:
        console.print("[yellow]Dry run — skipping writes and email.[/yellow]")
        _print_summary_table(rss_queue, email_only)
        return {"queued": len(rss_queue), "email_only": len(email_only), "errors": {}}

    # 9. Write outputs
    settings.public_dir.mkdir(parents=True, exist_ok=True)
    base_url = prefs.feed.base_url or settings.pages_base_url

    for cat_key in active_categories:
        cat_cfg = prefs.categories.get(cat_key)
        slug = cat_cfg.slug if cat_cfg else cat_key.replace("_", "-")
        xml = build_category_feed(
            episodes=all_surfaced,
            category=cat_key,
            prefs=prefs,
            base_url=base_url,
            state=state,
            public_dir=settings.public_dir,
        )
        (settings.public_dir / f"{slug}.xml").write_text(xml, encoding="utf-8")
        console.print(f"  [green]Wrote public/{slug}.xml[/green]")

    listen_xml = build_feed(rss_queue, prefs, "listen", base_url, state, public_dir=settings.public_dir)
    all_xml = build_feed(all_surfaced, prefs, "all", base_url, state, public_dir=settings.public_dir)
    (settings.public_dir / "listen.xml").write_text(listen_xml, encoding="utf-8")
    (settings.public_dir / "all.xml").write_text(all_xml, encoding="utf-8")

    data_dir = settings.public_dir / "data"
    data_dir.mkdir(exist_ok=True)
    latest_json: dict[str, object] = {
        "run_date": run_date,
        "queued": [_ep_to_dict(r) for r in rss_queue],
        "email_only": [_ep_to_dict(r) for r in email_only],
        "accumulated_week": [_ep_to_dict(r) for r in accumulated_this_week],
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

    # 10. Mark queued episodes as published in state.
    # Note: add_to_playlist is already called inside build_category_feed for
    # new items that won a category-feed slot — this step handles the broader
    # add_published for listen.xml items that may not have an enclosure match.
    for r in rss_queue:
        state.add_published(r.episode.guid)

    state.prune_old()
    state.update_last_run()
    state.snapshot_history(run_date, latest_json)
    state.save()

    # 11. Email digest
    smtp = _smtp_from_env()
    if smtp:
        feed_url = f"{base_url}/listen.xml" if base_url else ""
        html_body = build_email_html(
            rss_queue, email_only, run_date, feed_url,
            accumulated_week=accumulated_this_week,
        )
        subject = f"Your Podcast Scout — {run_date} ({len(rss_queue)} queued)"
        try:
            send_digest(smtp, subject, html_body)
        except Exception as exc:
            console.print(f"[red]Email failed: {exc}[/red]")
    else:
        console.print("[yellow]SMTP not configured — skipping email.[/yellow]")

    _print_summary_table(rss_queue, email_only)
    return latest_json


def _ep_to_dict(r: RankedEpisode) -> dict[str, object]:
    ep = r.episode
    return {
        "guid": ep.guid,
        "show": ep.show_title,
        "title": ep.episode_title,
        "score": round(r.score, 1),
        "classification": r.classification,
        "classification_reason": r.classification_reason,
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
