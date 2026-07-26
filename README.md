# 🎙️ Podcast Scout

> **Personal podcast intelligence agent** — daily AI-ranked briefing for retail & eCommerce product leaders.

Podcast Scout automatically discovers, scores, and curates podcast episodes from your subscribed feeds and the open web. It runs every weekday via GitHub Actions, publishes an RSS feed to GitHub Pages, and optionally sends a styled HTML email digest — all for roughly **$0–2 per week** in LLM costs.

---

## ✨ Features

- **Multi-source discovery** — reads your OPML subscriptions, Podcast Index API, iTunes Search, and web search (Brave / Serper)
- **Two-stage AI ranking** — fast metadata pre-filter (Stage 1) then deep Gemini LLM scoring against your interest rubric (Stage 2)
- **Per-category queues** — episodes are bucketed (e.g. `ai_retail`, `leadership`) so high-volume categories never crowd out others
- **Weekly synthesis** — cross-episode insight report generated on demand
- **GitHub Pages output** — per-category RSS feeds (`{slug}.xml`), legacy `listen.xml` / `all.xml`, `index.html` briefing, and `data/latest.json`
- **Email digest** — HTML email via any SMTP server
- **State management** — deduplication across runs, history snapshots, automatic pruning
- **Zero-infra** — runs entirely inside GitHub Actions; no server required

---

## 🏗️ Architecture

```
podcast-curation-agent/
├── src/podcast_scout/
│   ├── cli.py              # Click CLI + async pipeline orchestrator
│   ├── config.py           # Pydantic Settings + YAML loaders
│   ├── discovery.py        # Multi-source episode discovery
│   ├── feeds.py            # RSS feed parsing helpers
│   ├── normalize.py        # Deduplication & field normalization
│   ├── opml.py             # OPML subscription parser
│   ├── ranking.py          # Stage-1 metadata scoring + Stage-2 LLM rubric
│   ├── summarization.py    # Per-episode LLM summarization
│   ├── synthesis.py        # Weekly cross-episode synthesis
│   ├── render.py           # Jinja2 HTML + Markdown rendering
│   ├── rss.py              # RSS/Atom feed builder
│   ├── state.py            # Run state, seen-GUIDs, history
│   ├── email_digest.py     # SMTP HTML email sender
│   ├── templates/          # Jinja2 HTML templates
│   └── providers/
│       ├── base.py             # Abstract provider interfaces
│       ├── llm.py              # Gemini provider (google-genai)
│       ├── podcast_search.py   # Podcast Index + iTunes providers
│       ├── transcription.py    # Cascade transcription (Whisper optional)
│       └── web_search.py       # Brave / Serper / Null providers
├── config/
│   ├── preferences.yaml            # Your interests, show priors, output caps
│   ├── discovery_queries.yaml      # Custom search query seeds
│   ├── shows.yaml                  # Show-level category & metadata overrides
│   └── subscriptions.opml.example  # Template OPML — copy & rename
├── data/                   # Runtime state (committed by CI bot)
├── public/                 # Generated outputs deployed to GitHub Pages
├── .github/workflows/
│   ├── daily.yml           # Mon–Fri 05:00 UTC pipeline
│   └── weekly_synthesis.yml # Weekly synthesis job
├── .env.example            # All environment variables documented
└── pyproject.toml          # Hatchling build, deps, ruff + mypy config
```

### Pipeline flow

```
OPML + Podcast Index + Web Search
           │
     discover_episodes()
           │
     dedup_episodes()        ← seen-GUIDs from state
           │
   ┌───────▼────────┐
   │ per-category   │  (ai_retail / leadership / …)
   │  Stage-1 score │  metadata heuristics, O(ms)
   │  Stage-2 LLM   │  Gemini rubric, token-budgeted
   │  build_queue() │  RSS + email-only split
   └───────┬────────┘
           │
  write public/*.xml, index.html, data/latest.json
           │
    update state → git commit → GitHub Pages deploy
           │
    send SMTP email digest (optional)
```

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A [Google AI Studio](https://aistudio.google.com/apikey) API key (free tier works)

### Local setup

```bash
# 1. Clone
git clone https://github.com/iitiff/podcast-curation-agent.git
cd podcast-curation-agent

# 2. Install
uv sync            # or: pip install -e .

# 3. Configure environment
cp .env.example .env
# Edit .env — at minimum set GEMINI_API_KEY

# 4. Add your podcast subscriptions
cp config/subscriptions.opml.example config/subscriptions.opml
# Edit config/subscriptions.opml with your actual feeds

# 5. Validate config
uv run podcast-scout validate

# 6. Dry run (no writes, no email)
uv run podcast-scout run --dry-run

# 7. Full run
uv run podcast-scout run
```

---

## ⚙️ Configuration

### Environment variables (`.env`)

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | **Yes** | Google Gemini key for LLM ranking & summarization |
| `GEMINI_STAGE2_MODEL` | No | Override model (default: `gemini-2.5-flash`) |
| `PODCAST_INDEX_KEY` | No | [Podcast Index](https://api.podcastindex.org) key for broader discovery |
| `PODCAST_INDEX_SECRET` | No | Podcast Index secret |
| `WEB_SEARCH_API_KEY` | No | Brave Search or Serper.dev key for outside-feed discovery |
| `WEB_SEARCH_PROVIDER` | No | `brave` (default) or `serper` |
| `ENABLE_AUDIO_TRANSCRIPTION` | No | `true` to enable Whisper transcription (increases cost) |
| `MAX_COST_USD_PER_RUN` | No | Cost circuit-breaker (default: `2.00`) |
| `MAX_LLM_TOKENS_PER_RUN` | No | Token budget across all categories (default: `500000`) |
| `SMTP_HOST` | No | SMTP server for email digest |
| `SMTP_PORT` | No | SMTP port (default: `587`) |
| `SMTP_USER` | No | SMTP login username |
| `SMTP_PASSWORD` | No | SMTP password |
| `SMTP_TO` | No | Digest recipient address |
| `PAGES_BASE_URL` | No | Base URL for feed links (set automatically in CI) |

### `config/preferences.yaml`

Defines your interest topics, show priors (boost/block shows by name), listen-time budget, per-category output caps, and the scoring rubric weights used by the LLM.

### `config/shows.yaml`

Overrides display name and **category** for individual shows. The category slug determines which RSS feed a show's episodes appear in.

### `config/discovery_queries.yaml`

Custom keyword queries injected into the web search and podcast search providers to surface episodes beyond your subscribed feeds.

---

## 🤖 GitHub Actions Automation

### Daily pipeline (Mon–Fri, 05:00 UTC)

Configured in [`.github/workflows/daily.yml`](.github/workflows/daily.yml). Supports `workflow_dispatch` with optional `lookback_days` and `dry_run` inputs.

**Required repository secrets:**

```
GEMINI_API_KEY
PODCAST_INDEX_KEY       # optional
PODCAST_INDEX_SECRET    # optional
WEB_SEARCH_API_KEY      # optional
SMTP_HOST               # optional
SMTP_PORT               # optional
SMTP_USER               # optional
SMTP_PASSWORD           # optional
SMTP_TO                 # optional
PAGES_BASE_URL          # e.g. https://iitiff.github.io/podcast-curation-agent
```

**One-time setup:**
1. Go to **Settings → Pages** and set source to **GitHub Actions**
2. Add the secrets above under **Settings → Secrets and variables → Actions**
3. Trigger manually via **Actions → Daily Podcast Scout → Run workflow** for the first run

### Weekly synthesis

Configured in [`.github/workflows/weekly_synthesis.yml`](.github/workflows/weekly_synthesis.yml). Generates a cross-episode insight report appended to the briefing.

---

## 📤 Outputs

After each run, the following are published to **GitHub Pages** at `https://<user>.github.io/<repo>/`:

| Path | Description |
|---|---|
| `index.html` | Human-readable weekly briefing |
| `listen.xml` | RSS feed — listen-queue episodes only |
| `all.xml` | RSS feed — all surfaced episodes |
| `{category-slug}.xml` | Per-category RSS feeds |
| `data/latest.json` | Machine-readable run output with scores, summaries, key ideas |
| `latest.md` | Markdown version of the briefing |

Add `listen.xml` (or any `{slug}.xml`) to your podcast app to receive your curated queue.

---

## 🛠️ Development

```bash
# Install with dev extras
uv sync --extra dev

# Lint
uv run ruff check src/

# Type check
uv run mypy src/

# Tests
uv run pytest
```

The project uses:
- **[Ruff](https://docs.astral.sh/ruff/)** for linting (line length 100, Python 3.12 target)
- **[mypy](https://mypy.readthedocs.io/)** in strict mode
- **[pytest-asyncio](https://pytest-asyncio.readthedocs.io/)** with `asyncio_mode = auto`

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `google-genai` | Gemini LLM for ranking & summarization |
| `feedparser` | RSS/Atom feed parsing |
| `httpx` | Async HTTP for all external calls |
| `pydantic` / `pydantic-settings` | Config models & env loading |
| `rapidfuzz` | Fuzzy show-name matching |
| `Jinja2` | HTML briefing & email templates |
| `tenacity` | Retry logic for flaky API calls |
| `rich` + `click` | CLI output and commands |

---

## 🗺️ Roadmap

- [ ] Web UI for preference editing
- [ ] Slack / Telegram digest delivery
- [ ] Additional LLM providers (OpenAI, Anthropic)
- [ ] Listener analytics dashboard
- [ ] OPML export of curated subscriptions

---

## 📄 License

MIT — see [LICENSE](LICENSE).
