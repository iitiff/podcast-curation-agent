# 🎙️ Podcast Scout

> **Personal podcast intelligence agent** — daily AI-ranked briefing for product leaders at the intersection of AI, retail, and eCommerce.

Podcast Scout automatically discovers, scores, and curates podcast episodes from your subscribed feeds and the open web. It runs every weekday via GitHub Actions, publishes per-category RSS feeds to GitHub Pages, and sends a styled HTML email digest.

> **Engine / instance split.** This repo is the **engine**: code, example configs, tests, and nothing personal. Your preferences, state, history, and brain belong in a separate **private instance repo** that installs this package as a dependency. See [MIGRATION.md](MIGRATION.md).

---

## 📡 Live Feeds

Subscribe to these RSS feeds in any podcast app (Overcast, Pocket Casts, Castro, etc.) to receive a curated AI-ranked queue — no setup required:

| Category | RSS Feed |
|---|---|
| 🤖 AI & Retail | `https://feeds.feedburner.com/Podcast-scout/ai-retail` |
| 🚀 Startup & Strategy | `https://feeds.feedburner.com/Podcast-scout/startup` |

📄 **[View the latest briefing →](https://iitiff.github.io/podcast-curation-agent/)**

---

## ✨ Features

- **Multi-source discovery** — polls RSS feeds directly for followed shows; uses Podcast Index API and web search (Brave / Serper) for outside-feed discovery
- **Two-stage AI ranking** — fast metadata pre-filter (Stage 1) then deep LLM scoring against a persona-aware rubric (Stage 2)
- **Pluggable LLM** — Gemini primary, with any OpenAI-compatible endpoint as an automatic per-call fallback (defaults to NVIDIA NIM)
- **Durable brain** — admitted signals become markdown Source pages linked to the Theses they bear on
- **Falsifier watch** — every active thesis declares what would change your mind; each run tests the day's signals against those falsifiers and leads the brief with whatever argues *against* you
- **Per-category queues** — episodes are bucketed into `ai_retail`, `startup`, and `personal_growth` so high-volume categories never crowd out others
- **Weekly synthesis** — cross-episode insight report generated on demand
- **GitHub Pages output** — per-category RSS feeds (`{slug}.xml`), `listen.xml` / `all.xml`, `index.html` briefing, and `data/latest.json`
- **Email digest** — HTML email via any SMTP server
- **State management** — deduplication across runs, history snapshots, automatic pruning
- **Zero-infra** — runs entirely inside GitHub Actions; no server required

---

## 🏗️ Architecture

```
podcast-curation-agent/
├── src/podcast_scout/
│   ├── cli.py              # Click CLI + async pipeline orchestrator
│   ├── config.py           # Settings + YAML loaders
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
│   ├── brain/              # Durable markdown brain
│   │   ├── schema.py           # Frontmatter + Thesis/Source/Pattern/Company
│   │   ├── store.py            # Read/write pages, evidence links, index
│   │   ├── falsifier.py        # Test signals against thesis falsifiers
│   │   └── writer.py           # Signals -> Source pages -> thesis evidence
│   ├── templates/          # Jinja2 HTML templates
│   └── providers/
│       ├── base.py             # Abstract provider interfaces
│       ├── llm.py              # Gemini (primary) + OpenAI-compatible (fallback)
│       ├── podcast_search.py   # Podcast Index + iTunes providers
│       ├── transcription.py    # Cascade transcription (Whisper optional)
│       └── web_search.py       # Brave / Serper / Null providers
├── config/
│   ├── preferences.example.yaml        # Persona, show priors, output caps, watchlists
│   ├── discovery_queries.example.yaml  # Custom search query seeds
│   ├── shows.example.yaml              # Per-show category & feed URL overrides
│   └── subscriptions.opml.example      # Template OPML — copy & rename
├── .github/workflows/
│   └── ci.yml              # ruff, mypy, pytest on every PR
├── scripts/
│   └── make_instance.py    # Assemble the private instance repo
├── .env.example            # All environment variables documented
├── MIGRATION.md            # Engine/instance split runbook
└── pyproject.toml          # Hatchling build, deps, ruff + mypy config
```

### Pipeline flow

```
RSS (followed shows) + Podcast Index + Web Search
           │
     discover_episodes()
           │
     dedup_episodes()        ← seen-GUIDs from state
           │
   ┌───────▼────────┐
   │ per-category   │  (ai_retail / startup / personal_growth)
   │  Stage-1 score │  show prior + guest/competitor signals, O(ms)
   │  Stage-2 LLM   │  Gemini, with OpenAI-compatible fallback
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

## 🧠 The brain

Ranking answers "what should I read today". The brain answers "what do I now
believe, and what would change my mind".

```bash
export BRAIN_DIR=brain
podcast-scout brain init      # scaffold + seed draft theses
podcast-scout brain status    # confidence and falsifier coverage
podcast-scout brain ask "..." # add an open question
podcast-scout brain review    # monthly "State of My Thinking"
```

Each Thesis page carries a **falsifier**: the evidence that would change your
mind. On every run, admitted episodes become `Source` pages, and one batched
LLM call tests them against the falsifiers of every active thesis. Anything
that *cuts against* a belief leads the brief — above the queue, in the email,
and in the HTML.

This exists because ranking optimises for relevance, which by construction
surfaces material that agrees with you. Without an explicit check against
falsifiers, a knowledge base accumulates confirmation and calls it learning.

Evidence links are appended to the thesis page with a date, a link, and the
reasoning. Weak hits are shown in the brief but never written, so the evidence
trail stays defensible rather than exhaustive. Writes are idempotent: re-running
over the same signals does not duplicate anything.

`BRAIN_DIR` unset disables all of it — the pipeline runs exactly as before.

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- A `GEMINI_API_KEY` (or any OpenAI-compatible endpoint via `LLM_FALLBACK_*`)

### Local setup

```bash
# 1. Clone
git clone https://github.com/iitiff/podcast-curation-agent.git
cd podcast-curation-agent

# 2. Install
uv sync            # or: pip install -e .

# 3. Configure environment
cp .env.example .env
# Edit .env — set GEMINI_API_KEY (and/or LLM_FALLBACK_API_KEY)

# 4. Validate config
uv run podcast-scout validate

# 5. Dry run (no writes, no email)
uv run podcast-scout run --dry-run

# 6. Full run
uv run podcast-scout run
```

---

## ⚙️ Configuration

### Environment variables (`.env`)

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | **Yes*** | Primary LLM. \*At least one of this or `LLM_FALLBACK_API_KEY` is required; with neither, every episode scores at the metadata floor and the curated feed stops updating. |
| `LLM_FALLBACK_API_KEY` | No | Any OpenAI-compatible endpoint, used automatically on any primary failure. Configure both to get retry behaviour. |
| `BRAIN_DIR` | No | Enables the brain and falsifier watch. Unset disables all brain writes. |
| `BRIEFING_DIR` | No | Where `index.html` / `latest.md` / `latest.json` are written. Defaults to `PUBLIC_DIR`; set it separately to keep the briefing out of what gets published. |
| `GEMINI_STAGE2_MODEL` | No | Override Gemini model (default: `gemini-3.6-flash`). Free-tier quota is **per model**, so this is the first thing to change on a 429. |
| `GEMINI_THINKING_BUDGET` | No | Thinking token budget (default `0`, disabled). Set `none` to omit the field for models that reject it. |
| `PODCAST_INDEX_KEY` | No | [Podcast Index](https://api.podcastindex.org) key for broader discovery |
| `PODCAST_INDEX_SECRET` | No | Podcast Index secret |
| `WEB_SEARCH_API_KEY` | No | Brave Search or Serper.dev key for outside-feed discovery |
| `WEB_SEARCH_PROVIDER` | No | `brave` (default) or `serper` |
| `ENABLE_AUDIO_TRANSCRIPTION` | No | `true` to enable Whisper transcription (increases cost) |
| `MAX_COST_USD_PER_RUN` | No | Cost circuit-breaker (default: `2.00`) |
| `MAX_LLM_TOKENS_PER_RUN` | No | Token budget across all categories (default: `500000`) |
| `SMTP_HOST` | No | SMTP server for email digest |
| `SMTP_PORT` | No | SMTP port — use `465` for SSL or `587` for STARTTLS |
| `SMTP_USER` | No | SMTP login username |
| `SMTP_PASSWORD` | No | SMTP password |
| `SMTP_TO` | No | Digest recipient address |
| `PAGES_BASE_URL` | No | Base URL for feed links (set automatically in CI) |

### `config/preferences.yaml`

Defines your **persona** (role, focus, seniority), **show priors** (per-show relevance weights), listen-time budget, per-category output caps, guest and competitor watchlists, and topic exclusions. Relevance scoring is delegated entirely to the Stage 2 LLM rubric.

### `config/shows.yaml`

Overrides display name, **category**, and RSS feed URL for individual shows. The category slug determines which RSS feed a show's episodes appear in (`ai_retail`, `startup`, or `personal_growth`). Add a `canonical_feed_url` here to fix shows that can't be resolved via iTunes search.

### `config/discovery_queries.yaml`

Custom keyword queries injected into the web search and podcast search providers to surface episodes beyond your subscribed feeds.

---

## 🤖 GitHub Actions Automation

### Daily pipeline (Mon–Fri, 05:00 UTC)

Scheduling lives in the **instance** repo, not here. `scripts/make_instance.py` generates a `daily.yml` that installs this package at a pinned commit and supports `workflow_dispatch` with `lookback_days` and `dry_run` inputs.

**Required repository secrets:**

```
GEMINI_API_KEY          # primary LLM
PODCAST_INDEX_KEY       # optional
PODCAST_INDEX_SECRET    # optional
WEB_SEARCH_API_KEY      # optional
SMTP_HOST               # optional
SMTP_PORT               # optional — use 465 (SSL) or 587 (STARTTLS)
SMTP_USER               # optional
SMTP_PASSWORD           # optional
SMTP_TO                 # optional
PAGES_BASE_URL          # e.g. https://iitiff.github.io/podcast-curation-agent
```

**One-time setup:**
1. Go to **Settings → Pages** and set source to **GitHub Actions**
2. Add the secrets above under **Settings → Secrets and variables → Actions**
3. Trigger manually via **Actions → Daily Podcast Scout → Run workflow** for the first run

> **GitHub Models is gone.** GitHub retired the product on 2026-07-30; the inference endpoint returns `410 Gone` and there is no replacement. Earlier revisions of this README described it as the free primary LLM — that is no longer true. Configure `GEMINI_API_KEY`, and ideally `LLM_FALLBACK_API_KEY` as well.

### Earnings via SEC EDGAR

The design docs rank earnings highest because it is the only class that can
contradict vendor marketing. The obstacle was distribution: most IR sites
publish no feed. EDGAR is the feed they do not provide — every US-listed
company files quarterly results as an 8-K tagged item 2.02, with the numbers
in exhibit EX-99.1.

```yaml
# config/sources.yaml
edgar:
  companies:
    - name: "Walmart"
      ticker: WMT      # or cik:
      tags: [competitive, market]
```

```bash
export SEC_USER_AGENT="Your Name (you@example.com)"
```

**`SEC_USER_AGENT` is required.** The SEC refuses requests that do not declare
a contact address, and answers with a throttle page rather than an error — so
the adapter disables itself when it is unset rather than inventing one. In the
instance repo it is a repository *variable*, not a secret: it is a contact
address, not a credential.

### Three cadences

| When | What | Command |
| --- | --- | --- |
| Mon–Fri | Daily brief and feeds | `podcast-scout run` |
| Fri | Adds the weekly cross-episode synthesis | `podcast-scout run --synthesis` |
| 1st of the month | *State of My Thinking* over the brain | `podcast-scout brain review` |

All three are scheduled from the instance repo; `make_instance.py` writes the
workflows. The weekly is a synthesis of what arrived. The monthly is the only
one that reads the **brain** rather than a window of the feed: it takes each
open question with the findings attached to it and asks whether the evidence
actually moved, which is a question a week of episodes cannot answer.

---

## 📤 Outputs

After each run, the following are published to **GitHub Pages** at `https://iitiff.github.io/podcast-curation-agent/`:

| Path | Description |
|---|---|
| `index.html` | Human-readable weekly briefing |
| `listen.xml` | RSS feed — listen-queue episodes only |
| `all.xml` | RSS feed — all surfaced episodes |
| `ai-retail.xml` | AI, retail & product craft episodes |
| `startup.xml` | Startup & business strategy episodes |
| `personal-growth.xml` | Personal growth & mindfulness episodes |
| `data/latest.json` | Machine-readable run output with scores, summaries, key ideas |
| `latest.md` | Markdown version of the briefing |

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
| `openai` | OpenAI-compatible fallback client (NVIDIA NIM by default) |
| `google-genai` | Gemini LLM fallback for ranking & summarization |
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
- [ ] Fix SMTP email delivery
- [ ] Listener analytics dashboard
- [ ] OPML export of curated subscriptions
- [ ] Fix broken RSS feeds (Future Commerce, Founders, Masters of Scale, AI + a16z, Lenny's Podcast)

---

## 📄 License

MIT — see [LICENSE](LICENSE).
