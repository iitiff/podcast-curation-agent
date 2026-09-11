#!/usr/bin/env python3
"""Assemble the PRIVATE instance repo from this public engine repo.

Splits the project in two:

  engine   (this repo, public)   code, example configs, tests
  instance (the new repo, PRIVATE) your config, your state, your brain

Everything that says something about the reader -- which shows they follow,
which companies they track, what they have already read -- belongs in the
instance. The engine keeps nothing personal.

This script only ever *reads* the current repo and *writes* into the target
directory. It never deletes anything here, so running it cannot break the live
pipeline. Removing the personal files from this repo is a separate, deliberate
step; see MIGRATION.md.

Usage:
    python scripts/make_instance.py ../podcast-brain --public-repo iitiff/podcast-curation-agent
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Copied verbatim into the instance. These are the files that make the pipeline
# *yours* rather than a generic engine.
PERSONAL_PATHS = [
    ("config/preferences.yaml", "config/preferences.yaml"),
    ("config/shows.yaml", "config/shows.yaml"),
    ("config/discovery_queries.yaml", "config/discovery_queries.yaml"),
    ("data/state.json", "data/state.json"),
    ("data/feedback.csv", "data/feedback.csv"),
    ("data/history", "data/history"),
]

GITIGNORE = """\
.env
.venv/
__pycache__/
*.pyc

# Generated each run and pushed to the public feeds repo; not source of truth.
public/
"""

ENV_EXAMPLE = """\
# ---- LLM (at least one required) ----------------------------------------
# GitHub Models was retired 2026-07-30 and returns 410 Gone. Gemini is primary.
GEMINI_API_KEY=
LLM_FALLBACK_API_KEY=
LLM_FALLBACK_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_FALLBACK_MODEL=meta/llama-3.3-70b-instruct

# ---- Discovery (optional) -----------------------------------------------
PODCAST_INDEX_KEY=
PODCAST_INDEX_SECRET=
WEB_SEARCH_API_KEY=

# ---- Email digest (optional) --------------------------------------------
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_TO=
SMTP_FROM=
SMTP_USE_TLS=true

# ---- Layout -------------------------------------------------------------
CONFIG_DIR=config
DATA_DIR=data
BRAIN_DIR=brain
BRIEFING_DIR=briefing   # private: index.html, latest.md, latest.json
PUBLIC_DIR=public       # published: *.xml only
PAGES_BASE_URL=

# ---- Publishing ---------------------------------------------------------
# Credential the publish step uses to push feeds to the PUBLIC repo. Set ONE,
# as an Actions secret — never in this file. If both are set the deploy key
# wins.
#
#   PAGES_DEPLOY_KEY  SSH deploy key, private half. Preferred: bound to that
#                     one repository, never expires. Needs a local ssh-keygen.
#   PAGES_PUSH_TOKEN  Fine-grained PAT, Contents: read and write on the public
#                     repo. Created entirely in the browser, but carries
#                     account-wide identity and expires.
PAGES_DEPLOY_KEY=
PAGES_PUSH_TOKEN=
"""

FEEDS_INDEX_HTML = """\
<!doctype html>
<meta charset="utf-8">
<title>Podcast Scout — Feeds</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{font:16px/1.6 system-ui,sans-serif;max-width:34rem;margin:4rem auto;padding:0 1.5rem;color:#1a1a1a}
  h1{font-size:1.4rem;margin:0 0 .5rem}
  p{color:#555}
  li{margin:.4rem 0}
  code{background:#f4f4f5;padding:.15rem .35rem;border-radius:3px;font-size:.9em}
</style>
<h1>Podcast Scout</h1>
<p>AI-ranked podcast queues. Subscribe to a feed in any podcast app.</p>
<ul>
  <li><a href="ai-retail.xml">AI &amp; Retail</a> — <code>ai-retail.xml</code></li>
  <li><a href="startup.xml">Startup &amp; Strategy</a> — <code>startup.xml</code></li>
  <li><a href="personal-growth.xml">Personal Growth</a> — <code>personal-growth.xml</code></li>
  <li><a href="all.xml">Everything</a> — <code>all.xml</code></li>
</ul>
<p style="font-size:.85em;color:#888">Curated automatically. The briefing itself is private.</p>
"""

README = """\
# {instance_name} — private instance

Private half of the split. The engine lives in [{public_repo}]({public_url})
and is installed as a dependency; this repo holds everything personal.

```
config/     preferences, show priors, watchlists   (personal)
data/       state, history, feedback               (personal)
brain/      theses, sources, patterns, companies   (the durable asset)
briefing/   index.html, latest.md, latest.json     (private output)
public/     *.xml only — pushed to the public repo (generated, gitignored)
```

## Why the split

`config/` and `data/` say what you read, who you follow, and what you track.
`brain/theses/` holds beliefs you are willing to defend. None of that belongs
in a public repo. The engine — ranking, feeds, delivery — has nothing personal
in it and stays open.

## The brain

```bash
podcast-scout brain init      # scaffold + seed draft theses
podcast-scout brain status    # active theses, confidence, falsifier coverage
```

Each thesis carries a **falsifier**: the evidence that would change your mind.
Every run tests the day's signals against them and leads the brief with
anything that *cuts against* what you believe. Ranking optimises for relevance,
so the queue agrees with you by construction; the falsifier watch is the only
part that pushes back.

Seeded theses ship as drafts with `confidence: Low` and TODOs. Rewrite the
statement and the falsifier in your own words — a thesis you did not write is
not a belief, and the watch is only as honest as the falsifier behind it.

## Local run

```bash
uv venv && uv pip install "podcast-scout @ git+{public_url}@{ref}"
cp .env.example .env      # fill in at least GEMINI_API_KEY
podcast-scout validate
podcast-scout run --dry-run
```
"""

DAILY_WORKFLOW = """\
name: Daily Podcast Scout

on:
  schedule:
    - cron: '0 5 * * 1-5'    # Mon-Fri 05:00 UTC
  workflow_dispatch:
    inputs:
      lookback_days:
        description: 'Override lookback window (days)'
        required: false
        default: ''
      dry_run:
        description: 'Dry run (no writes/email)'
        required: false
        type: boolean
        default: false
      synthesis:
        description: 'Force the weekly cross-episode synthesis'
        required: false
        type: boolean
        default: false

permissions:
  contents: write

jobs:
  scout:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh && echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Install engine
        # Pinned to a tag, not a branch: an engine change must never alter a
        # scheduled run without an explicit bump here.
        run: uv pip install --system "podcast-scout @ git+{public_url}@{ref}"

      - name: Run scout
        env:
          CONFIG_DIR: config
          DATA_DIR: data
          BRAIN_DIR: brain
          BRIEFING_DIR: briefing
          PUBLIC_DIR: public
          GEMINI_API_KEY: ${{{{ secrets.GEMINI_API_KEY }}}}
          LLM_FALLBACK_API_KEY: ${{{{ secrets.LLM_FALLBACK_API_KEY }}}}
          # Forwarded under several spellings: a secret only reaches the run
          # under the exact name the workflow names, and an unset one is just
          # an empty string.
          OPENROUTER_API_KEY: ${{{{ secrets.OPENROUTER_API_KEY }}}}
          OPEN_ROUTER_API_KEY: ${{{{ secrets.OPEN_ROUTER_API_KEY }}}}
          OPENROUTER_KEY: ${{{{ secrets.OPENROUTER_KEY }}}}
          OPENROUTER_API: ${{{{ secrets.OPENROUTER_API }}}}
          OPEN_ROUTER_API: ${{{{ secrets.OPEN_ROUTER_API }}}}
          OPEN_ROUTER_KEY: ${{{{ secrets.OPEN_ROUTER_KEY }}}}
          OPEN_ROUTER: ${{{{ secrets.OPEN_ROUTER }}}}
          OPENROUTER: ${{{{ secrets.OPENROUTER }}}}
          # Model selection lives in repository VARIABLES, not secrets: a model
          # name is not a credential, and keeping it visible and editable
          # matters because Gemini free-tier quota is per-model -- switching
          # models is the first thing to try on a 429. Unset renders as an
          # empty string, which the engine treats as "use the default".
          GEMINI_STAGE1_MODEL: ${{{{ vars.GEMINI_STAGE1_MODEL }}}}
          GEMINI_STAGE2_MODEL: ${{{{ vars.GEMINI_STAGE2_MODEL }}}}
          LLM_FALLBACK_BASE_URL: ${{{{ vars.LLM_FALLBACK_BASE_URL }}}}
          LLM_FALLBACK_MODEL: ${{{{ vars.LLM_FALLBACK_MODEL }}}}
          # Episodes per Stage 2 request. The main lever on request count,
          # which is what free-tier Gemini limits (20/minute).
          STAGE2_BATCH_SIZE: ${{{{ vars.STAGE2_BATCH_SIZE }}}}
          # A contact address the SEC requires in the User-Agent, not a
          # credential -- so a repository VARIABLE, visible and editable.
          # Unset disables the earnings adapter rather than 403-looping.
          SEC_USER_AGENT: ${{{{ vars.SEC_USER_AGENT }}}}
          PODCAST_INDEX_KEY: ${{{{ secrets.PODCAST_INDEX_KEY }}}}
          PODCAST_INDEX_SECRET: ${{{{ secrets.PODCAST_INDEX_SECRET }}}}
          WEB_SEARCH_API_KEY: ${{{{ secrets.WEB_SEARCH_API_KEY }}}}
          SMTP_HOST: ${{{{ secrets.SMTP_HOST }}}}
          SMTP_PORT: ${{{{ secrets.SMTP_PORT }}}}
          SMTP_USER: ${{{{ secrets.SMTP_USER }}}}
          SMTP_PASSWORD: ${{{{ secrets.SMTP_PASSWORD }}}}
          SMTP_TO: ${{{{ secrets.SMTP_TO }}}}
          SMTP_FROM: ${{{{ secrets.SMTP_FROM }}}}
          SMTP_USE_TLS: ${{{{ secrets.SMTP_USE_TLS }}}}
          PAGES_BASE_URL: ${{{{ secrets.PAGES_BASE_URL }}}}
        run: |
          ARGS=""
          if [ -n "${{{{ inputs.lookback_days }}}}" ]; then ARGS="$ARGS --lookback ${{{{ inputs.lookback_days }}}}"; fi
          if [ "${{{{ inputs.dry_run }}}}" = "true" ]; then ARGS="$ARGS --dry-run"; fi
          # Friday's run also produces the weekly cross-episode synthesis.
          # Derived from the date rather than matched against the cron string,
          # so editing the schedule above cannot silently drop the synthesis.
          # The event_name guard keeps a manual Friday run from getting it
          # unasked; --synthesis is an explicit dispatch input for that.
          if [ "${{{{ github.event_name }}}}" = "schedule" ] && [ "$(date -u +%u)" = "5" ]; then
            ARGS="$ARGS --synthesis"
          elif [ "${{{{ inputs.synthesis }}}}" = "true" ]; then
            ARGS="$ARGS --synthesis"
          fi
          podcast-scout run $ARGS

      - name: Commit state, briefing and brain
        if: ${{{{ inputs.dry_run != true }}}}
        run: |
          git config user.name "podcast-scout[bot]"
          git config user.email "podcast-scout[bot]@users.noreply.github.com"
          git add data briefing brain
          git diff --cached --quiet || git commit -m "chore: daily run [skip ci]"
          git pull --rebase origin "${{{{ github.ref_name }}}}"
          git push origin "HEAD:${{{{ github.ref_name }}}}"

      - name: Publish feeds to the public repo
        if: ${{{{ inputs.dry_run != true }}}}
        env:
          PAGES_DEPLOY_KEY: ${{{{ secrets.PAGES_DEPLOY_KEY }}}}
          PAGES_PUSH_TOKEN: ${{{{ secrets.PAGES_PUSH_TOKEN }}}}
        run: |
          # Only *.xml crosses the boundary. The briefing (scores, rejected
          # items, synthesis) describes how the reader thinks and stays here.
          if ! ls public/*.xml >/dev/null 2>&1; then
            echo "No feeds generated this run — nothing to publish."; exit 0
          fi
          # Either credential works. A deploy key is preferred: it is bound to
          # the one repository and never expires. A PAT is the fallback because
          # it can be created entirely in the browser, with no local keygen.
          if [ -n "$PAGES_DEPLOY_KEY" ]; then
            echo "Publishing over SSH (deploy key)."
            mkdir -p ~/.ssh
            # printf '%s\\n' guarantees the trailing newline. An SSH private key
            # without one is rejected outright as "invalid format", and pasting
            # a key into a GitHub secret is the usual way to lose it.
            printf '%s\\n' "$PAGES_DEPLOY_KEY" > ~/.ssh/pages_key
            chmod 600 ~/.ssh/pages_key  # ssh refuses a group/world-readable key
            ssh-keyscan -t rsa,ecdsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
            # IdentitiesOnly stops ssh offering any other key it finds.
            export GIT_SSH_COMMAND="ssh -i ~/.ssh/pages_key -o IdentitiesOnly=yes"
            REMOTE="git@github.com:{public_repo}.git"
          elif [ -n "$PAGES_PUSH_TOKEN" ]; then
            echo "Publishing over HTTPS (token). Actions masks the token in logs."
            REMOTE="https://x-access-token:$PAGES_PUSH_TOKEN@github.com/{public_repo}.git"
          else
            echo "Neither PAGES_DEPLOY_KEY nor PAGES_PUSH_TOKEN set — skipping publish."
            exit 0
          fi
          git clone --depth 1 --branch gh-pages "$REMOTE" pages \\
            || git clone --depth 1 "$REMOTE" pages
          cd pages
          if git checkout gh-pages 2>/dev/null; then
            echo "Publishing onto the existing gh-pages branch."
          else
            # Creating gh-pages: the clone above fetched the DEFAULT branch, and
            # `checkout --orphan` keeps that working tree, so without this the
            # whole repo -- README, workflows, and any stale briefing files
            # still committed under public/ -- lands on gh-pages alongside the
            # feeds. Clear it so the branch holds only what is published.
            git checkout --orphan gh-pages
            git rm -rf --cached . >/dev/null 2>&1 || true
            find . -maxdepth 1 ! -name . ! -name .git -exec rm -rf {{}} +
          fi
          # Never `rm -rf *` here: a run that generated no feeds would wipe
          # every live feed URL. Copy over the top instead.
          cp ../public/*.xml .
          [ -f ../public/index.html ] && cp ../public/index.html .
          touch .nojekyll
          git config user.name "podcast-scout[bot]"
          git config user.email "podcast-scout[bot]@users.noreply.github.com"
          git add -A
          git diff --cached --quiet || git commit -m "chore: publish feeds [skip ci]"
          git push origin gh-pages
"""

LLM_DOCTOR_WORKFLOW = """\
name: LLM Doctor

# Manual only. Answers "what can this key actually call?" by asking the API
# rather than reading the docs: docs say which models exist, not which ones
# this project can reach or has quota for.
on:
  workflow_dispatch:
    inputs:
      probe:
        description: 'Send a minimal request to each Flash model'
        required: false
        type: boolean
        default: true

permissions:
  contents: read

jobs:
  doctor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh && echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Install engine
        run: uv pip install --system "podcast-scout @ git+{public_url}@{ref}"

      - name: Ask the API what this key can use
        env:
          GEMINI_API_KEY: ${{{{ secrets.GEMINI_API_KEY }}}}
          GEMINI_STAGE2_MODEL: ${{{{ vars.GEMINI_STAGE2_MODEL }}}}
          # The fallback is half of what this command checks; without these it
          # silently reports "no fallback configured" no matter what is set.
          LLM_FALLBACK_API_KEY: ${{{{ secrets.LLM_FALLBACK_API_KEY }}}}
          # Forwarded under several spellings: a secret only reaches the run
          # under the exact name the workflow names, and an unset one is just
          # an empty string.
          OPENROUTER_API_KEY: ${{{{ secrets.OPENROUTER_API_KEY }}}}
          OPEN_ROUTER_API_KEY: ${{{{ secrets.OPEN_ROUTER_API_KEY }}}}
          OPENROUTER_KEY: ${{{{ secrets.OPENROUTER_KEY }}}}
          OPENROUTER_API: ${{{{ secrets.OPENROUTER_API }}}}
          OPEN_ROUTER_API: ${{{{ secrets.OPEN_ROUTER_API }}}}
          OPEN_ROUTER_KEY: ${{{{ secrets.OPEN_ROUTER_KEY }}}}
          OPEN_ROUTER: ${{{{ secrets.OPEN_ROUTER }}}}
          OPENROUTER: ${{{{ secrets.OPENROUTER }}}}
          LLM_FALLBACK_BASE_URL: ${{{{ vars.LLM_FALLBACK_BASE_URL }}}}
          LLM_FALLBACK_MODEL: ${{{{ vars.LLM_FALLBACK_MODEL }}}}
        run: |
          if [ "${{{{ inputs.probe }}}}" = "false" ]; then
            podcast-scout llm-doctor --no-probe
          else
            podcast-scout llm-doctor
          fi
"""


MONTHLY_REVIEW_WORKFLOW = """\
name: Monthly State of My Thinking

# The daily brief says what arrived and Friday's synthesis says what the week
# added up to. Neither asks whether anything actually changed in what you
# believe -- that needs the accumulated brain, not one window of the feed, so
# it runs on its own cadence over its own inputs.
on:
  schedule:
    - cron: '0 6 1 * *'    # 06:00 UTC on the 1st, after that day's brief
  workflow_dispatch:
    inputs:
      lookback_days:
        description: 'Window to review (days)'
        required: false
        default: '30'
      dry_run:
        description: 'Print the review without committing it'
        required: false
        type: boolean
        default: false

permissions:
  contents: write

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh && echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Install engine
        run: uv pip install --system "podcast-scout @ git+{public_url}@{ref}"

      - name: Review the brain
        env:
          CONFIG_DIR: config
          DATA_DIR: data
          BRAIN_DIR: brain
          GEMINI_API_KEY: ${{{{ secrets.GEMINI_API_KEY }}}}
          LLM_FALLBACK_API_KEY: ${{{{ secrets.LLM_FALLBACK_API_KEY }}}}
          # Forwarded under several spellings: a secret only reaches the run
          # under the exact name the workflow names, and an unset one is just
          # an empty string.
          OPENROUTER_API_KEY: ${{{{ secrets.OPENROUTER_API_KEY }}}}
          OPEN_ROUTER_API_KEY: ${{{{ secrets.OPEN_ROUTER_API_KEY }}}}
          OPENROUTER_KEY: ${{{{ secrets.OPENROUTER_KEY }}}}
          OPENROUTER_API: ${{{{ secrets.OPENROUTER_API }}}}
          OPEN_ROUTER_API: ${{{{ secrets.OPEN_ROUTER_API }}}}
          OPEN_ROUTER_KEY: ${{{{ secrets.OPEN_ROUTER_KEY }}}}
          OPEN_ROUTER: ${{{{ secrets.OPEN_ROUTER }}}}
          OPENROUTER: ${{{{ secrets.OPENROUTER }}}}
          GEMINI_STAGE1_MODEL: ${{{{ vars.GEMINI_STAGE1_MODEL }}}}
          GEMINI_STAGE2_MODEL: ${{{{ vars.GEMINI_STAGE2_MODEL }}}}
          LLM_FALLBACK_BASE_URL: ${{{{ vars.LLM_FALLBACK_BASE_URL }}}}
          LLM_FALLBACK_MODEL: ${{{{ vars.LLM_FALLBACK_MODEL }}}}
        run: |
          ARGS="--lookback ${{{{ inputs.lookback_days || '30' }}}}"
          if [ "${{{{ inputs.dry_run }}}}" = "true" ]; then ARGS="$ARGS --dry-run"; fi
          podcast-scout brain review $ARGS

      - name: Commit the review
        if: ${{{{ inputs.dry_run != true }}}}
        run: |
          git config user.name "podcast-scout[bot]"
          git config user.email "podcast-scout[bot]@users.noreply.github.com"
          git add brain
          git diff --cached --quiet || git commit -m "chore: monthly review [skip ci]"
          git pull --rebase origin "${{{{ github.ref_name }}}}"
          git push origin "HEAD:${{{{ github.ref_name }}}}"
"""


def copy_path(src: Path, dst: Path) -> str:
    if not src.exists():
        return f"  skip (absent)  {src.relative_to(REPO_ROOT)}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        count = sum(1 for _ in src.rglob("*") if _.is_file())
        return f"  copied {count:>4} files  {src.relative_to(REPO_ROOT)}/"
    shutil.copy2(src, dst)
    return f"  copied          {src.relative_to(REPO_ROOT)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="Directory for the new private instance repo")
    parser.add_argument(
        "--public-repo",
        default="iitiff/podcast-curation-agent",
        help="owner/name of the public engine repo",
    )
    parser.add_argument("--ref", default="v1", help="Engine tag to pin the instance to")
    args = parser.parse_args()

    target: Path = args.target.expanduser().resolve()
    public_url = f"https://github.com/{args.public_repo}"

    if target.exists() and any(target.iterdir()):
        print(f"Refusing to write into non-empty directory: {target}", file=sys.stderr)
        return 1

    print(f"Building private instance at {target}\n")
    target.mkdir(parents=True, exist_ok=True)

    for src_rel, dst_rel in PERSONAL_PATHS:
        print(copy_path(REPO_ROOT / src_rel, target / dst_rel))

    fmt = {
        "public_repo": args.public_repo,
        "public_url": public_url,
        "ref": args.ref,
        # Derived from the target directory so the instance README carries
        # the repo's real name rather than a hardcoded one.
        "instance_name": target.name,
    }
    (target / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (target / ".env.example").write_text(ENV_EXAMPLE, encoding="utf-8")
    (target / "README.md").write_text(README.format(**fmt), encoding="utf-8")

    workflows = target / ".github" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    (workflows / "daily.yml").write_text(DAILY_WORKFLOW.format(**fmt), encoding="utf-8")
    (workflows / "llm-doctor.yml").write_text(LLM_DOCTOR_WORKFLOW.format(**fmt), encoding="utf-8")
    (workflows / "monthly-review.yml").write_text(
        MONTHLY_REVIEW_WORKFLOW.format(**fmt), encoding="utf-8"
    )

    static = target / "static"
    static.mkdir(exist_ok=True)
    (static / "index.html").write_text(FEEDS_INDEX_HTML, encoding="utf-8")

    (target / "briefing").mkdir(exist_ok=True)
    (target / "briefing" / ".gitkeep").touch()

    # Scaffold the brain in-process so the instance arrives with seeded theses.
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from podcast_scout.brain import BrainStore  # noqa: PLC0415

    BrainStore(target / "brain").ensure_dirs()

    print("\n  wrote           .gitignore, .env.example, README.md")
    print("  wrote           .github/workflows/daily.yml")
    print("  wrote           .github/workflows/monthly-review.yml")
    print("  wrote           static/index.html (public feed listing)")
    print("  scaffolded      brain/\n")
    print("Next:")
    print(f"  cd {target}")
    print("  BRAIN_DIR=brain podcast-scout brain init   # seed draft theses")
    print("  git init && git add -A && git commit -m 'Initial private instance'")
    print("  # create a PRIVATE repo, then: git remote add origin ... && git push -u origin main")
    print("\nThen follow MIGRATION.md in the engine repo for the cutover.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
