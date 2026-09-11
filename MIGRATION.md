# Migration: splitting the engine from the instance

This repo is becoming the **public engine**: code, example configs, tests, and
nothing personal. Your preferences, state, history, and brain move to a new
**private instance repo** that installs the engine as a dependency.

## Why

`config/preferences.yaml` and `data/` currently say which shows you follow,
which companies you track, which people you watch for, and everything you have
already read — in a public, MIT-licensed repo. `brain/theses/` will hold
beliefs you are willing to defend. None of that should be public.

## Order matters

The daily pipeline is live. **Nothing in this repo is deleted until the private
instance has produced one green run.** Do the steps in order; step 7 is the
only destructive one.

---

### 1. Build the instance

```bash
python scripts/make_instance.py ../product-leadership-brain
cd ../product-leadership-brain
```

Copies `config/`, `data/state.json`, `data/history/`, `data/feedback.csv`, then
writes a `.gitignore`, `.env.example`, `README.md`, and a workflow that installs
the engine and publishes only `*.xml`. It reads this repo and never modifies it.

### 2. Seed the brain

```bash
BRAIN_DIR=brain podcast-scout brain init
BRAIN_DIR=brain podcast-scout brain status
```

Four draft theses are written at `confidence: Low` with TODOs. **Rewrite the
statement and the falsifier of each in your own words before the first real
run.** A thesis you did not write is not a belief, and the falsifier watch only
flags what its falsifier describes. `brain status` marks any thesis with no
falsifier in red — those can only ever accumulate confirmation.

### 3. Create the private repo

```bash
git init && git add -A && git commit -m "Initial private instance"
# create a PRIVATE repo on GitHub, then:
git remote add origin git@github.com:<you>/product-leadership-brain.git
git push -u origin main
```

Confirm it is **Private** before pushing — this commit contains your full
reading history.

### 4. Add secrets to the private repo

Copy from this repo's existing Actions secrets:
`GEMINI_API_KEY`, `LLM_FALLBACK_API_KEY`, `PODCAST_INDEX_KEY`,
`PODCAST_INDEX_SECRET`, `WEB_SEARCH_API_KEY`, `SMTP_*`, `PAGES_BASE_URL`.

Then add one new secret, `PAGES_PUSH_TOKEN`: a fine-grained PAT scoped to
**only this public repo** with `Contents: read and write`. It lets the private
run push feeds here. Nothing else needs it.

### 5. Point GitHub Pages at a branch

The private repo pushes feeds to this repo's `gh-pages` branch instead of
uploading a Pages artifact.

**Settings → Pages → Source: Deploy from a branch → `gh-pages` / root.**

Feed URLs do not change (`…github.io/podcast-curation-agent/ai-retail.xml`), so
FeedBurner and existing subscribers are unaffected.

### 6. Verify a green run

Run the private repo's workflow manually (`workflow_dispatch`), first with
**dry run = true**, then for real. Confirm:

- [ ] the run completes green
- [ ] `public/*.xml` was pushed to this repo's `gh-pages`
- [ ] the live feed URLs still resolve and show recent episodes
- [ ] the email digest arrived
- [ ] `data/state.json` and `brain/` were committed in the private repo
- [ ] `briefing/index.html` exists **in the private repo only**

### 7. Cut over this repo (destructive — only after step 6)

```bash
git rm -r --cached config/preferences.yaml config/shows.yaml \
                   config/discovery_queries.yaml data/
rm -rf data/ config/preferences.yaml config/shows.yaml config/discovery_queries.yaml
git rm .github/workflows/daily.yml .github/workflows/weekly_synthesis.yml
git commit -m "Split: move personal config, state and scheduling to private instance"
```

The example configs (`config/*.example.yaml`) stay, so the engine remains
runnable by anyone who clones it.

> Delete the workflows **in the same commit** as the config. A scheduled run
> that survives without `config/preferences.yaml` fails every weekday morning.

### 8. What this does not do

Removing these files changes `HEAD` only. **Your config and history remain in
this repo's git history** — 200+ public commits going back to 2026-07-25.
Treat anything already committed as public. Scrubbing it would mean rewriting
history (`git filter-repo`, force-push, breaking the existing fork) or
recreating the repo, both of which you decided against.

The practical consequence: rotate nothing (no credentials were committed), but
assume the show list, watchlists, and reading history are already indexed.

---

## After the split

| | engine (public) | instance (private) |
|---|---|---|
| code, tests, examples | ✅ | — |
| preferences, shows, queries | example only | ✅ real |
| state, history, feedback | — | ✅ |
| brain: theses, sources | — | ✅ |
| briefing HTML / latest.json | — | ✅ |
| published `*.xml` | `gh-pages` | generated, pushed |
| scheduled workflows | — | ✅ |

Bump the engine by moving the tag the instance pins (`@v1` in its workflow).
A change here never alters a scheduled run until you bump it deliberately.
