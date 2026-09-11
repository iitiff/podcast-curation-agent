# Roadmap

The plan lived only in a chat transcript until now, which is why nobody could
answer "what was in phase 3". This is the durable version.

## The architecture, in one distinction

**Filter — evergreen.** Persona, show priors, watchlists, the two-stage rubric.
Answers *"is this worth my time, given who I am?"* Tuned over weeks against
real output; it changes slowly, if ever. It is the quality bar.

**Radar — question-driven.** Points at whatever is currently being figured out.
Answers *"where might the thing I'm trying to understand be discussed?"*
Changes whenever the questions change.

Two lanes, one bar:

```
QUESTIONS ──► generated queries ──► candidates ──┐
                                                 ├─► FILTER ──► ranked queue
FOLLOWED SHOWS ──► RSS ─────────────────────────┘        │
                                                          └─► WATCH ──► brief
```

Everything is judged by the same evergreen rubric regardless of which lane it
arrived through. The radar decides what gets *looked at*; the filter decides
what is worth *reading*. Conflating the two is what made the radar podcast-only:
the evergreen seeds are all phrased "... podcast", so no query could ever
surface a paper.

## Why this matters more than it sounds

An open question like *"What comes after next-best-action?"* is answered by a
RecSys paper or an engineering blog far better than by any podcast. The source
mix should follow the question, not the other way round. A question tagged
`architecture` should reach for papers; one tagged `competitive` should reach
for earnings calls.

---

## Done

- **Phase 0 — Engine/instance split.** Public engine, private instance holding
  config, state and brain. Engine pip-installable, pinned by commit.
- **Phase 1 — Brain.** Markdown pages for Source, Thesis, Question, Pattern,
  Company. Falsifier watch for theses; relevance watch for questions.
- **Unplanned but load-bearing.** CI (did not exist); nine LLM resilience
  fixes; `llm-doctor`; the degradation guard that stops a failed run from
  freezing the feed.
- **Phase 3 — Question-driven, multi-source radar.** Questions generate
  discovery queries alongside the evergreen seeds; source classes
  (`earnings-call`, `research-paper`, `trade-press`, `vendor`) each carry
  their own credibility and bias notes; the question's tags decide which
  classes are reached for.
- **Three cadences.** Daily brief, weekly synthesis on Fridays, monthly
  *State of My Thinking* over the brain (`podcast-scout brain review`).
- **Separate reading and listening budgets.** Non-podcast items get their own
  `max_reading` cap and their own Stage-2 allowance. Previously they consumed
  `max_listen_fully` slots they could never fill, and `rss.py` then dropped
  them for want of an enclosure — publishing an empty feed.

- **Earnings via SEC EDGAR.** The highest-value class. IR sites publish no
  feed, but every US-listed company files its results as an 8-K tagged item
  2.02 with the numbers in EX-99.1, so EDGAR is the feed they do not provide.
- **Rubric generalised off podcast assumptions.** Written sources declare
  their class and credibility instead of an empty guest list and a zero
  duration, and the confidence penalty no longer fires on a paper for having
  no transcript.

## Next

1. **Question-aware weekly synthesis** (Phase 2, below) — the last cadence
   still regenerating opinions from scratch.
2. **Corroboration links.** Earnings can now contradict a vendor claim, but
   nothing yet *pairs* them. The Source pages exist; what is missing is the
   pass that says "this filing contradicts that vendor post".

Earnings before vendor blogs. A vendor blog asserting that agentic commerce is
inevitable is a pattern source; a retailer's earnings call reporting that agent
traffic did not convert is evidence. The design docs are explicit that vendor
claims require corroboration, and earnings is the only class that provides it.

## Sources

| Class | Trust | Reached via |
| --- | --- | --- |
| `earnings-call` | high | SEC EDGAR, 8-K item 2.02 → EX-99.1 |
| `research-paper` | high | arXiv and other research RSS |
| `trade-press` | medium | publication RSS |
| `vendor` | low | vendor blog RSS, off by default |

EDGAR needs `SEC_USER_AGENT` set to a contact address — the SEC refuses
undeclared requests and answers with a throttle page rather than an error, so
the adapter disables itself rather than guessing one.

## Cadences

| When | What | Command |
| --- | --- | --- |
| Mon–Fri | Daily brief and feeds | `podcast-scout run` |
| Fri | Adds the weekly cross-episode synthesis | `podcast-scout run --synthesis` |
| 1st of the month | *State of My Thinking* over the brain | `podcast-scout brain review` |

The monthly review is the only one that reads the brain rather than a window
of the feed. The weekly says what the week added up to; the monthly asks
whether anything actually changed in what you believe, which needs the
accumulated findings under each open question.

## Later

- **Phase 2 — Question-aware synthesis.** The weekly synthesis still
  regenerates opinions from scratch; it should read the open questions the way
  the monthly review does, and propose updates as pull requests so a person
  still decides.
- **Phase 4 — Judgment gym.** `executive-challenge`,
  `product-strategy-review`, `L7-case-interview` as prompt assets. Cheap, no
  infrastructure.

## Deliberately not doing

- **A routing framework.** LiteLLM solves multi-provider routing properly at
  scale, but every failure here has been a dead credential, model or endpoint
  rather than routing logic.
- **A vector database.** Retrieval is ripgrep plus a generated index until
  that genuinely stops working.
- **More entity types in anticipation.** Add one when a real page will not fit
  in an existing type.
- **Making theses mandatory.** The watch costs nothing when no question or
  thesis is defined. A thesis nobody wrote is not a belief, and an
  unmaintained one accumulates supporting evidence until it looks like rigor.
