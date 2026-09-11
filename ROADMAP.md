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

## Next — Phase 3: question-driven, multi-source radar

Reordered ahead of Phase 2 deliberately. Synthesis over a podcast-only corpus
is synthesis over the wrong corpus.

1. **Questions generate discovery queries**, alongside the evergreen seeds
   rather than replacing them.
2. **Source adapters** — trade press RSS, vendor blogs, research feeds,
   earnings transcripts — each carrying its own `credibility` and
   `bias_notes`.
3. **Question-aware source selection** — the question's tags decide which
   adapters are reached for.
4. **Generalise off podcast assumptions** — `source_type` is currently a
   hardcoded constant; the rubric still scores `learning_per_minute` and
   `listen_nuance`.

Earnings before vendor blogs. A vendor blog asserting that agentic commerce is
inevitable is a pattern source; a retailer's earnings call reporting that agent
traffic did not convert is evidence. The design docs are explicit that vendor
claims require corroboration, and earnings is the only class that provides it.

## Later

- **Phase 2 — Question-aware synthesis.** Weekly synthesis reads open
  questions instead of regenerating opinions from scratch; proposes updates as
  pull requests so a person still decides.
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
