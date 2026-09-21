"""Two-stage episode ranking engine."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any

from pydantic import BaseModel, Field

from .config import Preferences
from .normalize import NormalizedEpisode, clean_snippet
from .providers.base import (
    BaseLLMProvider,
    LLMMessage,
    TranscriptResult,
    describe_exception,
)

log = logging.getLogger(__name__)


class RubricScore(BaseModel):
    relevance: float = 0        # 0-30
    novelty: float = 0          # 0-15
    guest_authority: float = 0  # 0-15
    actionability: float = 0    # 0-15
    evidence: float = 0         # 0-10
    strategic_importance: float = 0  # 0-10
    learning_per_minute: float = 0   # 0-5
    # Penalties (negative values)
    repetition_penalty: float = 0    # up to -15
    generic_penalty: float = 0       # up to -15
    weak_evidence_penalty: float = 0 # up to -10
    confidence_penalty: float = 0    # up to -15
    motivational_penalty: float = 0  # up to -10
    relevance_penalty: float = 0     # up to -20

    @property
    def total(self) -> float:
        base = (
            self.relevance + self.novelty + self.guest_authority
            + self.actionability + self.evidence
            + self.strategic_importance + self.learning_per_minute
        )
        penalties = (
            self.repetition_penalty + self.generic_penalty
            + self.weak_evidence_penalty + self.confidence_penalty
            + self.motivational_penalty + self.relevance_penalty
        )
        return max(0.0, min(100.0, base + penalties))


class Stage1Result(BaseModel):
    guid: str
    score: float
    reason: str
    should_deep_process: bool
    # Best-guess lane from metadata alone, used only to reserve Stage 2 slots.
    # Stage 2 makes the real routing call with the full text in front of it;
    # this just decides who gets to be looked at. Empty when nothing matched.
    predicted_category: str = ""


class RankedEpisode(BaseModel):
    episode: NormalizedEpisode
    score: float
    rubric: RubricScore
    classification: str  # Listen Fully | Read Summary Only | Skip
    classification_reason: str = ""
    evidence_confidence: str = "low"  # high | medium | low
    summary: str = ""
    key_ideas: list[str] = Field(default_factory=list)
    implications: str = ""
    who_should_listen: str = ""
    summary_captures_value: str = ""
    listen_nuance: str = ""
    # Topical lane chosen by Stage 2. Empty means "no opinion" -- the caller
    # keeps whatever the show-title mapping decided. Only ever a key the
    # caller offered, so an unknown value degrades to the show's own lane.
    assigned_category: str = ""
    transcript_source: str = "none"
    tokens_used: int = 0


class _Stage1LLMOutput(BaseModel):
    score: float
    reason: str
    should_deep_process: bool


class _Stage2LLMOutput(BaseModel):
    rubric: dict[str, float]
    classification: str
    classification_reason: str
    summary: str
    key_ideas: list[str]
    implications: str
    who_should_listen: str
    summary_captures_value: str
    listen_nuance: str
    category: str = ""


def _normalize_apostrophes(s: str) -> str:
    """Normalize curly/fancy apostrophes to straight ASCII so prior lookups
    match regardless of whether the RSS feed uses ’ vs '.
    """
    return unicodedata.normalize("NFKD", s).replace("’", "'").replace("‘", "'")


def _show_prior(show_title: str, prefs: Preferences) -> float:
    title = _normalize_apostrophes(show_title).lower()
    for name, prior in prefs.show_priors.items():
        if _normalize_apostrophes(name).lower() in title:
            return prior
    return 0.5


def _is_followed(show_title: str, prefs: Preferences) -> bool:
    title = _normalize_apostrophes(show_title).lower()
    return any(_normalize_apostrophes(name).lower() in title for name in prefs.show_priors)


def _is_acquired(show_title: str) -> bool:
    return "acquired" in show_title.lower()


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """Robustly parse JSON from LLM output."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        result: dict[str, Any] = json.loads(text)
        return result
    except json.JSONDecodeError:
        pass
    last_brace = text.rfind("}")
    if last_brace != -1:
        try:
            result = json.loads(text[: last_brace + 1])
            return result
        except json.JSONDecodeError:
            pass
    try:
        import json_repair
        repaired: dict[str, Any] = json_repair.loads(text)
        return repaired
    except Exception:
        pass
    raise ValueError(f"Could not parse LLM JSON output (length={len(raw)})")


def _parse_llm_json_array(raw: str) -> list[Any]:
    """Parse a JSON array from LLM output, handling markdown fences."""
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        result: list[Any] = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    try:
        import json_repair
        repaired: list[Any] = json_repair.loads(text)
        if isinstance(repaired, list):
            return repaired
        if isinstance(repaired, dict):
            for v in repaired.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass
    raise ValueError(f"Could not parse LLM JSON array (length={len(raw)})")


# Words too common to distinguish one item from another once the persona's focus
# and the category routing hints are shredded into terms. Kept deliberately short:
# the length floor in `_topic_terms` does most of the filtering.
_STAGE1_STOPWORDS = frozenset({
    "about", "above", "across", "after", "against", "already", "always", "among",
    "another", "anything", "around", "because", "been", "before", "being", "below",
    "better", "between", "beyond", "both", "build", "building", "built", "came",
    "cannot", "come", "coming", "could", "course", "design", "designed", "different",
    "does", "doing", "done", "during", "each", "either", "enough", "especially", "even",
    "every", "everything", "from", "further", "getting", "give", "given", "going",
    "great", "have", "having", "here", "however", "into", "itself", "just", "keep",
    "known", "large", "later", "least", "less", "like", "likely", "made", "make",
    "makes", "making", "many", "matter", "more", "most", "much", "must", "need",
    "needs", "never", "next", "nothing", "often", "once", "only", "other", "others",
    "over", "particular", "perhaps", "place", "rather", "really", "right", "same",
    "seems", "several", "should", "since", "some", "something", "sometimes",
    "somewhere", "still", "such", "take", "taken", "than", "that", "their", "them",
    "then", "there", "these", "they", "thing", "things", "think", "this", "those",
    "though", "three", "through", "time", "together", "toward", "under", "until",
    "upon", "used", "uses", "using", "very", "want", "well", "were", "what", "when",
    "where", "whether", "which", "while", "whole", "will", "with", "within", "without",
    "would", "your",
})

# Terms are derived per (focus, routing hints) pair, not per episode: Stage 1 runs
# once per discovered item and there can be several hundred in a run.
_TOPIC_TERM_CACHE: dict[tuple[str, ...], tuple[tuple[str, ...], tuple[str, ...]]] = {}


def _topic_terms(prefs: Preferences) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (phrases, words) describing what this reader is actually about.

    `phrases` are the subjects named in `persona.focus`, split apart below — a hit
    on one of those is strong evidence. `words` are the distinctive single words
    from that focus plus every category's `routing_hint`, which is already a
    curated topical vocabulary for its lane. A word hit is weak evidence on its
    own, so it is worth a quarter of a phrase hit.
    """
    key = (prefs.persona.focus,) + tuple(
        (c.routing_hint or "") for _, c in sorted(prefs.categories.items())
    )
    cached = _TOPIC_TERM_CACHE.get(key)
    if cached is not None:
        return cached

    # Split on " and " as well as on commas. A focus part like "contextual bandits
    # and off-policy evaluation" is two subjects, and neither ever appears in a
    # title joined by that "and" -- matched only as the whole part, the phrase tier
    # almost never fires and the score collapses to the weak word tier.
    phrases: list[str] = []
    for part in re.split(r"[,;\u2013\u2014]", prefs.persona.focus):
        for sub in re.split(r"\band\b", part):
            cleaned = " ".join(sub.split()).strip().lower()
            if len(cleaned) >= 8:
                phrases.append(cleaned)

    words: set[str] = set()
    hint_text = " ".join((c.routing_hint or "") for _, c in sorted(prefs.categories.items()))
    for token in re.findall(r"[a-z][a-z\-]{5,}", f"{prefs.persona.focus} {hint_text}".lower()):
        if token not in _STAGE1_STOPWORDS:
            words.add(token)

    result = (tuple(phrases), tuple(sorted(words)))
    _TOPIC_TERM_CACHE[key] = result
    return result


def _topic_affinity(text: str, prefs: Preferences) -> tuple[float, int, int]:
    """Points (0-20) for how much of the reader's subject this item's metadata names.

    Stage 1 previously had NO topical signal at all: the score was show prior plus
    guest and competitor name hits. For any source where every item shares one show
    title -- an arXiv feed, an engineering blog -- that made every item score
    identically, so which ones reached Stage 2 was decided by feed order. Observed:
    20 of 32 cs.IR papers were never LLM-scored, including several squarely on the
    reader's declared specialty, while less relevant ones from the same feed were.
    Followed podcasts had the same problem one tier up, all tied at the 50 floor.
    """
    phrases, words = _topic_terms(prefs)
    phrase_hits = sum(1 for ph in phrases if ph in text)
    word_hits = sum(1 for w in words if w in text)
    points = min(20.0, phrase_hits * 4.0 + word_hits * 1.0)
    return points, phrase_hits, word_hits



# Per-lane vocabularies, cached like _topic_terms and for the same reason.
_CATEGORY_TERM_CACHE: dict[tuple[str, ...], dict[str, tuple[str, ...]]] = {}


def _category_terms(prefs: Preferences) -> dict[str, tuple[str, ...]]:
    """Distinctive words per category, taken from its routing hint.

    The hint is written to tell Stage 2 what belongs in the lane, which makes it
    the best lane vocabulary already in the config. Falling back to the feed
    description is deliberate but weak -- subscriber-facing copy rarely
    discriminates ("Curated AI and retail episodes").
    """
    key = tuple(
        f"{name}\x00{cfg.routing_hint or cfg.description}"
        for name, cfg in sorted(prefs.categories.items())
    )
    cached = _CATEGORY_TERM_CACHE.get(key)
    if cached is not None:
        return cached

    per_cat: dict[str, set[str]] = {}
    for name, cfg in prefs.categories.items():
        source = cfg.routing_hint or cfg.description
        per_cat[name] = {
            t for t in re.findall(r"[a-z][a-z\-]{5,}", source.lower())
            if t not in _STAGE1_STOPWORDS
        }

    # A word shared by every lane cannot assign an item to one of them.
    if len(per_cat) > 1:
        shared = set.intersection(*per_cat.values())
        for terms in per_cat.values():
            terms -= shared

    result = {name: tuple(sorted(terms)) for name, terms in per_cat.items()}
    _CATEGORY_TERM_CACHE[key] = result
    return result


def predict_category(text: str, prefs: Preferences) -> str:
    """Guess an item's lane from its metadata, or "" when nothing matches.

    Deliberately crude: measured against Stage 2's own assignments over 124
    routed items it agrees about two thirds of the time, and it does not
    normalise for hint length, so a lane with a long routing hint wins ties of
    substance. That is the right trade here because this only decides who gets
    LOOKED AT, not where anything lands. Recall on the scarce lane is what
    matters (19 of 21 in that sample); a false positive costs one reserved slot,
    which is then spent on a candidate that would very likely have won a global
    slot anyway. Stage 2 still assigns the real category from the full text.
    """
    best_name, best_hits = "", 0
    for name, terms in sorted(_category_terms(prefs).items()):
        hits = sum(1 for t in terms if t in text)
        if hits > best_hits:
            best_name, best_hits = name, hits
    return best_name


# Why an episode carries a metadata-only score. The distinction matters at
# persist time: one of these is a decision, the others are failures.
#
#   STAGE1_ONLY      — never chosen for Stage 2. A deliberate budget decision.
#                      Re-running it tomorrow spends the same budget on the
#                      same losing candidate, so it must NOT be retried.
#   BUDGET_EXHAUSTED — chosen, then the run ran out of tokens before reaching it.
#   BATCH_MISSING    — chosen, called, and the answer never arrived.
#
# The last two are transient: the episode was meant to be scored and was not.
STAGE1_ONLY_REASON = "stage1 only"
BUDGET_EXHAUSTED_REASON = "token budget exhausted"
BATCH_MISSING_REASON = "LLM batch entry missing; metadata fallback"

_TRANSIENT_FALLBACK_REASONS = (BUDGET_EXHAUSTED_REASON, BATCH_MISSING_REASON)


def is_transient_fallback(reason: str) -> bool:
    """True when this score is a floor left by a failure, not by a judgement.

    Substring rather than equality because a carried-over episode has
    " (carried over)" appended to whatever reason it was first given.
    """
    return any(r in reason for r in _TRANSIENT_FALLBACK_REASONS)


def stage1_metadata_score(ep: NormalizedEpisode, prefs: Preferences) -> Stage1Result:
    """Fast metadata-only pre-filter score. No LLM call.

    When running without a Gemini key, this score is the final score.
    Followed shows are guaranteed a minimum score of 50 (Read Summary)
    so they always surface rather than being silently dropped.
    Scoring is otherwise based on show priors, guest watchlist,
    competitor watchlist, duration, topic exclusions, and topic affinity
    against the persona's focus and the category routing hints.
    """
    score = 0.0
    reasons: list[str] = []

    text = f"{ep.show_title} {ep.episode_title} {ep.description}".lower()

    # Show prior boost — primary signal
    prior = _show_prior(ep.show_title, prefs)
    score += prior * 40  # scaled to give 0.95-prior shows a score of 38
    reasons.append(f"show_prior={prior:.2f}")

    # Guest watchlist boost
    for guest in prefs.guest_watchlist:
        if guest.lower() in text:
            score += 15
            reasons.append(f"guest:{guest}")
            break

    # Competitor / entity watchlist boost
    competitor_hits = sum(1 for c in prefs.competitor_watchlist if c.lower() in text)
    score += min(10.0, competitor_hits * 3)

    # Duration penalty
    dur = ep.duration_minutes
    if dur > 0:
        if dur < prefs.length.preferred_min_minutes:
            score -= 5
        elif dur > prefs.length.hard_max_minutes and not _is_acquired(ep.show_title):
            score -= 10

    # Topic exclusion penalty
    for excl in prefs.topic_exclusions:
        if excl.lower() in text:
            score -= 20
            reasons.append(f"excluded_topic:{excl}")

    score = max(0.0, min(100.0, score))

    # Floor: any followed show always surfaces as at least Read Summary
    # so the daily queue is never empty just because no LLM key is set.
    if _is_followed(ep.show_title, prefs) and score < 50:
        score = 50.0
        reasons.append("followed_show_floor")

    # Topic affinity is added AFTER the floor, not before it. Applied before, it
    # would be erased for exactly the items that need it most: a followed show
    # under 50 gets clamped up to 50 whether its topic points were 0 or 20, which
    # is the tie this signal exists to break.
    topic_points, phrase_hits, word_hits = _topic_affinity(text, prefs)
    if topic_points:
        score = min(100.0, score + topic_points)
        reasons.append(f"topic={topic_points:.0f}(p{phrase_hits}/w{word_hits})")

    # Lower threshold: send any followed show or score >= 20 to Stage 2
    # so that shows whose RSS title doesn't perfectly match the prior key
    # still get LLM evaluation rather than being silently dropped.
    should_deep = (
        score >= 20
        or _is_followed(ep.show_title, prefs)
        or any(g.lower() in text for g in prefs.guest_watchlist)
    )

    return Stage1Result(
        guid=ep.guid,
        score=score,
        reason=", ".join(reasons) or "metadata_baseline",
        should_deep_process=should_deep,
        predicted_category=predict_category(text, prefs),
    )


def _classify(score: float, prefs: Preferences) -> str:
    """Assign an episode action from its score and configured thresholds."""
    if score >= prefs.classification.listen_fully_min_score:
        return "Listen Fully"
    if score >= prefs.classification.read_summary_min_score:
        return "Read Summary Only"
    return "Skip"


def _reconcile_classification(
    llm_label: str, score: float, prefs: Preferences
) -> tuple[str, bool]:
    """Let the score decide, except within `boundary_override_max` of a threshold.

    The thresholds are the reader's stated policy. The model's own label was
    previously taken verbatim and `_classify` used only when the field was
    missing, so the policy was advisory: an episode scoring 81 against a
    listen threshold of 75 was filed as "Read Summary Only" because the model
    said so, and every playable feed came out empty.

    `boundary_override_max` has been config since the beginning and was never
    read by anything. This is what it was for: near a boundary the model's
    judgement is worth more than a point of arithmetic, and further away the
    reader's threshold wins. Returns the label and whether an override was
    allowed.
    """
    by_score = _classify(score, prefs)
    if not llm_label or llm_label == by_score:
        return by_score, False
    margin = prefs.classification.boundary_override_max
    thresholds = (
        prefs.classification.listen_fully_min_score,
        prefs.classification.read_summary_min_score,
    )
    if any(abs(score - t) <= margin for t in thresholds):
        return llm_label, True
    return by_score, False


def _uncapped(limit: int) -> bool:
    """A non-positive cap means "no limit", not "nothing".

    Every cap below reads this, so the convention is the same wherever a
    number appears in preferences.yaml. It exists because the natural way to
    say "surface everything that clears the threshold" is to remove the cap,
    and there was previously no way to say that at all.
    """
    return limit <= 0


def build_daily_queue(
    ranked: list[RankedEpisode],
    max_minutes: float = 480.0,
    max_listen_fully: int = 3,
    # The email cap. This used to be `max_read_summary` in the signature and
    # `max_email_only` in the body: the caller passed the former, the body read
    # the latter, and nothing connected them. So the per-category
    # max_read_summary in preferences.yaml was inert and the real limit was
    # this parameter's default, applied once PER CATEGORY -- which is how a run
    # configured for "4 summaries" emitted 34.
    max_read_summary: int = 10,
    max_outside: int = 3,
    max_reading: int = 5,
) -> tuple[list[RankedEpisode], list[RankedEpisode], list[RankedEpisode]]:
    """Split ranked items into the RSS queue, email-only overflow, and reading.

    Returns (rss_queue, email_only, reading).

    Three tracks, because listening and reading are not interchangeable and
    must not be funded from one pot:

    * rss_queue    — playable podcast episodes only. Honours max_listen_fully,
      max_outside, and the listen-time budget (max_minutes).
    * email_only   — podcast episodes worth a summary but not a listen.
    * reading      — everything the radar found that is not a podcast
      (papers, trade press, earnings transcripts), under its own max_reading
      cap.

    Non-playable items used to fall through the "Listen Fully" path. They
    cannot be listened to, and rss.py drops them for want of an <enclosure>,
    so each one silently consumed a listening slot and left the feed short.
    They also competed with podcasts for the email cap, and a radar run
    returning 18 articles could crowd the podcast summaries out entirely.
    """
    rss: list[RankedEpisode] = []
    email_only: list[RankedEpisode] = []
    reading: list[RankedEpisode] = []

    listen_count = 0
    outside_count = 0
    minutes_used = 0.0

    for r in ranked:
        if r.classification == "Skip":
            continue

        # Anything without playable audio is a read, whatever the classifier
        # called it. Its budget is separate in both directions: it can neither
        # take a listening slot nor be pushed out of the brief by one.
        if not getattr(r.episode, "is_playable", True):
            if _uncapped(max_reading) or len(reading) < max_reading:
                reading.append(r)
            continue

        # Read Summary Only → always email, never RSS
        if r.classification == "Read Summary Only":
            if _uncapped(max_read_summary) or len(email_only) < max_read_summary:
                email_only.append(r)
            continue

        # Listen Fully below
        is_outside = getattr(r.episode, "is_outside_feed", False)

        if not _uncapped(max_listen_fully) and listen_count >= max_listen_fully:
            if _uncapped(max_read_summary) or len(email_only) < max_read_summary:
                email_only.append(r)
            continue
        if is_outside and not _uncapped(max_outside) and outside_count >= max_outside:
            if _uncapped(max_read_summary) or len(email_only) < max_read_summary:
                email_only.append(r)
            continue
        if (
            max_minutes > 0
            and minutes_used + r.episode.duration_minutes > max_minutes
            and rss
        ):
            if _uncapped(max_read_summary) or len(email_only) < max_read_summary:
                email_only.append(r)
            continue

        rss.append(r)
        listen_count += 1
        minutes_used += r.episode.duration_minutes
        if is_outside:
            outside_count += 1

    return rss, email_only, reading


# The transcript path and the description path are capped separately, because
# they are not the same kind of text.
#
# A real transcript is the whole point of Stage 2. Clipping it to 2500
# characters left the model ~400-600 words -- an opening ad read and a round of
# introductions -- so summaries and key ideas were being derived from material
# that had not reached the substance yet, negating the transcript fetch.
#
# A description is not a transcript and gets no such room. It was previously
# passed whole, which was harmless while every item was a podcast and
# `description` meant a short RSS blurb -- an earnings exhibit carries
# thousands of words and would silently blow the batch's token budget.
_MAX_SOURCE_TEXT = 40_000
_MAX_DESCRIPTION_TEXT = 2500

# Output-token allowances per item, used to size the completion budget to what
# the batch actually contains. A transcript-backed item yields a fuller rubric,
# a longer summary and more key ideas than a description-only one, so it needs
# more room to come back without truncating the JSON array.
_TOKENS_PER_ITEM_DESCRIPTION = 800
_TOKENS_PER_ITEM_TRANSCRIPT = 1_500


def _batch_token_budget(
    items: list[tuple[NormalizedEpisode, TranscriptResult]],
    default: int = 8_000,
) -> int:
    """Return an output-token budget sized to actual batch content.

    The caller's value is a floor, never a ceiling: this only ever raises the
    budget. It guards the JSON array against being truncated mid-way, which
    drops every unmatched episode to the metadata floor.
    """
    high_confidence_sources = {"publisher", "whisper"}
    total = sum(
        _TOKENS_PER_ITEM_TRANSCRIPT if transcript.source in high_confidence_sources
        else _TOKENS_PER_ITEM_DESCRIPTION
        for _, transcript in items
    )
    return max(default, total)


def _media_guidance() -> str:
    """Extra prompt section, added only when the batch is not all podcasts.

    The rubric's field names are podcast-shaped and stay that way: they are
    persisted in state, history and latest.json, and renaming them would
    silently orphan every carried-over score. What changes is how the model is
    told to read them for a source that has no guest and no runtime. Omitted
    entirely for an all-podcast batch, so the common case pays no tokens for it.
    """
    return """
THIS BATCH MIXES MEDIA. Each item declares its type. The rubric's names are
podcast-shaped for historical reasons; read them like this for written sources:

- guest_authority -> AUTHOR authority. For a paper, the authors and institution
  and whether the work is reproducible. For a filing, it is management speaking
  under legal constraint. For a vendor post, the named engineer behind it — a
  post with no named author is a brochure and scores low.
- learning_per_minute -> signal density per unit of reading, not per minute.
- Do NOT apply confidence_penalty to a written source for "no transcript". The
  text IS the source. Apply it only when the text is genuinely a stub.
- Do NOT penalise a written source for having no guests or no duration.

WEIGH THE SOURCE CLASS, which each item declares with its credibility:
- earnings-call (high): reported figures are the strongest evidence available
  here and the only class that can contradict a vendor's claims. Management's
  framing of those figures is still self-interested — score the numbers, not
  the adjectives.
- research-paper (high): strong on mechanism, weak on whether it survives
  production. A benchmark result is not a deployment.
- trade-press (medium): access-driven. Tends to amplify whoever granted the
  interview and rarely revisits a prediction that missed.
- vendor (low): marketing. Treat every performance claim as unverified unless
  the item itself contains the method or the data. A vendor post can still
  score well as a PATTERN — what the vendor believes the market wants — but
  never as evidence that the thing works.
"""



def _category_routing_block(prefs: Preferences) -> str:
    """Ask Stage 2 which topical lane an item belongs in.

    Category is otherwise derived purely from the show title, which cannot
    express "this particular episode is about X". A show-shaped mapping puts
    every Lenny's episode in one lane whether it is about hiring or about
    ranking architecture. This lets a specialist lane be filled by topic while
    still carrying real audio, which a source-shaped lane (papers, engineering
    blogs) never can.
    """
    if len(prefs.categories) < 2:
        return ""
    lines = []
    for key, cfg in prefs.categories.items():
        hint = cfg.routing_hint or cfg.description or cfg.title
        lines.append(f'  - "{key}": {hint}')
    catalogue = "\n".join(lines)
    return f"""
CATEGORY — which lane does this item belong in?
Return a "category" field on every object, chosen from EXACTLY these keys:
{catalogue}

Judge the ITEM, not the show it came from: a show that usually sits in one lane
can publish an episode that plainly belongs in another, and that is the whole
point of asking. Choose on what the item is substantively about, not on which
lane sounds most flattering. When an item spans two lanes, pick the one whose
description covers the majority of its running time. When genuinely unsure,
return "" and the show's own lane is kept.
"""


def _build_item_block(idx: int, ep: NormalizedEpisode, transcript: TranscriptResult) -> str:
    """One item for the Stage 2 batch prompt.

    Shaped by source type. A paper has no guests and no runtime, and printing
    "GUESTS: unknown / DURATION: 0 min" on it invites the model to apply a
    penalty for something that was never missing. Non-podcast items instead
    declare their class and credibility, which is what should move their score.
    """
    source_text = (
        transcript.text[:_MAX_SOURCE_TEXT]
        if transcript.text
        else f"{ep.episode_title}\n\n{ep.description}"[:_MAX_DESCRIPTION_TEXT]
    )
    if ep.source_type == "podcast":
        return (
            f"--- ITEM {idx} (podcast) ---\n"
            f"SHOW: {ep.show_title}\n"
            f"TITLE: {ep.episode_title}\n"
            f"GUESTS: {', '.join(ep.guests) or 'unknown'}\n"
            f"DURATION: {ep.duration_minutes:.0f} min\n"
            f"TRANSCRIPT CONFIDENCE: {transcript.confidence}\n"
            f"TEXT:\n{source_text}\n"
        )
    bias = f"\nKNOWN BIAS: {ep.bias_notes}" if ep.bias_notes else ""
    return (
        f"--- ITEM {idx} ({ep.source_type}) ---\n"
        f"PUBLICATION: {ep.show_title}\n"
        f"TITLE: {ep.episode_title}\n"
        f"SOURCE CLASS: {ep.source_type} (credibility: {ep.credibility}){bias}\n"
        f"TEXT:\n{source_text}\n"
    )



async def stage2_batch_rank(
    items: list[tuple[NormalizedEpisode, TranscriptResult]],
    prefs: Preferences,
    llm: BaseLLMProvider,
    token_budget: int = 8000,
    persona_emphasis: str = "",
) -> list[RankedEpisode]:
    """Rank multiple episodes in a SINGLE LLM call to conserve API quota."""
    if not items:
        return []

    # Treat the caller-supplied value as a floor; increase it only when needed.
    token_budget = _batch_token_budget(items, default=token_budget)

    persona_ctx = (
        f"You are ranking research material for a {prefs.persona.seniority} "
        f"{prefs.persona.role} whose focus is: {prefs.persona.focus}. "
        f"Preferred depth: {prefs.persona.preferred_depth}."
    )
    if persona_emphasis:
        # Placed after the persona so it can qualify it. The persona sets the
        # standing bar; this says how that bar applies to this lane's subject.
        persona_ctx += f"\n\nFOR THIS BATCH SPECIFICALLY: {persona_emphasis}"
    kinds = {ep.source_type for ep, _ in items}
    mixed_media = kinds != {"podcast"}

    episode_blocks = "\n".join(
        _build_item_block(i, ep, tr) for i, (ep, tr) in enumerate(items)
    )

    system_prompt = f"""{persona_ctx}

You will receive {len(items)} item(s). Score EACH on a 100-point rubric and return a
JSON ARRAY (one object per item, in the same order). Do NOT wrap in markdown fences.
{_media_guidance() if mixed_media else ""}{_category_routing_block(prefs)}
SCORING PHILOSOPHY for this persona:
- The persona's focus list above IS the reward list. Score substance on ANY item in it, and
  do not substitute a narrower idea of what "strategic" content looks like. Business framing
  and technical depth are equally valid routes to a high score: how a ranking objective is
  chosen, calibrated and measured counts exactly as much as market structure analysis.
- Prioritise concrete specifics — named systems, real deployments, numbers, mechanisms,
  first-hand accounts of what broke. A well-defended technical argument is concrete; an
  enthusiastic overview of a relevant topic is not, whatever its subject.
- Penalise heavily: generic communication/soft-skills content (e.g. "how to give feedback",
  "speak with confidence"), motivational fluff, and items that could apply to anyone at any
  level rather than to someone doing this persona's job.
- A SHORT SOFT-SKILLS episode scoring near 79 is almost certainly wrong; those belong below
  60 unless the speaker is a top-tier authority. That is a rule about soft-skills content,
  NOT a rule about length. A dense 20-minute technical episode is a good use of 20 minutes
  and must not be marked down for being short.

RUBRIC (base points):
- relevance: 0-30  (is the topic directly useful to this persona's strategic focus?)
- novelty: 0-15  (does it surface new frameworks, data, or perspectives?)
- guest_authority: 0-15  (does whoever is speaking have DIRECT experience of the thing
  under discussion — they built it, they run it, they decided it, or they researched it
  rigorously? A researcher presenting their own work scores as highly here as an operator
  presenting their own P&L. What scores low is a commentator, a coach, or a host
  summarising work that is not theirs.)
- actionability: 0-15  (does it produce decisions or strategies the listener can act on?)
- evidence: 0-10  (are claims backed by data, case studies, or first-hand experience?)
- strategic_importance: 0-10  (score the HIGHER of two routes.
  (a) SCOPE — does it change how someone at {prefs.persona.seniority} scope allocates people,
      capital or years: multi-org blast radius, multi-year bets, or problems where the
      objective itself is still contested?
  (b) MASTERY — is it a rigorous treatment of a discipline the persona's focus names as their
      own, the kind of material someone builds durable and teachable expertise from?
  Route (b) exists because depth of craft inside a declared specialty IS a multi-year bet for
  this reader; without it, every deep technical item on their own subject would score near
  zero here. A tactic a single team ships next sprint that generalises to nothing still
  scores low on both routes.)
- learning_per_minute: 0-5  (signal density relative to length — of the episode for a
  podcast, of the text for anything else)

PENALTIES (negative):
- repetition_penalty: 0 to -15  (topic covered in recent episodes of same show)
- generic_penalty: 0 to -15  (content applies to anyone, not specifically to this persona)
- weak_evidence_penalty: 0 to -10  (opinion without data or real examples)
- confidence_penalty: 0 to -15  (reserve this for an item whose available text is
  UNUSUALLY thin for its kind -- a two-line blurb, a title with no description. Do NOT
  apply it merely because a transcript is absent: transcripts are off by default, so
  every podcast lacks one, and a penalty every podcast takes is a handicap on audio
  rather than a signal about any particular episode. Does NOT apply to a written source
  either: an article, paper or filing IS its own full text.)
- motivational_penalty: 0 to -10  (inspirational/feel-good without strategic substance)
- relevance_penalty: 0 to -20  (off-topic relative to THIS persona's stated focus above --
  judge against that focus, not against a general "tech content" bar. Depth inside the focus
  is not off-topic: a technical paper or engineering post on a named focus area is squarely
  on-topic and must not be penalised here for being narrow or academic.)

CLASSIFICATION (the labels are historical and apply to written sources too —
"Listen Fully" means "worth the full text", not literally audio):
- "Listen Fully" if total >= 75
- "Read Summary Only" if total >= 50
- "Skip" otherwise

KEY IDEAS — what counts as an insight vs. a topic label:
A "key idea" is NOT a restatement of the episode's topic, title, or theme. It is a specific,
non-obvious claim, framework, number, or contrarian take that a {prefs.persona.seniority}
{prefs.persona.role} — someone who already knows the basics of {prefs.persona.focus} — would
find genuinely new. Ground every key idea in something the TEXT actually says (a claim, a
number, a named example) rather than a category label for what the episode is "about".

  BAD (topic label — do not produce this style):
    "AI agents as both attackers and defenders in cybersecurity."
    "Strategic implications for AI product security and risk management."

  GOOD (specific, sourced, persona-relevant):
    "The guest argues patch-cycle security becomes obsolete once attackers automate
    exploit discovery, forcing a shift to continuous agent-vs-agent defense within 2 years —
    a budget line most CISOs haven't created yet."
    "Cites a case where an AI red-team found a zero-day in 40 minutes that a human pentest
    team missed for 6 months, used to argue AI-assisted offense now outpaces AI-assisted
    defense by default."

Each key_idea must pass this test: could this sentence be copy-pasted onto a DIFFERENT
episode about the same broad topic without becoming false? If yes, it's a topic label, not
an insight — rewrite it or drop it.

If TRANSCRIPT CONFIDENCE is low or the source text is description-only, do not fabricate
specificity that isn't in the text — return fewer key_ideas (even an empty list) rather than
disguising a topic label as an insight.

Low transcript confidence is a reason to claim LESS, not to rate lower twice. Whatever
discount it deserves belongs in the rubric numbers alone; do not then also downgrade the
classification for the same reason. Classify from the total you produced.

For EACH episode return an object with keys:
  item (the integer from that item's "--- ITEM N ---" header; copy it exactly,
    it is how the answer is matched back to the right episode),
  rubric (dict), classification, classification_reason,
  summary (100-200 words — what the episode covers, for orientation),
  key_ideas (list of 0-3 strings — specific, sourced, persona-relevant insights per the
    definition above; this is NOT a compressed restatement of summary),
  implications, who_should_listen (who on the team should read or listen to this),
  summary_captures_value ("yes"|"partial"|"no"),
  listen_nuance (what is lost by reading only the summary — for a written source, what
    the full text carries that a précis cannot)
  category (one of the category keys listed above, or "" if none was listed or you are
    genuinely unsure)

Return ONLY a raw JSON array of {len(items)} objects. No prose, no markdown."""

    user_msg = f"Rank these {len(items)} episode(s):\n\n{episode_blocks}"

    tokens_used = 0
    try:
        resp = await llm.complete(
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_msg),
            ],
            max_tokens=token_budget,
        )
        tokens_used = resp.input_tokens + resp.output_tokens
        entries = _parse_llm_json_array(resp.content)
        # A short array means the response was truncated mid-JSON. Every
        # unmatched episode below silently degrades to the metadata floor
        # (50.0, no summary, no key ideas), so make the cause explicit rather
        # than emitting N separate "batch entry missing" lines that look like
        # N unrelated problems.
        if len(entries) < len(items):
            log.warning(
                "Stage 2 returned %d entries for %d episodes — response was "
                "TRUNCATED. %d episode(s) will fall back to metadata-only "
                "scoring. Consider lowering _BATCH_SIZE or raising "
                "token_budget_per_episode.",
                len(entries), len(items), len(items) - len(entries),
            )
    except Exception as exc:
        log.warning(
            "Batch Stage 2 LLM call failed (%s) — falling back to metadata for all",
            describe_exception(exc),
        )
        entries = []

    # Match on the echoed item index, not on position.
    #
    # The prompt says "one object per item, in the same order" and nothing
    # enforced it. A single reordered or dropped entry shifted every later one
    # silently, so an episode was scored, summarised and routed using ANOTHER
    # item's answer -- observed live: a Jason & Scot podcast episode published
    # into the personalization feed carrying an arXiv RAG paper's summary,
    # score and lane. Nothing errors, and the output looks entirely plausible
    # until someone reads the summary next to the title.
    by_index: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("item")
        if raw is None:
            continue
        try:
            by_index[int(raw)] = entry
        except (TypeError, ValueError):
            continue

    if by_index and len(by_index) < len(items):
        log.warning(
            "Stage 2 echoed %d item index(es) for %d episodes — the rest fall "
            "back to metadata rather than borrowing a neighbour's answer.",
            len(by_index), len(items),
        )

    results: list[RankedEpisode] = []
    for i, (ep, transcript) in enumerate(items):
        if by_index:
            # Once any entry is labelled, trust only labels: a positional guess
            # mixed in is exactly the failure this replaces.
            data = by_index.get(i, {})
        else:
            # No entry carried an index. Older behaviour, kept so a model that
            # ignores the field still produces a usable run.
            data = entries[i] if i < len(entries) and isinstance(entries[i], dict) else {}
        if not data:
            log.warning(
                "No batch result for episode %d (%s) — using metadata fallback",
                i,
                ep.episode_title,
            )
            s1 = stage1_metadata_score(ep, prefs)
            results.append(
                RankedEpisode(
                    episode=ep,
                    score=s1.score,
                    rubric=RubricScore(relevance=min(30, s1.score * 0.4)),
                    classification=_classify(s1.score, prefs),
                    classification_reason=BATCH_MISSING_REASON,
                    evidence_confidence="low",
                    summary=clean_snippet(ep.description, 300) or "Summary unavailable — no AI analysis for this episode.",
                    transcript_source=transcript.source,
                    tokens_used=0,
                )
            )
            continue

        try:
            rubric_data = data.get("rubric", {})
            rubric = RubricScore(
                **{
                    key: float(value)
                    for key, value in rubric_data.items()
                    if key in RubricScore.model_fields
                }
            )
            score = rubric.total

            def _str(value: Any, fallback: str = "") -> str:
                if value is None:
                    return fallback
                if isinstance(value, bool):
                    return "yes" if value else "no"
                return str(value)

            key_ideas = data.get("key_ideas", [])
            if not isinstance(key_ideas, list):
                key_ideas = []

            results.append(
                RankedEpisode(
                    episode=ep,
                    score=score,
                    rubric=rubric,
                    classification=_reconcile_classification(
                        _str(data.get("classification")), score, prefs
                    )[0],
                    classification_reason=_str(
                        data.get("classification_reason")
                    ),
                    evidence_confidence=transcript.confidence,
                    summary=_str(data.get("summary")),
                    key_ideas=[_str(idea) for idea in key_ideas],
                    implications=_str(data.get("implications")),
                    who_should_listen=_str(data.get("who_should_listen")),
                    summary_captures_value=_str(
                        data.get("summary_captures_value")
                    ),
                    listen_nuance=_str(data.get("listen_nuance")),
                    assigned_category=_str(data.get("category")),
                    transcript_source=transcript.source,
                    tokens_used=tokens_used // len(items),
                )
            )
        except Exception as exc:
            log.warning(
                "Could not parse batch entry %d for %s: %s",
                i,
                ep.episode_title,
                exc,
            )
            s1 = stage1_metadata_score(ep, prefs)
            results.append(
                RankedEpisode(
                    episode=ep,
                    score=s1.score,
                    rubric=RubricScore(relevance=min(30, s1.score * 0.4)),
                    classification=_classify(s1.score, prefs),
                    classification_reason="batch parse error; metadata fallback",
                    evidence_confidence="low",
                    summary=clean_snippet(ep.description, 300) or "Summary unavailable — no AI analysis for this episode.",
                    transcript_source=transcript.source,
                    tokens_used=0,
                )
            )

    return results
