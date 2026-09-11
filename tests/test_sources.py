"""Non-podcast sources, and questions driving the radar."""
from datetime import UTC, datetime, timedelta

import pytest

from podcast_scout.brain import Question
from podcast_scout.normalize import Enclosure, NormalizedEpisode
from podcast_scout.sources import (
    CLASS_DEFAULTS,
    ArticleSource,
    SourcesConfig,
    fetch_articles,
    load_sources,
    queries_for_questions,
    sources_for_questions,
)

# ---------------------------------------------------------------------------
# Source classes carry their own trust. Vendor material is a pattern source,
# never neutral evidence; earnings is the only class that can contradict it.
# ---------------------------------------------------------------------------

def test_class_defaults_rank_vendor_below_earnings():
    assert CLASS_DEFAULTS["vendor"][0] == "low"
    assert CLASS_DEFAULTS["earnings-call"][0] == "high"
    assert "corroborated" in CLASS_DEFAULTS["vendor"][1]


def test_explicit_credibility_overrides_the_class_default():
    s = ArticleSource(name="X", url="u", source_type="vendor", credibility="high")
    assert s.resolved()[0] == "high"


def test_missing_sources_file_means_podcasts_only(tmp_path):
    """Absence must not be an error: the pipeline was podcast-only before."""
    assert load_sources(tmp_path).sources == []


def test_sources_file_is_parsed(tmp_path):
    (tmp_path / "sources.yaml").write_text(
        "sources:\n"
        "  - name: Modern Retail\n    url: https://x/feed\n    source_type: trade-press\n"
        "  - name: No URL\n    source_type: vendor\n"
    )
    cfg = load_sources(tmp_path)
    assert [s.name for s in cfg.sources] == ["Modern Retail"], "an entry with no url is skipped"


# ---------------------------------------------------------------------------
# Questions drive the radar. The evergreen seeds all end in "podcast", which is
# what confined discovery to one medium.
# ---------------------------------------------------------------------------

def test_question_generates_queries_without_a_medium_suffix():
    qs = [Question(id="q", title="T", question="What comes after next-best-action?")]
    queries = queries_for_questions(qs)
    assert "What comes after next-best-action" in queries
    # The interrogative opener is stripped for the second variant: search
    # engines handle noun phrases better than full questions.
    assert "next-best-action" in queries
    assert not any("podcast" in q.lower() for q in queries)


def test_untitled_question_is_skipped():
    assert queries_for_questions([Question(id="q", title="", question="")]) == []


def test_question_tags_narrow_the_source_classes():
    cfg = SourcesConfig(sources=[
        ArticleSource(name="a", url="u", source_type="trade-press"),
        ArticleSource(name="b", url="u", source_type="research-paper"),
        ArticleSource(name="c", url="u", source_type="vendor"),
    ])
    qs = [Question(id="q", title="T", question="Q?", tags=["architecture"])]
    kept = {s.source_type for s in sources_for_questions(qs, cfg).sources}
    assert kept == {"research-paper", "vendor"}


def test_untagged_questions_do_not_narrow_anything():
    """An untagged question is not a reason to read less."""
    cfg = SourcesConfig(sources=[ArticleSource(name="a", url="u", source_type="trade-press")])
    qs = [Question(id="q", title="T", question="Q?")]
    assert len(sources_for_questions(qs, cfg).sources) == 1


# ---------------------------------------------------------------------------
# Fetching. An article is an episode minus the audio.
# ---------------------------------------------------------------------------

_FEED = """<?xml version="1.0"?><rss version="2.0"><channel>
<title>Trade Weekly</title>
<item><title>Agentic checkout underperforms</title>
<link>https://ex.com/a</link><guid>a1</guid>
<description>Rollout data.</description>
<pubDate>{date}</pubDate></item>
</channel></rss>"""


class _Client:
    def __init__(self, status=200, body=""):
        self.status, self.body = status, body
        self.requested: list[str] = []

    async def get(self, url):
        self.requested.append(url)
        outer = self

        class _R:
            status_code = outer.status
            text = outer.body
        return _R()

    async def aclose(self): return None


def _recent_feed():
    when = (datetime.now(UTC) - timedelta(hours=6)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    return _FEED.format(date=when)


@pytest.mark.asyncio
async def test_article_has_no_audio_and_carries_its_class():
    cfg = SourcesConfig(sources=[
        ArticleSource(name="Trade Weekly", url="https://x/feed", source_type="trade-press")
    ])
    items = await fetch_articles(cfg, lookback_days=3, client=_Client(200, _recent_feed()))
    assert len(items) == 1
    item = items[0]
    assert item.source_type == "trade-press"
    assert item.credibility == "medium"
    assert item.bias_notes, "class default bias must be attached"
    assert item.enclosure is None and item.duration_seconds == 0
    assert item.is_playable is False, "an article must never enter an audio feed"


@pytest.mark.asyncio
async def test_a_failing_source_does_not_take_down_the_run():
    cfg = SourcesConfig(sources=[ArticleSource(name="Dead", url="https://x/404")])
    assert await fetch_articles(cfg, 3, client=_Client(404, "")) == []


@pytest.mark.asyncio
async def test_disabled_sources_are_not_fetched():
    client = _Client(200, _recent_feed())
    cfg = SourcesConfig(sources=[ArticleSource(name="Off", url="u", enabled=False)])
    assert await fetch_articles(cfg, 3, client=client) == []
    assert client.requested == []


@pytest.mark.asyncio
async def test_a_cross_posted_enclosure_is_stripped_from_a_non_podcast_source():
    """A blog feed occasionally carries audio; it must not inherit playability."""
    cfg = SourcesConfig(sources=[
        ArticleSource(name="Blog", url="u", source_type="vendor")
    ])
    items = await fetch_articles(cfg, 3, client=_Client(200, _recent_feed()))
    for item in items:
        assert item.enclosure is None


def test_podcast_item_with_audio_is_playable():
    ep = NormalizedEpisode(
        guid="g", source_feed_url="u", show_title="S", episode_title="E",
        published=datetime.now(UTC),
        enclosure=Enclosure(url="https://x/a.mp3"),
    )
    assert ep.is_playable is True
