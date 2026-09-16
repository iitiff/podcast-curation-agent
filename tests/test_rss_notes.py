"""The notes a podcast player actually shows.

The summary was always in the feed -- in <description> and <content:encoded> --
but not in <itunes:summary>, which is the field Apple Podcasts, Pocket Casts
and Overcast read for the notes panel. These tests pin the fields, their
encoding, and the namespace handling that carrying an item to a second day
depends on.
"""
import re
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from xml.etree.ElementTree import Element

import feedparser
import pytest
from bs4 import BeautifulSoup

from podcast_scout.config import FeedConfig, Preferences
from podcast_scout.normalize import Enclosure, NormalizedEpisode
from podcast_scout.ranking import RankedEpisode, RubricScore
from podcast_scout.rss import (
    _RSS_NS,
    _add_prior_item,
    _build_channel,
    _load_prior_items,
    _subtitle,
    _xml_string,
    build_feed,
)
from podcast_scout.state import StateManager


def _ranked(**kwargs) -> RankedEpisode:
    episode = NormalizedEpisode(
        guid=kwargs.pop("guid", "g1"),
        source_feed_url="https://feeds.example.com/show",
        original_guid="ep1",
        show_title="The a16z Show",
        episode_title="After next-best-action",
        description="Short RSS blurb.",
        published=datetime(2026, 9, 8, tzinfo=UTC),
        duration_seconds=3915,
        episode_url="https://a16z.example/ep",
        enclosure=Enclosure(
            url="https://cdn.example.com/a.mp3", mime_type="audio/mpeg", length=1
        ),
    )
    return RankedEpisode(
        episode=episode,
        score=kwargs.pop("score", 82.0),
        rubric=RubricScore(),
        classification="Listen Fully",
        classification_reason="Strong on decision architecture.",
        evidence_confidence="medium",
        summary=kwargs.pop(
            "summary", "Ben & Marc on what replaces next-best-action. Worth the hour."
        ),
        key_ideas=kwargs.pop("key_ideas", ["Arbitration beats ranking"]),
        implications=kwargs.pop("implications", "Re-scope the roadmap."),
    )


def _feed_for(ranked: list[RankedEpisode]) -> str:
    """Render through the function production actually calls.

    These assertions previously ran against `_add_new_items`, a helper nothing
    in the pipeline invoked. So the show-notes guarantees below were verified on
    a code path no subscriber ever received, and `build_feed` could have
    diverged from every one of them without a test going red.
    """
    with tempfile.TemporaryDirectory() as directory:
        return build_feed(
            ranked, Preferences(feed=FeedConfig(title="t", description="d")),
            "listen", "https://x", StateManager(Path(directory)),
        )


def _rendered(entry) -> str:
    return BeautifulSoup(entry.summary, "lxml").get_text(" ", strip=True)


def test_itunes_summary_is_present():
    """The field the player reads. Its absence is why the summary never showed
    despite being in the feed twice over."""
    assert "<itunes:summary>" in _feed_for([_ranked()])


def test_the_summary_text_reaches_a_parsing_client():
    feed = feedparser.parse(_feed_for([_ranked()]))

    assert not feed.bozo
    rendered = _rendered(feed.entries[0])
    assert "Ben & Marc on what replaces next-best-action." in rendered
    assert "Arbitration beats ranking" in rendered
    assert "Re-scope the roadmap." in rendered


def test_notes_are_emitted_as_cdata():
    """What every real podcast feed uses. Clients disagree about entity-escaped
    markup: some render it, some print the tags."""
    xml = _feed_for([_ranked()])

    assert "<![CDATA[" in xml
    assert "&lt;p&gt;" not in xml


def test_html_entities_survive_the_cdata_round_trip():
    """The payload is escaped once for HTML and again by the serialiser;
    only the serialiser's layer may be removed. Get it wrong and the reader
    sees a literal &amp; or loses text to a phantom tag."""
    ranked = _ranked(
        summary="Tokens & costs.", key_ideas=["A <literal> tag & an ampersand"]
    )
    feed = feedparser.parse(_feed_for([ranked]))
    rendered = _rendered(feed.entries[0])

    assert "Tokens & costs." in rendered
    assert "&amp;" not in rendered
    assert "A <literal> tag & an ampersand" in rendered


def test_subtitle_is_one_plain_sentence_within_apples_limit():
    """Some clients drop an over-long subtitle rather than truncating it."""
    line = _subtitle(_ranked())

    assert line.startswith("82/100 — Ben & Marc")
    assert "Worth the hour" not in line          # one sentence only
    assert "<" not in line                        # plain text, no markup
    assert len(line) <= 255


def test_a_very_long_summary_is_truncated_for_the_subtitle():
    line = _subtitle(_ranked(summary="word " * 400))

    assert len(line) <= 255
    assert line.endswith("…")


# -- carrying an item to a second day ---------------------------------------

OLD_STYLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:atom="http://www.w3.org/2005/Atom" \
xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" \
xmlns:content="http://purl.org/rss/1.0/modules/content/" version="2.0">
<channel><title>t</title>
<item><title>Carried</title><guid isPermaLink="false">gOLD</guid>
<pubDate>{pub}</pubDate>
<enclosure url="https://cdn.example.com/ep.mp3" type="audio/mpeg" length="123"/>
<itunes:duration>3600</itunes:duration>
<content:encoded>&lt;p&gt;&lt;strong&gt;Score: 90/100&lt;/strong&gt;&lt;/p&gt;\
&lt;h3&gt;Summary&lt;/h3&gt;&lt;p&gt;Ben &amp;amp; Marc on arbitration.&lt;/p&gt;</content:encoded>
<description>short blurb</description></item>
</channel></rss>"""


def _carried_feed() -> str:
    pub = (datetime.now(UTC) - timedelta(days=2)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    directory = Path(tempfile.mkdtemp())
    (directory / "listen.xml").write_text(
        OLD_STYLE_FEED.format(pub=pub), encoding="utf-8"
    )
    priors = _load_prior_items(
        directory / "listen.xml",
        StateManager(Path(tempfile.mkdtemp())),
        datetime.now(UTC) - timedelta(days=21),
    )
    assert priors, "the item should be within the retention window"

    rss = Element("rss", attrib=_RSS_NS)
    channel = _build_channel(
        rss, FeedConfig(title="t", description="d"), None, "https://x/l.xml", "https://x"
    )
    for prior in priors:
        _add_prior_item(channel, prior)
    return _xml_string(rss)


def test_a_carried_item_keeps_its_itunes_prefix():
    """ElementTree re-serialises a parsed itunes: tag as ns0:, a prefix nothing
    declares. The feed is then not well-formed, and it only breaks once an item
    survives to a second day — which is why it was never seen."""
    xml = _carried_feed()

    assert not re.search(r"<ns\d+:", xml)
    assert "<itunes:duration>" in xml


def test_a_carried_item_gains_the_notes_fields():
    """Otherwise a notes fix reaches the feed only as items age out — three
    weeks at the current retention."""
    xml = _carried_feed()

    assert "<itunes:summary>" in xml
    assert "<![CDATA[" in xml


def test_a_carried_item_still_parses_and_reads_correctly():
    feed = feedparser.parse(_carried_feed())

    assert not feed.bozo
    assert "Ben & Marc on arbitration." in _rendered(feed.entries[0])


@pytest.mark.parametrize("undeclared", ["ns0", "ns1", "ns2"])
def test_no_undeclared_prefix_appears_in_any_feed(undeclared):
    for xml in (_feed_for([_ranked()]), _carried_feed()):
        assert f"{undeclared}:" not in xml


# ---------------------------------------------------------------------------
# The summaries feed: the read-only tier, readable in a player
# ---------------------------------------------------------------------------

def _summary_item(classification: str, *, audio: bool = True, guid: str = "g1", score: float = 60.0):
    from datetime import UTC, datetime

    from podcast_scout.normalize import Enclosure, NormalizedEpisode
    from podcast_scout.ranking import RankedEpisode, RubricScore

    ep = NormalizedEpisode(
        guid=guid, source_feed_url="https://f.example/x", original_guid=guid,
        show_title="A Show", episode_title="An Episode", description="Body text",
        published=datetime(2026, 9, 12, tzinfo=UTC), duration_seconds=1800,
        enclosure=Enclosure(url="https://cdn/x.mp3", mime_type="audio/mpeg", length=1) if audio else None,
    )
    return RankedEpisode(
        episode=ep, score=score, rubric=RubricScore(),
        classification=classification, summary="What this episode covers.",
    )


def _summaries_feed(items):
    import tempfile
    from pathlib import Path

    from podcast_scout.config import Preferences
    from podcast_scout.rss import build_feed
    from podcast_scout.state import StateManager

    with tempfile.TemporaryDirectory() as d:
        return build_feed(items, Preferences(), "summaries", "", StateManager(Path(d)))


def test_summaries_feed_carries_the_read_only_tier():
    xml = _summaries_feed([_summary_item("Read Summary Only")])

    assert "SUMMARY" in xml
    assert "An Episode" in xml


def test_summaries_feed_excludes_listen_fully():
    """Those already have listen.xml; duplicating them defeats the split."""
    xml = _summaries_feed([_summary_item("Listen Fully", guid="lf")])

    assert "An Episode" not in xml


def test_summaries_feed_excludes_skips():
    xml = _summaries_feed([_summary_item("Skip", guid="sk")])

    assert "An Episode" not in xml


def test_summaries_feed_requires_audio():
    """A player hides an item it cannot play, making the feed look broken."""
    xml = _summaries_feed([_summary_item("Read Summary Only", audio=False, guid="na")])

    assert "An Episode" not in xml


def test_summary_lands_where_players_actually_look():
    """content:encoded alone is not read by most clients -- the #30 lesson."""
    xml = _summaries_feed([_summary_item("Read Summary Only")])

    assert "itunes:summary" in xml
    assert "itunes:subtitle" in xml


# ── Every published feed is audio-only ────────────────────────────────
#
# These are RSS feeds a podcast client subscribes to, and a client hides an
# item it cannot play. Written sources reach the reader through the briefing
# and the email digest instead.

_NO_AUDIO_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:atom="http://www.w3.org/2005/Atom" \
xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" \
xmlns:content="http://purl.org/rss/1.0/modules/content/" version="2.0">
<channel><title>t</title>
<item><title>A paper</title><guid isPermaLink="false">gPAPER</guid>
<pubDate>{pub}</pubDate>
<content:encoded>&lt;p&gt;&lt;strong&gt;Score: 93/100&lt;/strong&gt;&lt;/p&gt;</content:encoded>
<description>no enclosure</description></item>
</channel></rss>"""


def test_prior_items_without_audio_are_not_carried_forward():
    """Filtering new items alone leaves the old ones in for FEED_RETENTION_DAYS.

    all.xml admitted written sources until the rule was removed; those items
    would otherwise have survived in the published feed for three more weeks.
    """
    pub = (datetime.now(UTC) - timedelta(days=2)).strftime("%a, %d %b %Y %H:%M:%S +0000")
    directory = Path(tempfile.mkdtemp())
    (directory / "all.xml").write_text(_NO_AUDIO_FEED.format(pub=pub), encoding="utf-8")
    priors = _load_prior_items(
        directory / "all.xml",
        StateManager(Path(tempfile.mkdtemp())),
        datetime.now(UTC) - timedelta(days=21),
    )
    assert priors == [], "an item with no enclosure must not be carried forward"


def _all_feed(items):
    import tempfile
    from pathlib import Path

    from podcast_scout.config import Preferences
    from podcast_scout.rss import build_feed
    from podcast_scout.state import StateManager

    with tempfile.TemporaryDirectory() as d:
        return build_feed(items, Preferences(), "all", "", StateManager(Path(d)))


def test_all_feed_excludes_items_with_no_audio():
    """all.xml was the last feed without the check, so it published papers —
    ranked with a headphones badge, and unplayable in any client."""
    xml = _all_feed([
        _summary_item("Listen Fully", guid="pod", score=80.0),
        _summary_item("Listen Fully", audio=False, guid="paper", score=93.0),
    ])
    assert "pod" in xml
    assert "paper" not in xml


def test_all_feed_still_carries_both_tiers_when_they_are_playable():
    xml = _all_feed([
        _summary_item("Listen Fully", guid="lf", score=80.0),
        _summary_item("Read Summary Only", guid="rso", score=60.0),
    ])
    assert "lf" in xml
    assert "rso" in xml


def _accumulates(feed_type: str, slug: str, classification: str) -> bool:
    """Publish one item, then a different one, and report whether the first survives."""
    import tempfile
    from pathlib import Path

    from podcast_scout.config import Preferences
    from podcast_scout.rss import build_feed
    from podcast_scout.state import EpisodeRecord, StateManager

    directory = Path(tempfile.mkdtemp())
    state = StateManager(Path(tempfile.mkdtemp()))
    for guid in ("day1", "day2"):
        state.mark_processed(EpisodeRecord(
            guid=guid, classification=classification, score=60.0,
            enclosure_url="https://cdn/x.mp3",
        ))
    day1 = build_feed(
        [_summary_item(classification, guid="day1")], Preferences(), feed_type, "",
        state, public_dir=directory,
    )
    (directory / slug).write_text(day1, encoding="utf-8")
    day2 = build_feed(
        [_summary_item(classification, guid="day2")], Preferences(), feed_type, "",
        state, public_dir=directory,
    )
    return "day1" in day2


def test_summaries_feed_keeps_its_history():
    """Prior-item retention defaulted to Listen Fully, which is every item this
    feed does NOT contain — so it dropped its whole history on every run and
    only ever showed the current day. A read-at-leisure feed that churns daily
    is the one failure it cannot have."""
    assert _accumulates("summaries", "summaries.xml", "Read Summary Only")


def test_listen_feed_keeps_its_history():
    assert _accumulates("listen", "listen.xml", "Listen Fully")


def test_all_feed_keeps_both_tiers():
    assert _accumulates("all", "all.xml", "Read Summary Only")
    assert _accumulates("all", "all.xml", "Listen Fully")


def test_carried_items_do_not_lose_their_cdata():
    """The round trip is lossy: parsing collapses a CDATA section to plain text
    and tostring() re-escapes it. An item therefore lost its CDATA on the first
    day it survived, which is why published feeds held a mix of both forms."""
    import tempfile
    from pathlib import Path

    from podcast_scout.config import Preferences
    from podcast_scout.state import EpisodeRecord, StateManager

    directory = Path(tempfile.mkdtemp())
    state = StateManager(Path(tempfile.mkdtemp()))
    for guid in ("day1", "day2", "day3"):
        state.mark_processed(EpisodeRecord(
            guid=guid, classification="Listen Fully", score=80.0,
            enclosure_url="https://cdn/x.mp3",
        ))

    xml = build_feed(
        [_summary_item("Listen Fully", guid="day1")], Preferences(), "listen",
        "https://x", state, public_dir=directory,
    )
    for guid in ("day2", "day3"):
        (directory / "listen.xml").write_text(xml, encoding="utf-8")
        xml = build_feed(
            [_summary_item("Listen Fully", guid=guid)], Preferences(), "listen",
            "https://x", state, public_dir=directory,
        )

    assert xml.count("<item>") == 3, "items should be accumulating"
    assert "&lt;p&gt;" not in xml, "a carried item re-escaped its notes"
    assert "<![CDATA[" in xml
