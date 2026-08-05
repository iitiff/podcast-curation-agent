"""Unit tests for feed parsing."""
from datetime import datetime, timezone
from xml.etree.ElementTree import fromstring

from podcast_scout.feeds import parse_feed_entries
from podcast_scout.rss import _PriorItem, _renumber_prior_title


SAMPLE_RSS = """
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Podcast</title>
    <link>https://example.com</link>
    <description>A test podcast</description>
    <item>
      <title>Episode 1: AI Deep Dive</title>
      <link>https://example.com/ep1</link>
      <guid>https://example.com/ep1</guid>
      <pubDate>Mon, 13 Jan 2025 10:00:00 +0000</pubDate>
      <itunes:duration>3600</itunes:duration>
      <description>A deep dive into AI strategy for retail companies.</description>
      <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg" length="100000"/>
    </item>
    <item>
      <title>Very Old Episode</title>
      <link>https://example.com/ep0</link>
      <guid>https://example.com/ep0</guid>
      <pubDate>Mon, 01 Jan 2024 10:00:00 +0000</pubDate>
      <description>Too old.</description>
    </item>
  </channel>
</rss>
"""


def test_parse_filters_old_episodes():
    cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
    eps = parse_feed_entries(SAMPLE_RSS, "https://feeds.example.com", "Test Podcast", cutoff)
    assert len(eps) == 1
    assert eps[0].episode_title == "Episode 1: AI Deep Dive"


def test_parse_enclosure():
    cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
    eps = parse_feed_entries(SAMPLE_RSS, "https://feeds.example.com", "Test Podcast", cutoff)
    ep = eps[0]
    assert ep.enclosure is not None
    assert ep.enclosure.url == "https://example.com/ep1.mp3"
    assert ep.duration_seconds == 3600


def test_parse_duration_from_itunes():
    cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
    eps = parse_feed_entries(SAMPLE_RSS, "https://feeds.example.com", "Test Podcast", cutoff)
    assert eps[0].duration_minutes == 60.0


# ---------------------------------------------------------------------------
# Prior-item rank renumbering
#
# Carried-over items are appended as raw XML from the previous feed, so their
# <title> embeds the rank they held in THAT run. Observed live in startup.xml
# on 2026-08-05: two separate items both rendered as "LISTEN #1".
# ---------------------------------------------------------------------------

def _prior(title: str) -> _PriorItem:
    el = fromstring("<item><title>x</title></item>")
    el.find("title").text = title
    return _PriorItem(
        guid="g",
        pub_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
        score=77.0,
        classification="Listen Fully",
        xml_element=el,
    )


def _title(p: _PriorItem) -> str:
    return p.xml_element.find("title").text


def test_renumber_replaces_existing_rank():
    p = _prior("\U0001f3a7 LISTEN #7 \u2014 Show: Ep")
    _renumber_prior_title(p, 3)
    assert "LISTEN #3" in _title(p)
    assert "#7" not in _title(p)


def test_renumber_adds_rank_when_absent():
    p = _prior("\U0001f3a7 LISTEN \u2014 Show: Ep")
    _renumber_prior_title(p, 2)
    assert "LISTEN #2" in _title(p)


def test_renumber_preserves_outside_prefix():
    p = _prior("\U0001f310 OUTSIDE \u2014 \U0001f3a7 LISTEN #1 \u2014 Show: Ep")
    _renumber_prior_title(p, 4)
    t = _title(p)
    assert t.startswith("\U0001f310 OUTSIDE")
    assert "LISTEN #4" in t


def test_renumber_leaves_summary_titles_untouched():
    """SUMMARY items never carry a rank, so nothing should change."""
    p = _prior("\U0001f4d6 SUMMARY \u2014 Show: Ep")
    before = _title(p)
    _renumber_prior_title(p, 9)
    assert _title(p) == before


def test_renumber_handles_missing_title_element():
    el = fromstring("<item><guid>g</guid></item>")
    p = _PriorItem(
        guid="g",
        pub_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
        score=77.0,
        classification="Listen Fully",
        xml_element=el,
    )
    _renumber_prior_title(p, 1)  # must not raise
    assert el.find("title") is None


def test_renumber_produces_sequential_ranks_for_duplicates():
    """Reproduces the live bug: two prior items both labelled #1."""
    items = [
        _prior("\U0001f3a7 LISTEN #1 \u2014 YC: Waymo Co-CEO Dmitri Dolgov"),
        _prior("\U0001f3a7 LISTEN #1 \u2014 The a16z Show: Joshua Achiam"),
    ]
    for i, p in enumerate(items, 1):
        _renumber_prior_title(p, i)
    assert "LISTEN #1" in _title(items[0])
    assert "LISTEN #2" in _title(items[1])
