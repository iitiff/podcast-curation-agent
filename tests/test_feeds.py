"""Unit tests for feed parsing."""
from datetime import datetime, timezone
from podcast_scout.feeds import parse_feed_entries


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
