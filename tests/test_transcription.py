"""Full-transcript acquisition: per-item publisher tags, chunked audio, the cascade."""
from datetime import UTC, datetime

import pytest

from podcast_scout.feeds import parse_feed_entries
from podcast_scout.normalize import Enclosure, NormalizedEpisode, TranscriptRef
from podcast_scout.providers.base import TranscriptResult
from podcast_scout.providers.transcription import (
    AudioChunkTranscriber,
    AudioTranscriptionProvider,
    CascadeTranscriptionProvider,
    _extract_transcript_urls,
    _is_frame_sync,
    extract_item_transcripts,
    split_mp3,
)

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Show</title>
    <item>
      <title>Two</title>
      <guid><![CDATA[ep-2]]></guid>
      <pubDate>Tue, 14 Jan 2025 10:00:00 +0000</pubDate>
      <enclosure url="https://cdn.example.com/2.mp3?a=1&amp;b=2" type="audio/mpeg" length="1"/>
      <podcast:transcript type="text/vtt" url="https://t.example.com/2.vtt"/>
      <podcast:transcript url="https://t.example.com/2.txt" type="text/plain"/>
    </item>
    <item>
      <title>One</title>
      <guid>ep-1</guid>
      <pubDate>Mon, 13 Jan 2025 10:00:00 +0000</pubDate>
      <enclosure url="https://cdn.example.com/1.mp3" type="audio/mpeg" length="1"/>
    </item>
  </channel>
</rss>
"""

LONG_TEXT = "We kept a spreadsheet of every refund by hand. " * 20


def _ep(**kw: object) -> NormalizedEpisode:
    base: dict[str, object] = {
        "guid": "g1",
        "source_feed_url": "https://feeds.example.com/show",
        "original_guid": "ep-1",
        "show_title": "Show",
        "episode_title": "One",
        "description": "Show notes.",
        "published": datetime(2025, 1, 13, tzinfo=UTC),
        "enclosure": Enclosure(url="https://cdn.example.com/1.mp3"),
    }
    base.update(kw)
    return NormalizedEpisode(**base)  # type: ignore[arg-type]


def test_tags_map_to_their_own_item_only():
    by_item = extract_item_transcripts(FEED)
    assert by_item["ep-2"] == [
        ("https://t.example.com/2.txt", "text/plain"),
        ("https://t.example.com/2.vtt", "text/vtt"),
    ]
    # The enclosure key is unescaped, matching what feedparser hands back.
    assert by_item["https://cdn.example.com/2.mp3?a=1&b=2"] == by_item["ep-2"]
    # The episode without tags must not inherit its neighbour's transcript.
    assert "ep-1" not in by_item


def test_type_before_url_is_still_read():
    tag = '<podcast:transcript type="text/plain" url="https://x/t.txt" />'
    assert _extract_transcript_urls(tag) == [("https://x/t.txt", "text/plain")]


def test_parsed_episode_carries_its_transcript_refs():
    eps = parse_feed_entries(FEED, "https://feeds.example.com/show", None,
                             datetime(2025, 1, 1, tzinfo=UTC))
    by_title = {e.episode_title: e for e in eps}
    assert [t.url for t in by_title["Two"].transcript_urls][0] == "https://t.example.com/2.txt"
    assert by_title["One"].transcript_urls == []


# -- MP3 chunking -------------------------------------------------------------

_FRAME = bytes([0xFF, 0xFB, 0x90, 0x64]) + b"\x00" * 413  # MPEG1 L3 128k 44.1k


def test_split_mp3_cuts_on_frame_headers_and_loses_nothing():
    data = b"ID3" + b"\x00" * 7 + _FRAME * 300
    chunks = split_mp3(data, 10_000)
    assert len(chunks) > 1
    assert b"".join(chunks) == data
    for chunk in chunks[1:]:
        assert _is_frame_sync(chunk, 0)


def test_small_audio_is_one_chunk():
    assert split_mp3(_FRAME * 3, 10_000) == [_FRAME * 3]


# -- Audio provider -------------------------------------------------------------

class _FakeTranscriber(AudioChunkTranscriber):
    source = "audio"

    def __init__(self, fail_part: int | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        self.fail_part = fail_part

    async def transcribe_chunk(self, data: bytes, mime_type: str, part: int, total: int) -> str:
        self.calls.append((part, total))
        if part == self.fail_part:
            raise RuntimeError("quota")
        return f"part {part}: " + LONG_TEXT


@pytest.fixture
def audio_bytes(httpx_mock):
    data = _FRAME * 300
    httpx_mock.add_response(url="https://cdn.example.com/1.mp3", content=data)
    return data


async def test_audio_transcribes_every_chunk_in_order(audio_bytes):
    fake = _FakeTranscriber()
    provider = AudioTranscriptionProvider(fake, chunk_mb=40_000 / (1024 * 1024))
    result = await provider.transcribe_episode(_ep())
    total = len(fake.calls)
    assert total > 1
    assert fake.calls == [(i, total) for i in range(1, total + 1)]
    assert result.source == "audio" and result.is_full
    assert result.text.index("part 1:") < result.text.index(f"part {total}:")


async def test_a_lost_chunk_fails_the_episode_instead_of_passing_as_complete(audio_bytes):
    provider = AudioTranscriptionProvider(
        _FakeTranscriber(fail_part=2), chunk_mb=40_000 / (1024 * 1024),
        retries=1, retry_delay_s=0,
    )
    result = await provider.transcribe_episode(_ep())
    assert result.text == "" and not result.is_full


async def test_repeated_failures_stop_audio_for_the_run(httpx_mock):
    httpx_mock.add_response(url="https://cdn.example.com/1.mp3", content=_FRAME * 30, is_reusable=True)
    fake = _FakeTranscriber(fail_part=1)
    provider = AudioTranscriptionProvider(fake, retries=0, retry_delay_s=0)
    for _ in range(3):
        await provider.transcribe_episode(_ep())
    assert len(fake.calls) == 2


async def test_audio_cap_per_run():
    provider = AudioTranscriptionProvider(_FakeTranscriber(), max_episodes=1)
    provider.episodes_attempted = 1
    assert (await provider.transcribe_episode(_ep())).text == ""


# -- Cascade ----------------------------------------------------------------------

class _StubAudio(AudioTranscriptionProvider):
    def __init__(self, result: TranscriptResult) -> None:
        super().__init__(_FakeTranscriber())
        self.result = result
        self.called = False

    async def transcribe_episode(self, episode: NormalizedEpisode) -> TranscriptResult:
        self.called = True
        return self.result


async def test_publisher_transcript_wins_and_skips_audio(httpx_mock):
    httpx_mock.add_response(url="https://t.example.com/1.txt", text=LONG_TEXT)
    audio = _StubAudio(TranscriptResult(text="audio", source="audio", confidence="high"))
    ep = _ep(transcript_urls=[TranscriptRef(url="https://t.example.com/1.txt", mime_type="text/plain")])
    result = await CascadeTranscriptionProvider(audio=audio).transcribe_episode(ep)
    assert result.source == "publisher" and result.text == LONG_TEXT
    assert not audio.called


async def test_feed_refetch_matches_the_episode_by_guid(httpx_mock):
    httpx_mock.add_response(url="https://feeds.example.com/show", text=FEED)
    httpx_mock.add_response(url="https://t.example.com/2.txt", text=LONG_TEXT)
    ep = _ep(original_guid="ep-2", enclosure=None)
    result = await CascadeTranscriptionProvider().transcribe_episode(ep)
    assert result.source == "publisher"


async def test_no_tag_falls_to_audio_then_description(httpx_mock):
    httpx_mock.add_response(url="https://feeds.example.com/show", text=FEED, is_reusable=True)
    audio = _StubAudio(TranscriptResult(text=LONG_TEXT, source="audio", confidence="high"))
    assert (await CascadeTranscriptionProvider(audio=audio).transcribe_episode(_ep())).source == "audio"

    fallback = await CascadeTranscriptionProvider().transcribe_episode(_ep())
    assert fallback.source == "description" and not fallback.is_full
