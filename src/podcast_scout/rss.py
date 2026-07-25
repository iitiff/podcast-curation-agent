"""RSS 2.0 feed generator for Pocket Casts-compatible output."""
from __future__ import annotations

import hashlib
import html
from datetime import datetime, timezone
from xml.etree.ElementTree import Element, SubElement, tostring, indent

from .config import Preferences
from .normalize import utcnow
from .ranking import RankedEpisode
from .state import StateManager

_RSS_NS = {
    "xmlns:atom": "http://www.w3.org/2005/Atom",
    "xmlns:itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "xmlns:content": "http://purl.org/rss/1.0/modules/content/",
    "version": "2.0",
}


def _e(text: str) -> str:
    """XML-safe escape."""
    return html.escape(str(text), quote=True)


def _prefix(r: RankedEpisode, rank: int | None = None) -> str:
    if r.episode.is_outside_feed:
        outside = "🌐 OUTSIDE — "
    else:
        outside = ""
    if r.classification == "Listen Fully":
        num = f" #{rank}" if rank else ""
        return f"{outside}🎧 LISTEN{num} — {r.episode.show_title}: {r.episode.episode_title}"
    return f"{outside}📖 SUMMARY — {r.episode.show_title}: {r.episode.episode_title}"


def _show_notes_html(r: RankedEpisode) -> str:
    lines = [
        f"<p><strong>Score: {r.score:.0f}/100</strong> | "
        f"{r.classification} | Confidence: {r.evidence_confidence}</p>",
        f"<p><em>{_e(r.classification_reason)}</em></p>",
        f"<h3>Summary</h3><p>{_e(r.summary)}</p>",
    ]
    if r.key_ideas:
        lines.append("<h3>Key Ideas</h3><ul>")
        for idea in r.key_ideas:
            lines.append(f"<li>{_e(idea)}</li>")
        lines.append("</ul>")
    if r.implications:
        lines.append(f"<h3>Implications</h3><p>{_e(r.implications)}</p>")
    if r.episode.episode_url:
        lines.append(f'<p><a href="{_e(r.episode.episode_url)}">Listen / Read original</a></p>')
    return "".join(lines)


def build_feed(
    episodes: list[RankedEpisode],
    prefs: Preferences,
    feed_type: str,  # "listen" | "all"
    base_url: str,
    state: StateManager,
    existing_items: list[dict] | None = None,
) -> str:
    """Build RSS 2.0 XML string."""
    rss = Element("rss", attrib=_RSS_NS)
    channel = SubElement(rss, "channel")

    feed_url = f"{base_url.rstrip('/')}/{'listen.xml' if feed_type == 'listen' else 'all.xml'}"

    SubElement(channel, "title").text = prefs.feed.title
    SubElement(channel, "description").text = prefs.feed.description
    SubElement(channel, "link").text = base_url or "https://example.com"
    SubElement(channel, "language").text = prefs.feed.language
    SubElement(channel, "lastBuildDate").text = utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")
    SubElement(channel, "itunes:explicit").text = "false"
    SubElement(channel, "itunes:block").text = "Yes"
    atom_link = SubElement(channel, "atom:link")
    atom_link.set("href", feed_url)
    atom_link.set("rel", "self")
    atom_link.set("type", "application/rss+xml")
    if prefs.feed.owner_name:
        owner = SubElement(channel, "itunes:owner")
        SubElement(owner, "itunes:name").text = prefs.feed.owner_name
        if prefs.feed.owner_email:
            SubElement(owner, "itunes:email").text = prefs.feed.owner_email

    # Filter by type
    if feed_type == "listen":
        items = [r for r in episodes if r.classification == "Listen Fully" and r.episode.enclosure]
    else:
        items = [r for r in episodes if r.classification in ("Listen Fully", "Read Summary Only")]

    listen_rank = 1
    for r in items:
        item = SubElement(channel, "item")
        rank = listen_rank if r.classification == "Listen Fully" else None
        SubElement(item, "title").text = _prefix(r, rank)
        if r.classification == "Listen Fully":
            listen_rank += 1
        SubElement(item, "link").text = r.episode.episode_url or ""
        SubElement(item, "guid", attrib={"isPermaLink": "false"}).text = r.episode.guid
        pub_date = r.episode.published.strftime("%a, %d %b %Y %H:%M:%S +0000")
        SubElement(item, "pubDate").text = pub_date
        SubElement(item, "itunes:duration").text = str(r.episode.duration_seconds)
        notes = SubElement(item, "content:encoded")
        notes.text = _show_notes_html(r)
        desc = SubElement(item, "description")
        desc.text = r.summary or r.episode.description[:300]
        if r.episode.enclosure:
            enc = SubElement(item, "enclosure")
            enc.set("url", r.episode.enclosure.url)
            enc.set("type", r.episode.enclosure.mime_type)
            enc.set("length", str(r.episode.enclosure.length))
        if r.episode.image_url:
            img = SubElement(item, "itunes:image")
            img.set("href", r.episode.image_url)
        state.add_published(r.episode.guid)

    indent(rss, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(rss, encoding="unicode")
