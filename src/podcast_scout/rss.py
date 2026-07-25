"""RSS 2.0 feed generation for listen.xml and all.xml."""
from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

from .config import Preferences
from .normalize import utcnow
from .ranking import RankedEpisode
from .state import StateManager

logger = logging.getLogger(__name__)

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "podcast": "https://podcastindex.org/namespace/1.0",
}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def _rfc822(dt: datetime) -> str:
    return dt.strftime("%a, %d %b %Y %H:%M:%S +0000")


def _safe(text: str) -> str:
    return html.escape(text or "", quote=False)


def _episode_label(ep: RankedEpisode, rank: int | None = None) -> str:
    prefix = {
        "Listen Fully": f"\U0001f3a7 LISTEN",
        "Read Summary Only": "\U0001f4d6 SUMMARY",
    }.get(ep.classification, "\u23ed\ufe0f SKIP")
    if ep.episode.is_outside_feed:
        prefix += " \U0001f310 OUTSIDE"
    rank_str = f" #{rank}" if rank and ep.classification == "Listen Fully" else ""
    return f"{prefix}{rank_str} \u2014 {ep.episode.show_title}: {ep.episode.episode_title}"


def _build_channel(
    root: ET.Element,
    feed_title: str,
    feed_desc: str,
    prefs: Preferences,
    self_url: str,
    artwork_url: str,
) -> ET.Element:
    channel = ET.SubElement(root, "channel")
    ET.SubElement(channel, "title").text = feed_title
    ET.SubElement(channel, "description").text = feed_desc
    ET.SubElement(channel, "language").text = prefs.feed.language
    ET.SubElement(channel, "lastBuildDate").text = _rfc822(utcnow())
    ET.SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}explicit").text = "false"
    ET.SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}block").text = "Yes"
    if artwork_url:
        img = ET.SubElement(channel, "image")
        ET.SubElement(img, "url").text = artwork_url
        ET.SubElement(img, "title").text = feed_title
        it_img = ET.SubElement(
            channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}image"
        )
        it_img.set("href", artwork_url)
    if self_url:
        atom_link = ET.SubElement(
            channel, "{http://www.w3.org/2005/Atom}link"
        )
        atom_link.set("rel", "self")
        atom_link.set("type", "application/rss+xml")
        atom_link.set("href", self_url)
    if prefs.feed.owner_name:
        owner = ET.SubElement(channel, "{http://www.itunes.com/dtds/podcast-1.0.dtd}owner")
        ET.SubElement(owner, "{http://www.itunes.com/dtds/podcast-1.0.dtd}name").text = prefs.feed.owner_name
    return channel


def _add_item(channel: ET.Element, ep: RankedEpisode, label: str) -> None:
    enclosure = ep.episode.enclosure
    if enclosure is None:
        logger.warning("Skipping '%s' — no enclosure", ep.episode.episode_title)
        return

    item = ET.SubElement(channel, "item")
    ET.SubElement(item, "title").text = _safe(label)
    ET.SubElement(item, "guid").text = ep.episode.guid
    ET.SubElement(item, "pubDate").text = _rfc822(utcnow())
    ET.SubElement(item, "link").text = ep.episode.episode_url or ""

    enc_el = ET.SubElement(item, "enclosure")
    enc_el.set("url", enclosure.url)
    enc_el.set("type", enclosure.mime_type)
    enc_el.set("length", str(enclosure.length))

    summary_html = f"""
<p><strong>Score:</strong> {ep.final_score:.0f}/100 &mdash;
<strong>Classification:</strong> {ep.classification} &mdash;
<strong>Confidence:</strong> {ep.confidence}</p>
<p><strong>Why ranked:</strong> {_safe(ep.why_ranked)}</p>
<h3>Executive Summary</h3>
<p>{_safe(ep.executive_summary)}</p>
{('<h3>Key Ideas</h3><ul>' + ''.join(f'<li>{_safe(i)}</li>' for i in ep.key_ideas) + '</ul>') if ep.key_ideas else ''}
{('<h3>Implications</h3><p>' + _safe(ep.implications) + '</p>') if ep.implications else ''}
<p><a href="{ep.episode.episode_url}">Listen to original episode</a></p>
""".strip()
    ET.SubElement(
        item, "{http://purl.org/rss/1.0/modules/content/}encoded"
    ).text = summary_html
    ET.SubElement(
        item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}summary"
    ).text = ep.executive_summary[:4000] if ep.executive_summary else ""
    if ep.episode.duration_seconds:
        ET.SubElement(
            item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}duration"
        ).text = str(ep.episode.duration_seconds)
    if ep.episode.image_url:
        img_el = ET.SubElement(
            item, "{http://www.itunes.com/dtds/podcast-1.0.dtd}image"
        )
        img_el.set("href", ep.episode.image_url)


def generate_feeds(
    ranked: list[RankedEpisode],
    prefs: Preferences,
    public_dir: Path,
    state: StateManager,
    retention_days: int = 90,
    artwork_url: str = "",
) -> None:
    """Write listen.xml (Listen Fully only) and all.xml (Listen + Summary)."""
    base_url = prefs.feed.base_url.rstrip("/")
    feed_title = prefs.feed.title
    feed_desc = prefs.feed.description

    # Build listen.xml — Listen Fully with enclosures only
    listen_root = ET.Element("rss", version="2.0")
    listen_channel = _build_channel(
        listen_root, feed_title, feed_desc, prefs,
        self_url=f"{base_url}/listen.xml" if base_url else "",
        artwork_url=artwork_url,
    )

    # Build all.xml
    all_root = ET.Element("rss", version="2.0")
    all_channel = _build_channel(
        all_root, f"{feed_title} — All", feed_desc, prefs,
        self_url=f"{base_url}/all.xml" if base_url else "",
        artwork_url=artwork_url,
    )

    listen_rank = 1
    for ep in ranked:
        if ep.classification == "Skip":
            continue
        label = _episode_label(ep, rank=listen_rank if ep.classification == "Listen Fully" else None)
        if ep.classification == "Listen Fully":
            _add_item(listen_channel, ep, label)
            listen_rank += 1
        _add_item(all_channel, ep, label)
        state.add_published(ep.episode.guid)

    public_dir.mkdir(parents=True, exist_ok=True)
    _write_xml(listen_root, public_dir / "listen.xml")
    _write_xml(all_root, public_dir / "all.xml")
    logger.info("RSS feeds written to %s", public_dir)


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with path.open("wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)
    # Validate by re-parsing
    try:
        ET.parse(path)  # noqa: S314
        logger.debug("XML validation passed: %s", path)
    except ET.ParseError as exc:
        logger.error("Generated XML is invalid at %s: %s", path, exc)
