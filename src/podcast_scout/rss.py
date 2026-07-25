"""RSS 2.0 feed generator — one feed per category."""
from __future__ import annotations

import html
from xml.etree.ElementTree import Element, SubElement, tostring, indent

from .config import CategoryFeedConfig, FeedConfig, Preferences
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
    return html.escape(str(text), quote=True)


def _prefix(r: RankedEpisode, rank: int | None = None) -> str:
    outside = "🌐 OUTSIDE — " if r.episode.is_outside_feed else ""
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


def _build_channel(
    rss: Element,
    feed_cfg: FeedConfig,
    cat_cfg: CategoryFeedConfig | None,
    feed_url: str,
    base_url: str,
) -> Element:
    channel = SubElement(rss, "channel")
    title = cat_cfg.title if cat_cfg else feed_cfg.title
    description = cat_cfg.description if (cat_cfg and cat_cfg.description) else feed_cfg.description
    SubElement(channel, "title").text = title
    SubElement(channel, "description").text = description
    SubElement(channel, "link").text = base_url or "https://example.com"
    SubElement(channel, "language").text = feed_cfg.language
    SubElement(channel, "lastBuildDate").text = utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")
    SubElement(channel, "itunes:explicit").text = "false"
    SubElement(channel, "itunes:block").text = "Yes"
    atom_link = SubElement(channel, "atom:link")
    atom_link.set("href", feed_url)
    atom_link.set("rel", "self")
    atom_link.set("type", "application/rss+xml")
    if feed_cfg.owner_name:
        owner = SubElement(channel, "itunes:owner")
        SubElement(owner, "itunes:name").text = feed_cfg.owner_name
        if feed_cfg.owner_email:
            SubElement(owner, "itunes:email").text = feed_cfg.owner_email
    return channel


def _add_items(
    channel: Element,
    episodes: list[RankedEpisode],
    state: StateManager,
) -> None:
    listen_rank = 1
    for r in episodes:
        item = SubElement(channel, "item")
        rank = listen_rank if r.classification == "Listen Fully" else None
        SubElement(item, "title").text = _prefix(r, rank)
        if r.classification == "Listen Fully":
            listen_rank += 1
        SubElement(item, "link").text = r.episode.episode_url or ""
        SubElement(item, "guid", attrib={"isPermaLink": "false"}).text = r.episode.guid
        SubElement(item, "pubDate").text = r.episode.published.strftime("%a, %d %b %Y %H:%M:%S +0000")
        SubElement(item, "itunes:duration").text = str(r.episode.duration_seconds)
        notes = SubElement(item, "content:encoded")
        notes.text = _show_notes_html(r)
        SubElement(item, "description").text = r.summary or r.episode.description[:300]
        if r.episode.enclosure:
            enc = SubElement(item, "enclosure")
            enc.set("url", r.episode.enclosure.url)
            enc.set("type", r.episode.enclosure.mime_type)
            enc.set("length", str(r.episode.enclosure.length))
        if r.episode.image_url:
            img = SubElement(item, "itunes:image")
            img.set("href", r.episode.image_url)
        state.add_published(r.episode.guid)


def _xml_string(rss: Element) -> str:
    indent(rss, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(rss, encoding="unicode")


def build_category_feed(
    episodes: list[RankedEpisode],
    category: str,
    prefs: Preferences,
    base_url: str,
    state: StateManager,
) -> str:
    """
    Build an RSS feed for a single category (e.g. 'ai_retail', 'startup', 'chinese').
    Only episodes whose show_category matches `category` are included.
    Episodes are split into Listen Fully / Read Summary using per-category caps.
    """
    cat_cfg = prefs.categories.get(category)
    slug = cat_cfg.slug if cat_cfg else category.replace("_", "-")
    feed_url = f"{base_url.rstrip('/')}/{slug}.xml" if base_url else ""

    max_listen = cat_cfg.max_listen_fully if cat_cfg else prefs.output_caps.max_listen_fully
    max_summary = cat_cfg.max_read_summary if cat_cfg else prefs.output_caps.max_read_summary

    cat_episodes = [r for r in episodes if getattr(r.episode, "category", None) == category]

    listen = [r for r in cat_episodes if r.classification == "Listen Fully" and r.episode.enclosure][:max_listen]
    summary = [r for r in cat_episodes if r.classification == "Read Summary Only"][:max_summary]
    items = listen + summary

    rss = Element("rss", attrib=_RSS_NS)
    channel = _build_channel(rss, prefs.feed, cat_cfg, feed_url, base_url)
    _add_items(channel, items, state)
    return _xml_string(rss)


def build_feed(
    episodes: list[RankedEpisode],
    prefs: Preferences,
    feed_type: str,
    base_url: str,
    state: StateManager,
    existing_items: list[dict] | None = None,
) -> str:
    """
    Legacy combined feed builder — kept for backward compatibility.
    feed_type: 'listen' | 'all'
    """
    feed_url = f"{base_url.rstrip('/')}/{'listen.xml' if feed_type == 'listen' else 'all.xml'}" if base_url else ""
    rss = Element("rss", attrib=_RSS_NS)
    channel = _build_channel(rss, prefs.feed, None, feed_url, base_url)

    if feed_type == "listen":
        items = [r for r in episodes if r.classification == "Listen Fully" and r.episode.enclosure]
    else:
        items = [r for r in episodes if r.classification in ("Listen Fully", "Read Summary Only")]

    _add_items(channel, items, state)
    return _xml_string(rss)
