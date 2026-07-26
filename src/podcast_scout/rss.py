"""RSS 2.0 feed generator — one feed per category.

Feeds are *persistent*: each run merges today's newly ranked episodes with
items already in the published feed, re-sorts by score, and drops anything
older than FEED_RETENTION_DAYS (21 days).  This means episodes curated into
a podcast-app playlist will not disappear on the next daily update.
"""
from __future__ import annotations

import html
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, fromstring, indent, tostring

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

FEED_RETENTION_DAYS = 21


def _e(text: str) -> str:
    return html.escape(str(text), quote=True)


def _prefix(r: RankedEpisode, rank: int | None = None) -> str:
    outside = "\U0001f310 OUTSIDE \u2014 " if r.episode.is_outside_feed else ""
    if r.classification == "Listen Fully":
        num = f" #{rank}" if rank else ""
        return f"{outside}\U0001f3a7 LISTEN{num} \u2014 {r.episode.show_title}: {r.episode.episode_title}"
    return f"{outside}\U0001f4d6 SUMMARY \u2014 {r.episode.show_title}: {r.episode.episode_title}"


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


_RFC822 = "%a, %d %b %Y %H:%M:%S %z"


def _parse_rfc822(s: str) -> datetime | None:
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S +0000"):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


class _PriorItem:
    __slots__ = ("guid", "pub_date", "score", "classification", "xml_element")

    def __init__(
        self,
        guid: str,
        pub_date: datetime,
        score: float,
        classification: str,
        xml_element: Element,
    ) -> None:
        self.guid = guid
        self.pub_date = pub_date
        self.score = score
        self.classification = classification
        self.xml_element = xml_element


def _load_prior_items(
    feed_path: Path,
    state: StateManager,
    retention_cutoff: datetime,
) -> list[_PriorItem]:
    if not feed_path.exists():
        return []
    try:
        tree = fromstring(feed_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    items: list[_PriorItem] = []
    channel = tree.find("channel")
    if channel is None:
        return []

    for item_el in channel.findall("item"):
        guid_el = item_el.find("guid")
        pub_el = item_el.find("pubDate")
        if guid_el is None or not guid_el.text:
            continue
        guid = guid_el.text.strip()
        pub_date = _parse_rfc822(pub_el.text) if pub_el is not None and pub_el.text else None
        if pub_date is None or pub_date < retention_cutoff:
            continue

        rec = state.get_record(guid)
        score = rec.score if rec else 0.0
        classification = rec.classification if rec else "Read Summary Only"

        if score == 0.0:
            notes_el = item_el.find("{http://purl.org/rss/1.0/modules/content/}encoded")
            if notes_el is not None and notes_el.text:
                m = re.search(r"Score:\s*(\d+(?:\.\d+)?)/100", notes_el.text)
                if m:
                    score = float(m.group(1))

        items.append(_PriorItem(
            guid=guid,
            pub_date=pub_date,
            score=score,
            classification=classification,
            xml_element=item_el,
        ))

    return items


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


def _add_new_items(
    channel: Element,
    episodes: list[RankedEpisode],
    state: StateManager,
    listen_rank_start: int = 1,
) -> int:
    listen_rank = listen_rank_start
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
    return listen_rank


def _add_prior_item(channel: Element, prior: _PriorItem) -> None:
    channel.append(prior.xml_element)


def build_category_feed(
    episodes: list[RankedEpisode],
    category: str,
    prefs: Preferences,
    base_url: str,
    state: StateManager,
    public_dir: Path | None = None,
) -> str:
    cat_cfg = prefs.categories.get(category)
    slug = cat_cfg.slug if cat_cfg else category.replace("_", "-")
    feed_url = f"{base_url.rstrip('/')}/{slug}.xml" if base_url else ""

    max_listen = cat_cfg.max_listen_fully if cat_cfg else prefs.output_caps.max_listen_fully
    max_summary = cat_cfg.max_read_summary if cat_cfg else prefs.output_caps.max_read_summary

    cat_episodes = [r for r in episodes if getattr(r.episode, "category", None) == category]
    new_listen = [r for r in cat_episodes if r.classification == "Listen Fully" and r.episode.enclosure]
    new_summary = [r for r in cat_episodes if r.classification == "Read Summary Only"]
    new_items_today: list[RankedEpisode] = (new_listen + new_summary)
    new_guids = {r.episode.guid for r in new_items_today}

    retention_cutoff = utcnow() - timedelta(days=FEED_RETENTION_DAYS)
    prior_items: list[_PriorItem] = []
    if public_dir is not None:
        feed_path = public_dir / f"{slug}.xml"
        for p in _load_prior_items(feed_path, state, retention_cutoff):
            if p.guid not in new_guids:
                prior_items.append(p)

    merged: list[tuple[float, str, str, RankedEpisode | _PriorItem]] = [
        (r.score, r.classification, "new", r) for r in new_items_today
    ] + [
        (p.score, p.classification, "prior", p) for p in prior_items
    ]
    merged.sort(key=lambda t: t[0], reverse=True)

    listen_count = 0
    summary_count = 0
    final_order: list[tuple[str, RankedEpisode | _PriorItem]] = []
    for _score, classification, kind, obj in merged:
        if classification == "Listen Fully":
            if listen_count >= max_listen:
                continue
            listen_count += 1
        else:
            if summary_count >= max_summary:
                continue
            summary_count += 1
        final_order.append((kind, obj))

    rss = Element("rss", attrib=_RSS_NS)
    channel = _build_channel(rss, prefs.feed, cat_cfg, feed_url, base_url)
    listen_rank = 1
    for kind, obj in final_order:
        if kind == "new":
            if not isinstance(obj, RankedEpisode):
                continue
            r = obj
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
        else:
            if not isinstance(obj, _PriorItem):
                continue
            p = obj
            channel.append(p.xml_element)

    return _xml_string(rss)


def build_feed(
    episodes: list[RankedEpisode],
    prefs: Preferences,
    feed_type: str,
    base_url: str,
    state: StateManager,
    public_dir: Path | None = None,
) -> str:
    slug = "listen.xml" if feed_type == "listen" else "all.xml"
    feed_url = f"{base_url.rstrip('/')}/{slug}" if base_url else ""
    retention_cutoff = utcnow() - timedelta(days=FEED_RETENTION_DAYS)

    if feed_type == "listen":
        new_items = [r for r in episodes if r.classification == "Listen Fully" and r.episode.enclosure]
    else:
        new_items = [r for r in episodes if r.classification in ("Listen Fully", "Read Summary Only")]

    new_guids = {r.episode.guid for r in new_items}
    prior_items: list[_PriorItem] = []
    if public_dir is not None:
        feed_path = public_dir / slug
        for p in _load_prior_items(feed_path, state, retention_cutoff):
            if p.guid not in new_guids:
                prior_items.append(p)

    merged: list[tuple[float, str, RankedEpisode | _PriorItem]] = [
        (r.score, "new", r) for r in new_items
    ] + [
        (p.score, "prior", p) for p in prior_items
    ]
    merged.sort(key=lambda t: t[0], reverse=True)

    rss = Element("rss", attrib=_RSS_NS)
    channel = _build_channel(rss, prefs.feed, None, feed_url, base_url)
    listen_rank = 1
    for _score, kind, obj in merged:
        if kind == "new":
            if not isinstance(obj, RankedEpisode):
                continue
            r = obj
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
        else:
            if not isinstance(obj, _PriorItem):
                continue
            p = obj
            channel.append(p.xml_element)

    return _xml_string(rss)


def _xml_string(rss: Element) -> str:
    indent(rss, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(rss, encoding="unicode")
