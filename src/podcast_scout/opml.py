"""OPML parser — extracts RSS feed URLs from Pocket Casts OPML exports."""
from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

from pydantic import BaseModel

_PRIVATE_FEED_PATTERNS = [
    re.compile(r"[?&](auth|token|key|secret|password|pw|apikey)=", re.I),
    re.compile(r"https?://[^/]+:[^@/]+@"),  # user:pass@host
]


class OPMLFeed(BaseModel):
    title: str
    xml_url: str
    html_url: str = ""
    is_private: bool = False


def _is_private(url: str) -> bool:
    return any(p.search(url) for p in _PRIVATE_FEED_PATTERNS)


def parse_opml(path: Path) -> list[OPMLFeed]:
    """Parse an OPML file and return a list of feed entries.

    Private or authenticated feed URLs are flagged but still returned
    so the caller can decide whether to skip them.
    """
    try:
        tree = ET.parse(path)  # noqa: S314
    except ET.ParseError as exc:
        raise ValueError(f"Malformed OPML at {path}: {exc}") from exc

    root = tree.getroot()
    feeds: list[OPMLFeed] = []

    for outline in root.iter("outline"):
        xml_url = outline.get("xmlUrl") or outline.get("url") or ""
        if not xml_url:
            continue
        raw_title = (
            outline.get("text")
            or outline.get("title")
            or outline.get("xmlUrl", "Unknown")
        )
        title: str = raw_title if raw_title is not None else "Unknown"
        html_url = outline.get("htmlUrl") or ""
        private = _is_private(xml_url)
        feeds.append(
            OPMLFeed(
                title=title.strip(),
                xml_url=xml_url.strip(),
                html_url=html_url.strip(),
                is_private=private,
            )
        )

    return feeds
