"""Earnings releases via SEC EDGAR.

The design docs rank earnings highest because it is the only class that can
contradict vendor marketing: a vendor asserting agentic commerce is inevitable
is a pattern source, a retailer reporting that agent traffic did not convert is
evidence. The obstacle was always distribution -- most IR sites publish no
feed, so there was nothing for the RSS adapter in sources.py to subscribe to.

EDGAR is the feed those sites do not provide. Every US-listed company files its
quarterly results as an 8-K tagged with item 2.02, "Results of Operations and
Financial Condition", and the numbers live in exhibit EX-99.1. That makes the
path deterministic rather than a scrape:

    submissions JSON  ->  8-K rows where items contains 2.02
    filing index      ->  the row whose Type is EX-99.1
    exhibit           ->  text

Three things about the SEC that shape the code below:

* It refuses requests whose User-Agent does not declare a contact address, so
  the agent string is required configuration, not a default. See _REQUIRED_UA.
* Its refusal is a 200-status HTML page, not a 4xx. It parses cleanly and
  extracts to plausible prose, so it must be detected explicitly or the
  rubric ends up scoring SEC's error message as an earnings report.
* Its fair-access policy caps request rate. Requests are spaced accordingly.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .normalize import NormalizedEpisode, make_guid

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

# "Results of Operations and Financial Condition" -- the earnings item. An 8-K
# can be filed for a dozen unrelated reasons (an executive departure, a debt
# covenant), so filtering on the form alone would mostly return noise.
EARNINGS_ITEM = "2.02"

# SEC fair access allows 10 requests/second. A quarter of that is still far
# faster than this pipeline needs and leaves the limit to real users.
_REQUEST_INTERVAL = 0.25

# Enough of the release for the rubric to judge without blowing the Stage 2
# token budget on the statutory boilerplate that ends every filing.
_MAX_EXHIBIT_CHARS = 12_000

_REQUIRED_UA = (
    "SEC_USER_AGENT is not set. The SEC refuses requests that do not declare a "
    "contact address and returns a throttle page instead of the filing, so the "
    'adapter does not guess. Set it to something like "Your Name '
    '(you@example.com)" and re-run.'
)

# Phrases from SEC's throttle page. Matched on the extracted text rather than
# the status code: the page is served with 200 in some paths and 403 in others,
# and in both it is well-formed HTML that extracts to readable English.
_THROTTLE_MARKERS = (
    "undeclared automated tool",
    "request rate threshold",
    "declare your traffic",
)


@dataclass
class EdgarCompany:
    name: str = ""
    ticker: str = ""
    cik: str = ""
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    def padded_cik(self) -> str:
        """CIK zero-padded to the 10 digits the submissions API expects."""
        return self.cik.strip().lstrip("CIK").zfill(10) if self.cik else ""


@dataclass
class EdgarConfig:
    user_agent: str = ""
    companies: list[EdgarCompany] = field(default_factory=list)
    max_filings_per_company: int = 2


def parse_edgar_config(raw: dict[str, Any], user_agent: str = "") -> EdgarConfig:
    """Build an EdgarConfig from the `edgar:` block of sources.yaml."""
    block = raw.get("edgar") or {}
    companies = [
        EdgarCompany(
            name=item.get("name", ""),
            ticker=str(item.get("ticker", "")).upper().strip(),
            cik=str(item.get("cik", "")).strip(),
            tags=item.get("tags", []),
            enabled=item.get("enabled", True),
        )
        for item in block.get("companies", [])
        if item.get("ticker") or item.get("cik")
    ]
    return EdgarConfig(
        user_agent=user_agent,
        companies=companies,
        max_filings_per_company=block.get("max_filings_per_company", 2),
    )


def is_throttle_page(text: str) -> bool:
    """True if this is SEC's rate-limit page rather than a filing.

    Load-bearing. The page is valid HTML that extracts to fluent prose about
    automated tools and open data, so without this check it flows into the
    ranker as an earnings release and scores like one.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in _THROTTLE_MARKERS)


# EDGAR prepends a viewer header to every served exhibit: the exhibit type,
# the sequence number, the filename, the type again, then "Document". It
# extracts as text and would otherwise be the first thing the ranker reads.
# The trailing "(\n<digits>)?" is not belt-and-braces: some filers (Walmart
# among them) repeat the sequence number after "Document", which otherwise
# survives as a bare "2" at the head of the extracted release.
_EDGAR_HEADER_RE = re.compile(
    r"\A\s*EX-[\w.\-]+\s*\n\s*\d+\s*\n[^\n]+\n\s*EX-[\w.\-]+\s*\n"
    r"\s*Document\s*\n(?:\s*\d+\s*\n)?",
    re.IGNORECASE,
)


def strip_edgar_header(text: str) -> str:
    """Remove EDGAR's viewer header from the top of an extracted exhibit."""
    return _EDGAR_HEADER_RE.sub("", text, count=1).lstrip()


def extract_text(html: str, max_chars: int = _MAX_EXHIBIT_CHARS) -> str:
    """Readable prose from a filing exhibit.

    Tables are dropped. An earnings exhibit is mostly XBRL-tagged financial
    statements, and flattening those to text yields thousands of orphaned
    numbers that crowd out the narrative the rubric can actually read. The
    numbers that matter are restated in the highlights prose above them.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return strip_edgar_header(text.strip())[:max_chars]


def find_earnings_exhibit(index_html: str) -> str:
    """Path of the EX-99.1 exhibit in a filing index page, or "".

    Reads the index's Type column rather than guessing from filenames: an
    exhibit may be called anything (earningsreleasefy27q2.htm), and the largest
    .htm in a filing is as often the slide deck as the release.
    """
    soup = BeautifulSoup(index_html, "lxml")
    fallback = ""
    for row in soup.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in row.find_all("td")]
        link = row.find("a")
        if not cells or link is None or not link.get("href"):
            continue
        types = [c.upper() for c in cells]
        href = str(link["href"])
        if any(t == "EX-99.1" for t in types):
            return href
        # EX-99 and EX-99.2 are the usual shapes when a filer does not use
        # .1 -- kept as a fallback so those companies are not silently skipped.
        if not fallback and any(t.startswith("EX-99") for t in types):
            fallback = href
    return fallback


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None


def earnings_filings(
    submissions: dict[str, Any],
    cutoff: datetime,
    limit: int,
) -> list[dict[str, str]]:
    """8-K filings tagged item 2.02 within the window, newest first."""
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    rows: list[dict[str, str]] = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        items = (recent.get("items") or [""] * len(forms))[i]
        if EARNINGS_ITEM not in items:
            continue
        filed = _parse_date((recent.get("filingDate") or [""] * len(forms))[i])
        if filed is None or filed < cutoff:
            continue
        rows.append(
            {
                "accession": (recent.get("accessionNumber") or [""] * len(forms))[i],
                "filing_date": (recent.get("filingDate") or [""] * len(forms))[i],
                "report_date": (recent.get("reportDate") or [""] * len(forms))[i],
                "items": items,
            }
        )
        if len(rows) >= limit:
            break
    return rows


class _SecClient:
    """httpx wrapper that applies the SEC's declared-agent and rate rules."""

    def __init__(self, user_agent: str, client: httpx.AsyncClient) -> None:
        self.user_agent = user_agent
        self.client = client
        self._last_request = 0.0

    async def get(self, url: str) -> str | None:
        elapsed = asyncio.get_event_loop().time() - self._last_request
        if elapsed < _REQUEST_INTERVAL:
            await asyncio.sleep(_REQUEST_INTERVAL - elapsed)
        self._last_request = asyncio.get_event_loop().time()
        try:
            resp = await self.client.get(url, headers={"User-Agent": self.user_agent})
        except Exception as exc:
            log.warning("EDGAR request failed (%s): %s", url, exc)
            return None
        if resp.status_code != 200:
            log.warning("EDGAR returned %s for %s", resp.status_code, url)
            return None
        if is_throttle_page(resp.text[:4000]):
            log.warning(
                "EDGAR served its throttle page for %s. The User-Agent is not "
                "being accepted; it must name a contact address.", url,
            )
            return None
        return resp.text


async def resolve_ciks(
    companies: list[EdgarCompany],
    sec: _SecClient,
) -> None:
    """Fill in missing CIKs from tickers, in place. One request for all."""
    needed = [c for c in companies if not c.cik and c.ticker]
    if not needed:
        return
    body = await sec.get(TICKERS_URL)
    if body is None:
        log.warning("Could not fetch the SEC ticker map; skipping %d company(ies).", len(needed))
        return
    import json

    try:
        mapping = {
            str(entry["ticker"]).upper(): str(entry["cik_str"])
            for entry in json.loads(body).values()
        }
    except Exception as exc:
        log.warning("Could not parse the SEC ticker map: %s", exc)
        return
    for company in needed:
        cik = mapping.get(company.ticker)
        if cik:
            company.cik = cik
        else:
            log.warning("No CIK found for ticker %s.", company.ticker)


async def fetch_earnings(
    config: EdgarConfig,
    lookback_days: int,
    client: httpx.AsyncClient | None = None,
) -> list[NormalizedEpisode]:
    """Recent earnings releases for every configured company.

    Never raises: one unreachable company must not take down the daily run,
    exactly as an unreachable podcast feed already does not.
    """
    enabled = [c for c in config.companies if c.enabled]
    if not enabled:
        return []
    if not config.user_agent.strip():
        log.warning(_REQUIRED_UA)
        return []

    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    sec = _SecClient(config.user_agent, client)
    items: list[NormalizedEpisode] = []

    try:
        await resolve_ciks(enabled, sec)
        for company in enabled:
            cik = company.padded_cik()
            if not cik:
                continue
            body = await sec.get(SUBMISSIONS_URL.format(cik=cik))
            if body is None:
                continue
            import json

            try:
                submissions = json.loads(body)
            except Exception as exc:
                log.warning("Could not parse submissions for %s: %s", company.ticker or cik, exc)
                continue

            name = company.name or submissions.get("name", company.ticker or cik)
            filings = earnings_filings(submissions, cutoff, config.max_filings_per_company)
            for filing in filings:
                episode = await _build_episode(sec, company, name, cik, filing)
                if episode is not None:
                    items.append(episode)
            log.info("EDGAR %s: %d earnings filing(s)", name, len(filings))
    finally:
        if owns_client:
            await client.aclose()

    return items


async def _build_episode(
    sec: _SecClient,
    company: EdgarCompany,
    name: str,
    cik: str,
    filing: dict[str, str],
) -> NormalizedEpisode | None:
    accession = filing["accession"]
    bare = accession.replace("-", "")
    # int() strips the zero padding: the Archives path uses the unpadded CIK.
    folder = f"{ARCHIVE_BASE}/{int(cik)}/{bare}"
    index_url = f"{folder}/{accession}-index.html"

    index_html = await sec.get(index_url)
    if index_html is None:
        return None
    href = find_earnings_exhibit(index_html)
    if not href:
        log.info("No EX-99 exhibit in %s %s — skipping.", name, accession)
        return None

    exhibit_url = href if href.startswith("http") else f"https://www.sec.gov{href}"
    exhibit_html = await sec.get(exhibit_url)
    if exhibit_html is None:
        return None
    text = extract_text(exhibit_html)
    if len(text) < 400:
        # Too short to rank on. Skipped rather than admitted as a bare link:
        # an item with no text scores on its title alone, which is exactly the
        # metadata-only fallback the degradation guard exists to catch.
        log.info("Exhibit for %s %s held too little text — skipping.", name, accession)
        return None

    published = _parse_date(filing["filing_date"]) or datetime.now(UTC)
    # reportDate on an 8-K is the date of the reported event -- the earnings
    # announcement -- not the end of the period being reported. Titled as the
    # announcement date, because that is what it is.
    announced = filing.get("report_date") or filing["filing_date"]
    return NormalizedEpisode(
        guid=make_guid(index_url, accession),
        source_feed_url=index_url,
        original_guid=accession,
        show_title=name,
        episode_title=f"{name} — earnings announced {announced} (8-K item 2.02)",
        description=text,
        published=published,
        duration_seconds=0,
        episode_url=index_url,
        enclosure=None,
        source_type="earnings-call",
        credibility="high",
        bias_notes=(
            "Management commentary in a filed 8-K: selective in emphasis and "
            "framing, but legally constrained on fact. Reported figures can "
            "corroborate or contradict vendor performance claims; the narrative "
            "around them is still the company's own."
        ),
    )
