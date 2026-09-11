"""Tests for the SEC EDGAR earnings adapter."""
from datetime import UTC, datetime, timedelta

import pytest

from podcast_scout.edgar import (
    EdgarCompany,
    EdgarConfig,
    earnings_filings,
    extract_text,
    find_earnings_exhibit,
    is_throttle_page,
    parse_edgar_config,
    strip_edgar_header,
)

# Trimmed from the live page SEC serves when the User-Agent is not accepted.
THROTTLE_HTML = """
<html><body><h1>Your Request Originates from an Undeclared Automated Tool</h1>
<p>To allow for equitable access to all users, SEC reserves the right to limit
requests originating from undeclared automated tools. Please declare your
traffic by updating your user agent to include company specific information.</p>
</body></html>
"""

# The Type column is what identifies the earnings release. Names vary per filer
# (earningsreleasefy27q2.htm, a2026q2ex-99.htm, exhibit991pressreleaseq220.htm).
INDEX_HTML = """
<table><tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/Archives/x/wmt-20260820.htm">wmt.htm</a></td><td>8-K</td><td>1</td></tr>
<tr><td>2</td><td>EX-99.1</td><td><a href="/Archives/x/earningsrelease.htm">rel.htm</a></td><td>EX-99.1</td><td>2</td></tr>
<tr><td>3</td><td>EX-99.2</td><td><a href="/Archives/x/deck.htm">deck.htm</a></td><td>EX-99.2</td><td>3</td></tr>
</table>
"""


def test_throttle_page_is_detected():
    """Load-bearing: SEC's refusal is well-formed HTML that extracts to fluent
    prose, so without this it is ranked as an earnings release."""
    assert is_throttle_page(THROTTLE_HTML)
    assert is_throttle_page(extract_text(THROTTLE_HTML))


def test_a_real_release_is_not_mistaken_for_a_throttle_page():
    assert not is_throttle_page(
        "Walmart Inc. reports Q2 results. eCommerce sales increased 24 percent."
    )


def test_exhibit_is_chosen_by_type_not_by_filename():
    href = find_earnings_exhibit(INDEX_HTML)
    assert href == "/Archives/x/earningsrelease.htm"


def test_ex_99_without_a_suffix_is_accepted():
    """Some filers use a bare EX-99; skipping them would silently drop a company."""
    html = INDEX_HTML.replace("EX-99.1", "EX-99").replace("EX-99.2", "GRAPHIC")
    assert find_earnings_exhibit(html) == "/Archives/x/earningsrelease.htm"


def test_no_exhibit_returns_empty():
    html = "<table><tr><td>1</td><td>8-K</td><td><a href='/x.htm'>x</a></td><td>8-K</td></tr></table>"
    assert find_earnings_exhibit(html) == ""


def test_tables_are_dropped_from_extracted_text():
    """An earnings exhibit is mostly XBRL-tagged statements; flattening them
    yields thousands of orphaned numbers that crowd out the narrative."""
    html = """
    <html><body><p>eCommerce sales increased 24%.</p>
    <table><tr><td>1234</td><td>5678</td></tr></table>
    <p>Advertising grew 38%.</p></body></html>
    """
    text = extract_text(html)
    assert "eCommerce sales increased 24%." in text
    assert "Advertising grew 38%." in text
    assert "1234" not in text


def test_edgar_viewer_header_is_stripped():
    raw = "EX-99.1\n2\nearningsreleasefy27q2.htm\nEX-99.1\nDocument\nWalmart U.S.\nSales grew."
    assert strip_edgar_header(raw).startswith("Walmart U.S.")


def test_header_stripping_leaves_a_normal_document_alone():
    body = "FOR IMMEDIATE RELEASE\nTarget Corporation Reports Second Quarter Earnings"
    assert strip_edgar_header(body) == body


def _submissions(rows):
    return {
        "name": "Walmart Inc.",
        "filings": {
            "recent": {
                "form": [r[0] for r in rows],
                "items": [r[1] for r in rows],
                "filingDate": [r[2] for r in rows],
                "reportDate": [r[2] for r in rows],
                "accessionNumber": [f"0000104169-26-{i:06d}" for i, _ in enumerate(rows)],
            }
        },
    }


def _days_ago(n):
    return (datetime.now(UTC) - timedelta(days=n)).strftime("%Y-%m-%d")


def test_only_8ks_tagged_item_2_02_are_earnings():
    """An 8-K is filed for a dozen unrelated reasons; the form alone is noise."""
    subs = _submissions([
        ("8-K", "2.02,9.01", _days_ago(5)),    # earnings
        ("8-K", "5.02", _days_ago(6)),         # an executive departure
        ("10-Q", "", _days_ago(7)),            # not an 8-K
        ("8-K", "2.02", _days_ago(400)),       # earnings, but out of window
    ])
    rows = earnings_filings(subs, datetime.now(UTC) - timedelta(days=100), limit=10)

    assert len(rows) == 1
    assert rows[0]["items"] == "2.02,9.01"


def test_filings_are_capped_per_company():
    subs = _submissions([("8-K", "2.02", _days_ago(i * 30)) for i in range(1, 5)])
    rows = earnings_filings(subs, datetime.now(UTC) - timedelta(days=365), limit=2)
    assert len(rows) == 2


def test_cik_is_padded_to_ten_digits():
    assert EdgarCompany(cik="104169").padded_cik() == "0000104169"
    assert EdgarCompany(cik="0000104169").padded_cik() == "0000104169"


def test_config_parses_companies_and_skips_entries_with_no_identifier():
    raw = {
        "edgar": {
            "max_filings_per_company": 3,
            "companies": [
                {"ticker": "wmt", "tags": ["competitive"]},
                {"name": "No identifier"},
                {"cik": "0000027419", "name": "Target"},
            ],
        }
    }
    config = parse_edgar_config(raw, user_agent="me (me@example.com)")

    assert [c.ticker or c.cik for c in config.companies] == ["WMT", "0000027419"]
    assert config.max_filings_per_company == 3
    assert config.user_agent == "me (me@example.com)"


def test_absent_edgar_block_yields_no_companies():
    assert parse_edgar_config({}).companies == []


@pytest.mark.asyncio
async def test_no_user_agent_disables_the_adapter_rather_than_guessing():
    """Inventing a contact address would breach SEC policy and fail silently."""
    from podcast_scout.edgar import fetch_earnings

    config = EdgarConfig(user_agent="", companies=[EdgarCompany(ticker="WMT")])
    assert await fetch_earnings(config, lookback_days=90) == []


@pytest.mark.asyncio
async def test_the_client_rejects_a_throttle_page_it_actually_receives():
    """Testing is_throttle_page() alone proves nothing: the hazard is the
    client admitting the page, and SEC serves it with a 200 in some paths."""
    import httpx

    from podcast_scout.edgar import _SecClient

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=THROTTLE_HTML)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        sec = _SecClient("me (me@example.com)", client)
        assert await sec.get("https://www.sec.gov/anything") is None


@pytest.mark.asyncio
async def test_the_client_passes_a_real_filing_through():
    import httpx

    from podcast_scout.edgar import _SecClient

    body = "<html><body><p>Walmart reports Q2 results.</p></body></html>"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
    async with httpx.AsyncClient(transport=transport) as client:
        sec = _SecClient("me (me@example.com)", client)
        assert await sec.get("https://www.sec.gov/anything") == body


@pytest.mark.asyncio
async def test_the_client_declares_the_configured_contact():
    """SEC refuses undeclared requests, so the agent must reach the wire."""
    import httpx

    from podcast_scout.edgar import _SecClient

    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers.get("User-Agent", "")
        return httpx.Response(200, text="ok")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await _SecClient("Scout (scout@example.com)", client).get("https://x/y")

    assert seen["ua"] == "Scout (scout@example.com)"


def test_a_repeated_sequence_number_is_stripped_with_the_header():
    """Some filers repeat the sequence number after "Document"; it otherwise
    survives as a bare "2" at the head of the release."""
    raw = "EX-99.1\n2\nearningsrelease.htm\nEX-99.1\nDocument\n2\nWalmart U.S.\nSales grew."

    assert strip_edgar_header(raw).startswith("Walmart U.S.")


def test_a_leading_number_that_is_part_of_the_document_survives():
    """Only a lone digit directly under the header is dropped."""
    raw = "2026 was a record year.\nRevenue grew."

    assert strip_edgar_header(raw).startswith("2026 was a record year.")
