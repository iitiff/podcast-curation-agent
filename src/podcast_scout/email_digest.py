"""Daily email digest sender via SMTP."""
from __future__ import annotations

import logging
import smtplib
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import NamedTuple

from .normalize import strip_html
from .ranking import RankedEpisode

log = logging.getLogger(__name__)


class SMTPConfig(NamedTuple):
    host: str
    port: int
    user: str
    password: str
    to: str
    from_addr: str = ""
    use_tls: bool = True


def _duration_str(minutes: float) -> str:
    if minutes <= 0:
        return "unknown length"
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h {m}m" if h else f"{m}m"


def _episode_row_html(r: RankedEpisode, label: str) -> str:
    """Render one episode row.

    Content priority is deliberate: **key ideas are the product**. They are the
    concrete takeaways worth scanning, so when they exist they are shown in
    full -- every idea, untruncated -- and the prose summary is omitted
    entirely. Showing both just restates the same content twice and buries the
    nuggets under a paragraph.

    The summary is a FALLBACK only, used when an episode has no key ideas
    (metadata-only scoring, or an LLM response that omitted them). In that case
    it is truncated, because it is usually raw publisher show-notes rather than
    analysis.
    """
    ep = r.episode
    link = ep.episode_url or ep.source_feed_url
    dur = _duration_str(ep.duration_minutes)
    score_badge = f"<span style='color:#888;font-size:12px'>Score {r.score:.0f}/100 · {dur}</span>"

    ideas = [strip_html(i).strip() for i in r.key_ideas]
    ideas = [i for i in ideas if i]

    if ideas:
        # Variable length by design: render ALL key ideas, in full.
        items = "".join(
            f"<li style='margin-bottom:4px'>{i}</li>" for i in ideas
        )
        body_html = (
            "<ul style='margin:8px 0 0;padding-left:20px;font-size:14px;"
            f"color:#333;line-height:1.55'>{items}</ul>"
        )
    else:
        # No key ideas — fall back to prose. strip_html() defends against an
        # unclosed tag from raw show-notes breaking the surrounding email HTML.
        raw_summary = strip_html(r.summary) or strip_html(ep.description)
        if raw_summary:
            snippet = raw_summary[:280].strip()
            if len(raw_summary) > 280:
                snippet = snippet.rsplit(" ", 1)[0].rstrip(".,;:-") + "…"
            body_html = (
                "<div style='font-size:14px;color:#444;margin-top:6px;"
                f"line-height:1.5'>{snippet}</div>"
            )
        else:
            body_html = (
                "<div style='font-size:13px;color:#999;margin-top:6px'>"
                "<em>No AI analysis available for this episode.</em></div>"
            )

    return f"""
<tr>
  <td style='padding:14px 0;border-bottom:1px solid #eee;vertical-align:top'>
    <div style='font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.5px'>{label}</div>
    <div style='font-size:16px;font-weight:600;margin:4px 0'>
      <a href='{link}' style='color:#1a1a1a;text-decoration:none'>{ep.show_title}: {ep.episode_title}</a>
    </div>
    <div style='margin:2px 0'>{score_badge}</div>
    {body_html}
    <div style='margin-top:8px'><a href='{link}' style='font-size:13px;color:#0066cc'>&#9654; Listen / Read &#8594;</a></div>
  </td>
</tr>"""


def build_email_html(
    queued: list[RankedEpisode],
    email_only: list[RankedEpisode],
    run_date: str,
    feed_url: str = "",
    accumulated_week: list[RankedEpisode] | None = None,
) -> str:
    """Build the full HTML email body.

    accumulated_week: episodes that scored well this week but never won a
    category-feed playlist slot. Only included on Friday synthesis runs.
    """
    sections: list[str] = []

    if queued:
        rows = ""
        for r in queued:
            if r.classification == "Listen Fully":
                label = "&#127911; In Your Queue"
            else:
                label = "&#128214; In Your Queue (Summary)"
            if r.episode.is_outside_feed:
                label = "&#127760; Outside Discovery &middot; " + label
            rows += _episode_row_html(r, label)
        sections.append(f"""
<h2 style='font-size:18px;margin:24px 0 8px;color:#111'>&#127911; Today's Queue ({len(queued)} episodes)</h2>
<p style='color:#555;font-size:13px;margin:0 0 12px'>These are loaded into your Pocket Casts feed.</p>
<table width='100%' cellpadding='0' cellspacing='0'>{rows}</table>""")

    if email_only:
        outside = [r for r in email_only if r.episode.is_outside_feed]
        followed = [r for r in email_only if not r.episode.is_outside_feed]

        if followed:
            rows = "".join(_episode_row_html(r, "&#128204; Worth Your Attention") for r in followed)
            sections.append(f"""
<h2 style='font-size:18px;margin:24px 0 8px;color:#111'>&#128204; Also Good This Week</h2>
<p style='color:#555;font-size:13px;margin:0 0 12px'>Scored well but didn't fit today's 2-hour queue.</p>
<table width='100%' cellpadding='0' cellspacing='0'>{rows}</table>""")

        if outside:
            rows = "".join(_episode_row_html(r, "&#127760; Outside Discovery") for r in outside)
            sections.append(f"""
<h2 style='font-size:18px;margin:24px 0 8px;color:#111'>&#127760; Outside Your Feed</h2>
<p style='color:#555;font-size:13px;margin:0 0 12px'>Discovered beyond your subscriptions.</p>
<table width='100%' cellpadding='0' cellspacing='0'>{rows}</table>""")

    # Weekly accumulated digest — only rendered on Friday synthesis runs
    if accumulated_week:
        top_accumulated = accumulated_week[:10]
        rows = "".join(
            _episode_row_html(r, "&#128197; This Week — Didn't Make It")
            for r in top_accumulated
        )
        sections.append(f"""
<h2 style='font-size:18px;margin:24px 0 8px;color:#111'>&#128197; This Week's Accumulated Queue</h2>
<p style='color:#555;font-size:13px;margin:0 0 12px'>Scored well this week but never won a daily playlist slot. Worth reading or saving for later.</p>
<table width='100%' cellpadding='0' cellspacing='0'>{rows}</table>""")

    feed_note = ""
    if feed_url:
        feed_note = f"<p style='font-size:12px;color:#999;margin-top:32px'>Pocket Casts feed: <a href='{feed_url}'>{feed_url}</a></p>"

    body = "".join(sections)
    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
</head>
<body style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:640px;margin:0 auto;padding:24px;color:#1a1a1a'>
<h1 style='font-size:22px;border-bottom:2px solid #eee;padding-bottom:12px'>Your Podcast Scout &#8212; {run_date}</h1>
{body}
{feed_note}
<p style='font-size:11px;color:#bbb;margin-top:40px'>Generated by podcast-curation-agent &middot; <a href='https://github.com/iitiff/podcast-curation-agent' style='color:#bbb'>source</a></p>
</body></html>"""


def send_digest(
    smtp: SMTPConfig,
    subject: str,
    html_body: str,
    text_body: str = "",
) -> None:
    from_addr = smtp.from_addr or smtp.user
    # Strip non-ASCII characters from address header values. Email addresses
    # must be ASCII per RFC 5321. A non-ASCII character such as U+00A0
    # (non-breaking space) pasted from a rich-text editor into SMTP_TO or
    # SMTP_USER fails the compat32 BytesGenerator that both as_bytes() and
    # send_message() use internally — even when switching to send_message().
    _clean = lambda s: s.encode("ascii", "ignore").decode("ascii").strip()

    msg = MIMEMultipart("alternative")
    # Encode subject as RFC 2047 UTF-8 so emoji and non-ASCII don't crash
    msg["Subject"] = Header(subject, "utf-8").encode()
    msg["From"] = _clean(from_addr)
    msg["To"] = _clean(smtp.to)

    if text_body:
        msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        if smtp.use_tls:
            with smtplib.SMTP(smtp.host, smtp.port) as server:
                server.ehlo()
                server.starttls()
                server.login(smtp.user, smtp.password)
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(smtp.host, smtp.port) as server:
                server.login(smtp.user, smtp.password)
                server.send_message(msg)
        log.info("Email digest sent to %s", smtp.to)
    except Exception as exc:
        log.error("Failed to send email digest: %s", exc)
        raise
