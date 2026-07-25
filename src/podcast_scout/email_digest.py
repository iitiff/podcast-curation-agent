"""Daily email digest — sends a summary of all scored episodes via SMTP."""
from __future__ import annotations

import logging
import smtplib
import ssl
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .ranking import RankedEpisode

logger = logging.getLogger(__name__)


class SMTPConfig:
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        to_address: str,
        use_tls: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.to_address = to_address
        self.use_tls = use_tls


def _build_html_digest(
    ranked: list[RankedEpisode],
    run_date: str,
    feed_url: str,
    templates_dir: Path,
) -> str:
    try:
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["html"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        tmpl = env.get_template("email.html.j2")
        listen = [e for e in ranked if e.classification == "Listen Fully"]
        summary_only = [e for e in ranked if e.classification == "Read Summary Only"]
        outside = [e for e in ranked if e.episode.is_outside_feed and e.classification != "Skip"]
        acquired = [
            e for e in ranked
            if "acquired" in e.episode.show_title.lower()
            and e.classification != "Skip"
        ]
        return tmpl.render(
            run_date=run_date,
            listen=listen,
            summary_only=summary_only,
            outside=outside,
            acquired=acquired,
            feed_url=feed_url,
        )
    except Exception as exc:
        logger.warning("Email template render failed: %s; using plain fallback.", exc)
        return _build_plain_fallback(ranked, run_date, feed_url)


def _build_plain_fallback(
    ranked: list[RankedEpisode],
    run_date: str,
    feed_url: str,
) -> str:
    lines = [f"<h2>Podcast Scout Digest — {run_date}</h2>"]

    sections = [
        ("\U0001f3a7 In Your Queue", [e for e in ranked if e.classification == "Listen Fully"]),
        ("\U0001f4d6 Worth Reading", [e for e in ranked if e.classification == "Read Summary Only"]),
        ("\U0001f310 Outside Your Feed", [e for e in ranked if e.episode.is_outside_feed and e.classification != "Skip"]),
    ]

    for section_title, episodes in sections:
        if not episodes:
            continue
        lines.append(f"<h3>{section_title}</h3><ul>")
        for ep in episodes:
            url = ep.episode.episode_url or "#"
            lines.append(
                f"<li><strong><a href='{url}'>{ep.episode.show_title}: "
                f"{ep.episode.episode_title}</a></strong> "
                f"(Score: {ep.final_score:.0f}, "
                f"{ep.episode.duration_minutes:.0f} min)<br>"
                f"{ep.executive_summary[:300]}...</li>"
            )
        lines.append("</ul>")

    if feed_url:
        lines.append(f"<p><a href='{feed_url}'>Open your curated feed in Pocket Casts</a></p>")
    return "\n".join(lines)


def send_digest(
    ranked: list[RankedEpisode],
    smtp_cfg: SMTPConfig,
    run_date: str,
    feed_url: str,
    templates_dir: Path,
) -> None:
    """Send the daily digest email. Logs and continues on failure."""
    # Only send if there's something worth reporting
    surfaced = [e for e in ranked if e.classification != "Skip"]
    if not surfaced:
        logger.info("No episodes to report; skipping email digest.")
        return

    html_body = _build_html_digest(ranked, run_date, feed_url, templates_dir)

    subject_counts = []
    listen_count = sum(1 for e in surfaced if e.classification == "Listen Fully")
    summary_count = sum(1 for e in surfaced if e.classification == "Read Summary Only")
    outside_count = sum(1 for e in surfaced if e.episode.is_outside_feed)
    if listen_count:
        subject_counts.append(f"{listen_count} to listen")
    if summary_count:
        subject_counts.append(f"{summary_count} to read")
    if outside_count:
        subject_counts.append(f"{outside_count} outside discovery")

    subject = f"Podcast Scout {run_date} — {', '.join(subject_counts) or 'Daily Digest'}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_cfg.user
    msg["To"] = smtp_cfg.to_address
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        if smtp_cfg.use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_cfg.host, smtp_cfg.port) as server:
                server.ehlo()
                server.starttls(context=context)
                server.login(smtp_cfg.user, smtp_cfg.password)
                server.sendmail(smtp_cfg.user, smtp_cfg.to_address, msg.as_string())
        else:
            with smtplib.SMTP_SSL(smtp_cfg.host, smtp_cfg.port) as server:
                server.login(smtp_cfg.user, smtp_cfg.password)
                server.sendmail(smtp_cfg.user, smtp_cfg.to_address, msg.as_string())
        logger.info("Digest email sent to %s", smtp_cfg.to_address)
    except Exception as exc:
        logger.error("Failed to send digest email: %s", exc)
