"""Read-only Gmail inbox access for the assistant-mode voice agent.

The bridge runs as its own process and cannot reach the Gmail MCP server
that lives inside a Claude Code session, so it uses its own credential:
an app password over IMAP (stdlib only — no Google Cloud OAuth).

Set in .env:
    GMAIL_ADDRESS=you@gmail.com
    GMAIL_APP_PASSWORD=xxxxxxxxxxxxxxxx   # 16-char app password, 2FA required

fetch_recent() returns a list of compact dicts the LLM can summarize aloud,
or raises EmailNotConfigured / EmailFetchError so the caller can tell the
model email isn't available rather than crashing the call.
"""

from __future__ import annotations

import email
import email.utils
import imaplib
import logging
import os
from datetime import datetime, timedelta, timezone
from email.header import decode_header, make_header

log = logging.getLogger(__name__)

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
_MAX_SNIPPET = 200


class EmailNotConfigured(RuntimeError):
    """GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set."""


class EmailFetchError(RuntimeError):
    """IMAP login or fetch failed."""


def _decode(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return value.strip()


def _snippet(msg: email.message.Message) -> str:
    """Best-effort short plain-text preview of the message body."""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain" and not part.get_filename():
                    payload = part.get_payload(decode=True)
                    if payload:
                        text = payload.decode(part.get_content_charset() or "utf-8", "replace")
                        break
            else:
                return ""
        else:
            payload = msg.get_payload(decode=True)
            if not payload:
                return ""
            text = payload.decode(msg.get_content_charset() or "utf-8", "replace")
    except Exception:
        return ""
    text = " ".join(text.split())
    return text[:_MAX_SNIPPET]


def fetch_recent(days: int = 7, max_results: int = 25, mailbox: str = "INBOX") -> list[dict]:
    """Return recent messages newest-first as
    [{from, subject, date, snippet}, …].

    Raises EmailNotConfigured if creds are missing, EmailFetchError on
    connection/login/search failure.
    """
    address = os.environ.get("GMAIL_ADDRESS", "").strip()
    password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    if not address or not password:
        raise EmailNotConfigured(
            "GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set — assistant email is not configured."
        )

    days = max(1, min(int(days or 7), 30))
    max_results = max(1, min(int(max_results or 25), 50))
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%d-%b-%Y")

    conn: imaplib.IMAP4_SSL | None = None
    try:
        # Socket timeout: without it the stdlib default is block-forever, and
        # a wedged connection here means dead air on a live call (the caller
        # asked for their email summary and nothing ever comes back).
        conn = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT, timeout=10)
        conn.login(address, password)
        conn.select(mailbox, readonly=True)
        typ, data = conn.search(None, f'(SINCE {since})')
        if typ != "OK":
            raise EmailFetchError(f"IMAP search failed: {typ}")
        ids = data[0].split()
        if not ids:
            return []
        ids = ids[-max_results:]  # newest are last; cap

        out: list[dict] = []
        for msg_id in reversed(ids):  # newest first
            typ, msg_data = conn.fetch(msg_id, "(RFC822)")
            if typ != "OK" or not msg_data or not isinstance(msg_data[0], tuple):
                continue
            msg = email.message_from_bytes(msg_data[0][1])
            date_raw = msg.get("Date", "")
            try:
                dt = email.utils.parsedate_to_datetime(date_raw)
                date_str = dt.strftime("%a %b %d %H:%M") if dt else date_raw
            except Exception:
                date_str = date_raw
            out.append({
                "from": _decode(msg.get("From")),
                "subject": _decode(msg.get("Subject")) or "(no subject)",
                "date": date_str,
                "snippet": _snippet(msg),
            })
        return out
    except imaplib.IMAP4.error as exc:
        raise EmailFetchError(f"Gmail IMAP error: {exc}") from exc
    except OSError as exc:
        raise EmailFetchError(f"Could not reach Gmail IMAP: {exc}") from exc
    finally:
        if conn is not None:
            try:
                conn.logout()
            except Exception:
                pass
