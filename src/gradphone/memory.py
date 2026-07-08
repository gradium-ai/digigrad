"""Per-tenant agent memory — the layer that makes the clone *remember you*.

Deliberately small and self-contained so it can be swapped for an external
user-modeling service (e.g. Honcho) later without touching callers: the bridge
only uses add_memory / search / render_digest. Backed by the existing SQLite
DB (the `memories` table created in tenants.py).

A "memory" is one short durable fact about a tenant ("prefers morning calls",
"dentist is Dr. Lemoine"), written either by the agent mid-call (remember tool)
or by the post-call extraction pass.
"""

from __future__ import annotations

import json
import logging
import os

from . import db
from .tenants import _now

log = logging.getLogger(__name__)

_MAX_FACT_LEN = 400
_DIGEST_LIMIT = 25
_EXTRACT_MAX_FACTS = 5


async def add_memory(tenant_id: int, fact: str, *, source: str = "", room: str = "") -> bool:
    """Store one durable fact for a tenant. Returns False on empty/duplicate."""
    fact = (fact or "").strip()[:_MAX_FACT_LEN]
    if not fact:
        return False
    # Skip an exact duplicate — cheap guard against the model re-remembering
    # the same thing in one call.
    dup = await db.fetch_one(
        "SELECT 1 AS x FROM memories WHERE tenant_id = :tid AND fact = :fact LIMIT 1",
        tid=tenant_id, fact=fact,
    )
    if dup:
        return False
    await db.execute(
        "INSERT INTO memories (tenant_id, fact, source, room, created_at) "
        "VALUES (:tid, :fact, :source, :room, :now)",
        tid=tenant_id, fact=fact, source=source or None, room=room or None, now=_now(),
    )
    return True


async def clear_memories(tenant_id: int) -> int:
    """Delete every stored fact for a tenant. Returns the count removed."""
    rows = await db.fetch_all(
        "SELECT COUNT(*) AS n FROM memories WHERE tenant_id = :tid", tid=tenant_id
    )
    n = int(rows[0]["n"]) if rows else 0
    await db.execute("DELETE FROM memories WHERE tenant_id = :tid", tid=tenant_id)
    return n


async def get_memories(tenant_id: int, limit: int = _DIGEST_LIMIT) -> list[str]:
    """Most-recent facts first."""
    rows = await db.fetch_all(
        "SELECT fact FROM memories WHERE tenant_id = :tid "
        "ORDER BY created_at DESC LIMIT :lim",
        tid=tenant_id, lim=max(1, limit),
    )
    return [r["fact"] for r in rows]


async def search_memories(tenant_id: int, query: str, limit: int = 10) -> list[str]:
    """Substring search across a tenant's facts (case-insensitive). Falls back
    to recent facts when the query is empty."""
    query = (query or "").strip()
    if not query:
        return await get_memories(tenant_id, limit)
    # LOWER(...) LIKE LOWER(...) is case-insensitive on both SQLite and Postgres
    # (SQLite's bare LIKE is already case-insensitive; Postgres's is not).
    rows = await db.fetch_all(
        "SELECT fact FROM memories WHERE tenant_id = :tid "
        "AND LOWER(fact) LIKE LOWER(:q) ORDER BY created_at DESC LIMIT :lim",
        tid=tenant_id, q=f"%{query}%", lim=max(1, limit),
    )
    return [r["fact"] for r in rows]


async def render_digest(tenant_id: int, limit: int = _DIGEST_LIMIT) -> str:
    """A compact bullet list of what we know, for injection into a prompt.
    Empty string when we know nothing yet (caller injects nothing)."""
    facts = await get_memories(tenant_id, limit)
    if not facts:
        return ""
    return "\n".join(f"- {f}" for f in facts)


_EXTRACT_SYSTEM = (
    "You read a phone-call transcript between a user and their personal "
    "assistant. Extract durable facts about the USER worth remembering for "
    "future calls — preferences, names, relationships, plans, recurring needs. "
    "Ignore one-off call logistics and anything the assistant said about itself. "
    f"Return a JSON array of at most {_EXTRACT_MAX_FACTS} short factual strings "
    "(each a complete sentence). If nothing is worth keeping, return []."
)


async def extract_facts(turns: list[tuple[str, str]]) -> list[str]:
    """Best-effort: ask the configured LLM for durable facts in a transcript.

    Returns [] on any failure (no endpoint, auth, parse) — extraction is a
    bonus on top of the explicit `remember` tool, never required.
    """
    if not turns:
        return []
    base = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("LLM_MODEL", "").strip()
    if not base or not model:
        return []
    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("GRADIUM_API_KEY", "").strip()
    )
    convo = "\n".join(f"{who}: {text}" for who, text in turns)[:8000]

    import aiohttp

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": convo},
        ],
        "temperature": 0,
    }
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{base}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as r:
                if r.status != 200:
                    log.warning("extract_facts: LLM HTTP %s", r.status)
                    return []
                data = await r.json()
        content = data["choices"][0]["message"]["content"]
        facts = _parse_fact_list(content)
        return facts[:_EXTRACT_MAX_FACTS]
    except Exception as exc:  # noqa: BLE001
        log.warning("extract_facts failed: %s", exc)
        return []


def _parse_fact_list(content: str) -> list[str]:
    """Pull a JSON string array out of the model's reply, tolerating fences."""
    content = (content or "").strip()
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        items = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in items if isinstance(x, (str, int, float)) and str(x).strip()]


async def extract_and_store(tenant_id: int, turns: list[tuple[str, str]], room: str = "") -> int:
    """Extract durable facts from a call and persist them. Returns count stored."""
    stored = 0
    for fact in await extract_facts(turns):
        if await add_memory(tenant_id, fact, source="post_call", room=room):
            stored += 1
    return stored
