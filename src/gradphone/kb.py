"""Keyword search over the scraped Gradium knowledge base.

The corpus is baked into the repo at data/gradium_kb.json by
scripts/scrape_gradium.py (marketing pages + every blog post, chunked to
~1.1KB). At first use it is loaded into an in-memory SQLite FTS5 index —
no network, no embeddings, no extra dependencies; BM25 keyword ranking is
plenty for spoken questions ("what's your pricing", "how fast is the TTS"),
and it can never add more than ~a millisecond to a live call turn.

Used by the conference "gradium" call mode's search_gradium_docs tool.
"""

from __future__ import annotations

import functools
import json
import re
import sqlite3
from pathlib import Path

_KB_PATH = Path(__file__).parent / "data" / "gradium_kb.json"

# Cap the text a single tool call can inject into a live call's LLM context.
_MAX_CHUNK_CHARS = 1200


def available() -> bool:
    return _KB_PATH.exists()


@functools.lru_cache(maxsize=1)
def _index() -> sqlite3.Connection:
    data = json.loads(_KB_PATH.read_text(encoding="utf-8"))
    # check_same_thread=False: the connection is only used from the event
    # loop's thread, but FastAPI/pytest may create it from a different thread
    # than the one that later queries it.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("CREATE VIRTUAL TABLE kb USING fts5(title, text, url UNINDEXED)")
    conn.executemany(
        "INSERT INTO kb (title, text, url) VALUES (?, ?, ?)",
        [(c["title"], c["text"], c["url"]) for c in data["chunks"]],
    )
    conn.commit()
    return conn


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 OR-query (drops all syntax chars)."""
    tokens = re.findall(r"[A-Za-z0-9]{2,}", query)
    return " OR ".join(tokens[:12])


def search(query: str, k: int = 4) -> list[dict]:
    """Top-k chunks by BM25 for a natural-language question.

    Returns [{title, url, text}], best first; [] on no match or empty query.
    """
    q = _fts_query(query or "")
    if not q or not available():
        return []
    rows = _index().execute(
        "SELECT title, url, text FROM kb WHERE kb MATCH ? ORDER BY bm25(kb) LIMIT ?",
        (q, int(k)),
    ).fetchall()
    return [
        {"title": t, "url": u, "text": x[:_MAX_CHUNK_CHARS]}
        for t, u, x in rows
    ]


# Curated top-line facts injected into the conference agent's system prompt so
# the most common questions need zero tool calls. Composed from the scraped
# corpus (see data/gradium_kb.json) — refresh alongside it.
DIGEST = (
    "Gradium builds audio language models — the technological backbone for "
    "voice applications: natural, expressive, ultra-low-latency voice "
    "interactions at scale. Founded by researchers who invented and published "
    "the methods behind most of today's voice and audio models (neural audio "
    "codecs, audio language models). Raised a $70M seed led by FirstMark "
    "Capital and Eurazeo, with participation from DST Global Partners, Eric "
    "Schmidt, Xavier Niel and others.\n"
    "Products: real-time Text-to-Speech and Speech-to-Text APIs; instant "
    "voice cloning from a short sample; Gradium Translate (real-time "
    "speech-to-text AND speech-to-speech translation across English, French, "
    "Spanish, German, Portuguese); Phonon (tiny on-device TTS for edge "
    "devices, consumer apps, NPCs — 1.00% WER on Seed-TTS while smaller than "
    "every comparable model); Gradbot (open framework to build voice agents "
    "in ~50 lines of code).\n"
    "Performance: ranked #1 on Coval's TTS benchmarks (May 2026) across all "
    "latency metrics including P50 time-to-first-audio, with state-of-the-art "
    "word error rate; semantic VAD for turn detection that uses meaning, not "
    "just silence.\n"
    "Pricing: free tier, then plans from $13/month (XS) up to enterprise; "
    "credit-based — TTS is 1 credit per character (~45k characters ≈ 1 hour "
    "of audio), STT 3 credits/second, speech-to-speech translation 30 "
    "credits/second.\n"
    "Used by: Acolad (enterprise interpretation), InteractionLabs/Ongo, RMC "
    "BFM Drive (personalized AI radio), Wonderful's voice agents, and the "
    "Invincible Voice project helping ALS patients speak with their own "
    "voice."
)
