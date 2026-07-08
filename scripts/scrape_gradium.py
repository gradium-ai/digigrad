#!/usr/bin/env python3
"""Scrape gradium.ai into the conference agent's knowledge base.

Fetches the marketing pages + every blog post, strips chrome (nav/footer/
scripts), and writes paragraph-aligned ~1.1KB chunks to
src/gradphone/data/gradium_kb.json — which is committed, so deploys carry
the KB without any runtime scraping. Re-run this script to refresh it.

Stdlib only. Usage:  python3 scripts/scrape_gradium.py
"""

from __future__ import annotations

import html as html_mod
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://gradium.ai"
# Marketing/docs pages (blog posts are discovered from /blog).
SEED_PATHS = [
    "/",
    "/pricing",
    "/translate",
    "/on-device-tts",
    "/gradbot",
    "/gallery",
    "/demo",
    "/press",
    "/about/who-we-are",
    "/about/careers",
    "/api_docs.html",
    "/blog",
]
UA = "Mozilla/5.0 (compatible; gradphone-kb/1.0; +https://gradium.ai)"
OUT = Path(__file__).resolve().parent.parent / "src" / "gradphone" / "data" / "gradium_kb.json"

CHUNK_TARGET = 1100   # chars per chunk (paragraph-aligned)
CHUNK_MIN = 200       # drop fragments smaller than this

# Nav/footer lines that repeat on every page — noise for retrieval.
_BOILER = re.compile(
    r"(?i)^(models?|showcase|pricing|about us|start free|product|on-device|"
    r"api docs|agent demo|gradbot|app gallery|who we are|blog|careers|contact|"
    r"press|socials|x|github|linkedin|discord|← back to blog|check it out.*|"
    r"introducing translate.*|terms of service|privacy policy|"
    r"© \d{4} gradium.*|\d+:\d+ / \d+:\d+)$"
)


def fetch(url: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        print(f"  !! {url}: {e}", file=sys.stderr)
        return None


def page_title(raw: str) -> str:
    m = re.search(r'<meta property="og:title" content="([^"]+)"', raw)
    if not m:
        m = re.search(r"<h1[^>]*>(.*?)</h1>", raw, re.S)
    if not m:
        m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S)
    t = html_mod.unescape(re.sub(r"<[^>]+>", " ", m.group(1))) if m else ""
    return re.sub(r"\s+", " ", t).strip()[:120]


def extract_text(raw: str) -> str:
    """HTML → readable text with paragraph breaks, chrome removed."""
    # Drop whole non-content elements first.
    raw = re.sub(
        r"<(script|style|noscript|nav|header|footer|svg|form)[^>]*>.*?</\1>",
        " ", raw, flags=re.S | re.I,
    )
    # Block-level closings become paragraph breaks so chunking can align.
    raw = re.sub(r"</(p|h[1-6]|li|blockquote|pre|tr|section|article|div)>", "\n", raw, flags=re.I)
    raw = re.sub(r"<br\s*/?>", "\n", raw, flags=re.I)
    text = html_mod.unescape(re.sub(r"<[^>]+>", " ", raw))
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line or _BOILER.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def chunk(text: str) -> list[str]:
    """Greedy paragraph-aligned chunks of ~CHUNK_TARGET chars."""
    out, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 1 > CHUNK_TARGET and len(cur) >= CHUNK_MIN:
            out.append(cur.strip())
            cur = ""
        cur += para + "\n"
    if len(cur.strip()) >= CHUNK_MIN:
        out.append(cur.strip())
    return out


def blog_post_paths(blog_index_raw: str) -> list[str]:
    hrefs = set(re.findall(r'href="(/blog/[a-z0-9-]+)"', blog_index_raw))
    return sorted(h for h in hrefs if not h.endswith((".xml", "/feed")))


def main() -> None:
    chunks: list[dict] = []
    seen_urls: set[str] = set()
    paths = list(SEED_PATHS)

    i = 0
    while i < len(paths):
        path = paths[i]
        i += 1
        url = BASE + ("" if path == "/" else path)
        if url in seen_urls:
            continue
        seen_urls.add(url)
        raw = fetch(url)
        if raw is None:
            continue
        if path == "/blog":
            posts = blog_post_paths(raw)
            print(f"blog index: {len(posts)} posts")
            paths.extend(posts)
            continue
        title = page_title(raw) or path
        text = extract_text(raw)
        pieces = chunk(text)
        print(f"{path}: {len(text)} chars -> {len(pieces)} chunks | {title[:60]}")
        for piece in pieces:
            chunks.append({"url": url, "title": title, "text": piece})
        time.sleep(0.3)  # be polite

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "source": BASE,
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chunks": chunks,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(chunks)} chunks from {len(seen_urls)} urls -> {OUT}")


if __name__ == "__main__":
    main()
