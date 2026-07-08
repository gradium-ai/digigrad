"""Tests for the Gradium knowledge-base search (offline, uses baked JSON)."""

from gradphone import kb


def test_kb_available():
    assert kb.available()


def test_search_finds_latency_post():
    hits = kb.search("time to first audio latency")
    assert hits, "expected hits for TTFA query"
    assert any("time-to-first-audio" in h["url"] or "latency" in h["title"].lower()
               for h in hits)


def test_search_finds_pricing():
    hits = kb.search("how much does it cost pricing plans")
    assert any("/pricing" in h["url"] for h in hits)


def test_search_finds_founders():
    hits = kb.search("who founded Gradium seed round investors")
    assert hits
    joined = " ".join(h["text"] for h in hits)
    assert "FirstMark" in joined or "founders" in joined.lower()


def test_search_result_shape_and_cap():
    hits = kb.search("voice cloning", k=3)
    assert 0 < len(hits) <= 3
    for h in hits:
        assert set(h) == {"title", "url", "text"}
        assert len(h["text"]) <= 1200


def test_empty_and_junk_queries():
    assert kb.search("") == []
    assert kb.search("???!!!") == []


def test_digest_mentions_core_products():
    for term in ("Text-to-Speech", "Translate", "cloning"):
        assert term in kb.DIGEST
