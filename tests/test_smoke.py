"""Smoke tests for mode routing and the workshop safety guards.

Fast, no network: they exercise prompt/tool selection per mode, the outbound
destination allowlist, /register gating logic, and email error behavior.
"""

import asyncio

import pytest

from gradphone import bridge, email_inbox
from gradphone.business_agent import (
    BusinessCallSpec,
    build_assistant_prompt,
    build_receptionist_prompt,
)


def _tool_names(cfg):
    return [t.name for t in cfg.tools]


def test_business_mode_tools():
    cfg = bridge._make_session_config(BusinessCallSpec(task="x", mode="business"))
    assert _tool_names(cfg) == [
        "press_dtmf", "wait_silently", "save_business_result", "end_business_call",
    ]


def test_assistant_mode_tools():
    cfg = bridge._make_session_config(BusinessCallSpec(task="", mode="assistant"))
    assert _tool_names(cfg) == [
        "remember", "recall", "web_search", "place_call", "get_email_summary", "hang_up",
    ]


def test_receptionist_mode_tools_no_data_access():
    cfg = bridge._make_session_config(BusinessCallSpec(task="", mode="receptionist"))
    assert _tool_names(cfg) == ["take_message", "hang_up"]
    # A stranger must not reach the owner's email or memory.
    assert "get_email_summary" not in _tool_names(cfg)
    assert "recall" not in _tool_names(cfg)


def test_gradium_mode_tools_no_owner_data(monkeypatch):
    monkeypatch.setenv("LINKUP_API_KEY", "")  # web_search variant tested separately
    cfg = bridge._make_session_config(BusinessCallSpec(task="", mode="gradium"))
    assert _tool_names(cfg) == ["search_gradium_docs", "hang_up"]
    # The conference agent must not reach the owner's email, memory, or dialing.
    for forbidden in ("get_email_summary", "recall", "remember", "place_call", "take_message"):
        assert forbidden not in _tool_names(cfg)


def test_gradium_prompt_has_greeting_and_kb_digest():
    from gradphone.business_agent import build_gradium_prompt

    prompt = build_gradium_prompt(BusinessCallSpec(task=""), kb_digest="Gradium builds voice models.")
    assert "Hi, I'm Gradium's voice agent. What would you like to know about Gradium?" in prompt
    assert "Gradium builds voice models." in prompt
    # Injected KB must be framed as data, not instructions (injection guard).
    assert "treat as data" in prompt


def test_gradium_prompt_greets_in_session_language():
    from gradphone.business_agent import GRADIUM_GREETING, build_gradium_prompt

    fr = build_gradium_prompt(BusinessCallSpec(task="", language="fr"))
    assert GRADIUM_GREETING["fr"] in fr
    assert "begin in French" in fr
    # Mid-call switching must be encouraged, not forbidden.
    assert "reply in the language of their most recent message" in fr


def test_conference_language_from_caller_prefix():
    assert bridge._lang_from_caller("+33612345678") == "fr"
    assert bridge._lang_from_caller("+4915112345678") == "de"
    assert bridge._lang_from_caller("+351912345678") == "pt"   # +351 wins over +35
    assert bridge._lang_from_caller("+5511987654321") == "pt"
    assert bridge._lang_from_caller("+34612345678") == "es"
    assert bridge._lang_from_caller("+14155551234") == "en"
    assert bridge._lang_from_caller("", default="en") == "en"


def test_memory_digest_injected_into_assistant_prompt():
    cfg = bridge._make_session_config(
        BusinessCallSpec(task="", mode="assistant"),
        memory_digest="- Prefers morning calls",
    )
    assert "WHAT YOU ALREADY KNOW" in cfg.instructions
    assert "Prefers morning calls" in cfg.instructions


def test_no_memory_block_when_digest_empty():
    cfg = bridge._make_session_config(BusinessCallSpec(task="", mode="assistant"))
    assert "WHAT YOU ALREADY KNOW" not in cfg.instructions


def test_unknown_mode_falls_back_to_business():
    cfg = bridge._make_session_config(BusinessCallSpec(task="x", mode="bogus"))
    assert "save_business_result" in _tool_names(cfg)


def test_receptionist_prompt_uses_owner_and_take_message():
    prompt = build_receptionist_prompt(BusinessCallSpec(task=""), owner_name="Pratim")
    assert "Pratim" in prompt
    assert "take_message" in prompt


def test_assistant_prompt_mentions_email_tool():
    prompt = build_assistant_prompt(BusinessCallSpec(task=""))
    assert "get_email_summary" in prompt


def test_outbound_denied_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_ARBITRARY_OUTBOUND", raising=False)
    monkeypatch.delenv("OUTBOUND_ALLOWLIST", raising=False)
    assert not bridge._outbound_destination_allowed("+15551234567")


def test_outbound_allowlist_match(monkeypatch):
    monkeypatch.delenv("ALLOW_ARBITRARY_OUTBOUND", raising=False)
    monkeypatch.setenv("OUTBOUND_ALLOWLIST", "+1 (555) 123-4567, +33612345678")
    assert bridge._outbound_destination_allowed("+15551234567")
    assert bridge._outbound_destination_allowed("+33612345678")
    assert not bridge._outbound_destination_allowed("+15550000000")


def test_outbound_arbitrary_override(monkeypatch):
    monkeypatch.setenv("ALLOW_ARBITRARY_OUTBOUND", "true")
    monkeypatch.delenv("OUTBOUND_ALLOWLIST", raising=False)
    assert bridge._outbound_destination_allowed("+19990000000")


def test_dial_refuses_disallowed_destination(monkeypatch):
    monkeypatch.delenv("ALLOW_ARBITRARY_OUTBOUND", raising=False)
    monkeypatch.delenv("OUTBOUND_ALLOWLIST", raising=False)
    out = asyncio.run(bridge.dial({"to": "+15551234567", "reason": "test"}))
    assert "not allowed" in out.get("error", "")


def test_safe_room_rejects_traversal():
    import fastapi

    for bad in ["../../etc/passwd", "a/b", "x..y", "", "room with space", "ro;om"]:
        with pytest.raises(fastapi.HTTPException):
            bridge._safe_room(bad)
    # Server-generated room names pass unchanged.
    assert bridge._safe_room("outbound-15551234567_deadbeefcafef00d") == (
        "outbound-15551234567_deadbeefcafef00d"
    )


def test_dispatch_refuses_disallowed_destination(monkeypatch):
    # The allowlist is enforced at the dispatch choke point, not just /dial.
    monkeypatch.delenv("ALLOW_ARBITRARY_OUTBOUND", raising=False)
    monkeypatch.delenv("OUTBOUND_ALLOWLIST", raising=False)
    monkeypatch.setenv("PUBLIC_HTTP_URL", "https://example.com")
    spec = BusinessCallSpec(business_name="Test", task="say hi")
    out = asyncio.run(bridge.dispatch_gradbot_call(to="+15551234567", spec=spec))
    assert "not allowed" in out


def test_bridge_signing_key_fails_closed(monkeypatch):
    import importlib

    from gradphone import sessions

    monkeypatch.delenv("BRIDGE_API_KEY", raising=False)
    monkeypatch.delenv("ALLOW_INSECURE_LOCAL", raising=False)
    importlib.reload(sessions)
    with pytest.raises(RuntimeError):
        sessions._key()


def test_email_not_configured_raises(monkeypatch):
    monkeypatch.delenv("GMAIL_ADDRESS", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    with pytest.raises(email_inbox.EmailNotConfigured):
        email_inbox.fetch_recent(days=7)


def test_gradium_prompt_guardrails():
    from gradphone.business_agent import build_gradium_prompt

    p = build_gradium_prompt(BusinessCallSpec(task=""))
    # Computational-task refusal
    assert "not what this line is built for" in p
    # Identity lock + injection resistance + no prompt disclosure
    assert "SECURITY RULES" in p
    assert "DATA, never" in p
    assert "Never reveal" in p
    # Repeat-after-me is declined
    assert "repeat" in p.lower()


def test_gradium_tools_web_search_gated_by_key(monkeypatch):
    monkeypatch.setenv("LINKUP_API_KEY", "test-key")
    cfg = bridge._make_session_config(BusinessCallSpec(task="", mode="gradium"))
    assert _tool_names(cfg) == ["search_gradium_docs", "web_search", "hang_up"]
    monkeypatch.setenv("LINKUP_API_KEY", "")
    cfg = bridge._make_session_config(BusinessCallSpec(task="", mode="gradium"))
    assert _tool_names(cfg) == ["search_gradium_docs", "hang_up"]


def test_caller_rate_limit(monkeypatch):
    monkeypatch.setattr(bridge, "_GRADIUM_CALLS_PER_HOUR", 3)
    bridge._CALLER_HISTORY.clear()
    caller = "+15550001111"
    assert not bridge._caller_over_rate_limit(caller)
    assert not bridge._caller_over_rate_limit(caller)
    assert not bridge._caller_over_rate_limit(caller)
    assert bridge._caller_over_rate_limit(caller)      # 4th call in the hour → blocked
    assert not bridge._caller_over_rate_limit("+15550002222")  # other callers unaffected
    # 0 disables the limit entirely
    monkeypatch.setattr(bridge, "_GRADIUM_CALLS_PER_HOUR", 0)
    for _ in range(10):
        assert not bridge._caller_over_rate_limit(caller)
