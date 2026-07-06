"""Tests for the per-turn latency accumulator (pure, no gradbot/network)."""

from gradphone.latency import CallLatency


def test_full_cascade_single_turn():
    # stream opened at wall=100.0; caller stops speaking at media t=2.0s
    lat = CallLatency(stream_started_wall=100.0)
    # transcript arrives at wall=102.4 → STT = 102.4 - (100.0 + 2.0) = 0.4s
    lat.on_stt(turn_idx=1, wall=102.4, stop_s=2.0)
    # first agent text at wall=103.1 → LLM = 0.7s
    lat.on_agent_text(turn_idx=1, wall=103.1)
    # first audio at wall=103.5 → TTS = 0.4s ; total = 103.5-102.4 = 1.1s
    lat.on_first_audio(turn_idx=1, wall=103.5)

    row = lat.summary()["turns"][0]
    assert row["stt_ms"] == 400.0
    assert row["llm_ms"] == 700.0
    assert row["tts_ms"] == 400.0
    assert row["response_ms"] == 1100.0
    assert row["tool_ms"] is None


def test_tool_call_included_in_turn():
    lat = CallLatency(stream_started_wall=0.0)
    lat.on_stt(turn_idx=2, wall=10.0, stop_s=None)
    lat.on_tool(turn_idx=2, name="web_search", duration_ms=1800.0)
    lat.on_tool(turn_idx=2, name="web_search", duration_ms=20.0)
    lat.on_agent_text(turn_idx=2, wall=12.0)
    lat.on_first_audio(turn_idx=2, wall=12.3)

    row = lat.summary()["turns"][0]
    assert row["tool_ms"] == 1820.0
    assert [t["name"] for t in row["tools"]] == ["web_search", "web_search"]
    # LLM includes the tool round-trip here (transcript→first text spans it).
    assert row["llm_ms"] == 2000.0


def test_stt_needs_stop_s_and_anchor():
    # No stop_s → STT unmeasurable, but the rest still computes.
    lat = CallLatency(stream_started_wall=0.0)
    lat.on_stt(turn_idx=1, wall=5.0, stop_s=None)
    lat.on_agent_text(turn_idx=1, wall=5.5)
    lat.on_first_audio(turn_idx=1, wall=5.8)
    row = lat.summary()["turns"][0]
    assert row["stt_ms"] is None
    assert row["llm_ms"] == 500.0
    assert row["response_ms"] == 800.0


def test_negative_and_oversized_deltas_dropped():
    lat = CallLatency(stream_started_wall=0.0)
    # first audio BEFORE first text (clock skew) → TTS dropped, not negative
    lat.on_stt(turn_idx=1, wall=1.0, stop_s=None)
    lat.on_agent_text(turn_idx=1, wall=3.0)
    lat.on_first_audio(turn_idx=1, wall=2.0)
    row = lat.summary()["turns"][0]
    assert row["tts_ms"] is None
    # LLM = 2.0s is fine; response = first_audio - stt = 1.0s is fine
    assert row["llm_ms"] == 2000.0

    # An LLM gap of 10 minutes is implausible → dropped.
    lat2 = CallLatency(stream_started_wall=0.0)
    lat2.on_stt(turn_idx=1, wall=0.0, stop_s=None)
    lat2.on_agent_text(turn_idx=1, wall=600.0)
    assert lat2.summary()["turns"][0]["llm_ms"] is None


def test_first_event_per_turn_wins():
    lat = CallLatency(stream_started_wall=0.0)
    lat.on_agent_text(turn_idx=1, wall=5.0)
    lat.on_agent_text(turn_idx=1, wall=6.0)  # later token, ignored
    lat.on_first_audio(turn_idx=1, wall=5.3)
    lat.on_first_audio(turn_idx=1, wall=9.9)  # later frame, ignored
    row = lat.summary()["turns"][0]
    assert row["tts_ms"] == 300.0


def test_aggregates_median_p95_over_turns():
    lat = CallLatency(stream_started_wall=0.0)
    for i, (stt_w, txt_w) in enumerate([(0.0, 0.2), (1.0, 1.4), (2.0, 3.0)], start=1):
        lat.on_stt(turn_idx=i, wall=stt_w, stop_s=None)
        lat.on_agent_text(turn_idx=i, wall=txt_w)
        lat.on_first_audio(turn_idx=i, wall=txt_w + 0.1)
    agg = lat.summary()["aggregates"]
    # LLM deltas: 200, 400, 1000 ms → median 400, max 1000
    assert agg["llm"]["median"] == 400.0
    assert agg["llm"]["max"] == 1000.0
    assert agg["llm"]["n"] == 3
    assert lat.summary()["turn_count"] == 3


def test_empty_call_has_no_aggregates():
    lat = CallLatency(stream_started_wall=0.0)
    s = lat.summary()
    assert s["turns"] == []
    assert s["turn_count"] == 0
    assert s["aggregates"]["llm"] is None
