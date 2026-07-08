"""Tests for the per-turn latency accumulator (pure, no gradbot/network)."""

from gradphone.latency import CallLatency


def test_full_cascade_single_turn():
    # stream opened at wall=100.0; caller stops speaking at media t=2.0s
    lat = CallLatency(stream_started_wall=100.0)
    # transcript arrives at wall=102.4 → STT = 102.4 - (100.0 + 2.0) = 0.4s
    lat.on_stt(turn_idx=1, wall=102.4, stop_s=2.0)
    # first agent text at wall=103.1; audio at 103.5 → first output = 103.1
    lat.on_agent_text(turn_idx=1, wall=103.1)
    lat.on_first_audio(turn_idx=1, wall=103.5)

    row = lat.summary()["turns"][0]
    assert row["stt_ms"] == 400.0
    # LLM / response = transcript → first agent output (103.1 - 102.4 = 0.7s)
    assert row["llm_ms"] == 700.0
    assert row["response_ms"] == 700.0
    assert row["tool_ms"] is None


def test_response_uses_earliest_output_when_audio_precedes_text():
    # Real gradbot: the audio frame can arrive before the tts_text event.
    lat = CallLatency(stream_started_wall=0.0)
    lat.on_stt(turn_idx=1, wall=10.0, stop_s=None)
    lat.on_first_audio(turn_idx=1, wall=11.0)   # audio first
    lat.on_agent_text(turn_idx=1, wall=11.4)    # text slightly after
    row = lat.summary()["turns"][0]
    assert row["response_ms"] == 1000.0         # earliest output wins, never negative


def test_partial_transcripts_coalesce_into_one_turn():
    # gradbot streams partial transcripts before the agent responds; they are
    # one caller turn, not many.
    lat = CallLatency(stream_started_wall=0.0)
    lat.on_stt(turn_idx=0, wall=5.0, stop_s=None)   # "the"
    lat.on_stt(turn_idx=1, wall=5.3, stop_s=None)   # "the weather"
    lat.on_stt(turn_idx=2, wall=5.6, stop_s=None)   # "the weather in Paris"
    lat.on_agent_text(turn_idx=3, wall=6.1)
    lat.on_first_audio(turn_idx=3, wall=6.2)
    s = lat.summary()
    assert s["turn_count"] == 1                      # not 3
    # transcript advanced to the latest partial (5.6): 6.1 - 5.6 = 0.5s
    assert s["turns"][0]["response_ms"] == 500.0


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
    # first output is the text at 5.5 (earlier than audio at 5.8)
    assert row["llm_ms"] == 500.0
    assert row["response_ms"] == 500.0


def test_oversized_deltas_dropped():
    # An LLM gap of 10 minutes is implausible (clock issue) → dropped.
    lat = CallLatency(stream_started_wall=0.0)
    lat.on_stt(turn_idx=1, wall=0.0, stop_s=None)
    lat.on_agent_text(turn_idx=1, wall=600.0)
    assert lat.summary()["turns"][0]["llm_ms"] is None


def test_first_event_per_turn_wins():
    lat = CallLatency(stream_started_wall=0.0)
    lat.on_stt(turn_idx=1, wall=4.5, stop_s=None)
    lat.on_agent_text(turn_idx=2, wall=5.0)   # different turn_idx (real gradbot)
    lat.on_agent_text(turn_idx=2, wall=6.0)   # later token, ignored
    lat.on_first_audio(turn_idx=2, wall=5.3)
    lat.on_first_audio(turn_idx=2, wall=9.9)  # later frame, ignored
    row = lat.summary()["turns"][0]
    assert row["response_ms"] == 500.0


def test_sequence_pairing_across_mismatched_turn_idx():
    # The bug the live test caught: caller stt and agent response carry
    # DIFFERENT turn_idx, so pairing must be by sequence, not by index.
    lat = CallLatency(stream_started_wall=100.0)
    lat.on_stt(turn_idx=3, wall=102.0, stop_s=2.0)   # caller turn 3
    lat.on_agent_text(turn_idx=4, wall=102.6)         # agent replies as turn 4
    lat.on_first_audio(turn_idx=4, wall=103.0)
    row = lat.summary()["turns"][0]
    assert row["stt_ms"] == 0.0        # 102.0 - (100.0 + 2.0)
    assert row["llm_ms"] == 600.0
    assert row["response_ms"] == 600.0


def test_opener_before_any_caller_turn_is_ignored():
    # Agent speaks first (the greeting) — no caller turn open, so it must not
    # be counted as a response.
    lat = CallLatency(stream_started_wall=0.0)
    lat.on_agent_text(turn_idx=0, wall=1.0)
    lat.on_first_audio(turn_idx=0, wall=1.4)
    assert lat.summary()["turn_count"] == 0


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
