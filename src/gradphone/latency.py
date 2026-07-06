"""Per-turn latency accounting for the cascaded voice pipeline.

The bridge sees four kinds of message cross the gradbot boundary, each
tagged with a ``turn_idx``: the finalized caller transcript (``stt_text``),
the agent's text tokens (``tts_text``), tool calls, and audio frames. By
recording the *wall-clock arrival time* of each — plus the exact duration
the bridge itself measures around a tool call — we can reconstruct the
cascade for every turn without any timing hooks inside gradbot's core:

    caller stops speaking            (media clock: stt.stop_s)
      │  STT + endpointing
      ▼
    final transcript arrives         (wall: on_stt)
      │  LLM (thinking / first token)
      ▼
    first agent text token           (wall: on_agent_text)
      │  [tool call, if any — measured exactly by the bridge]
      │  TTS (time to first audio)
      ▼
    first audio frame                (wall: on_first_audio)

    total response = transcript → first audio (the silence the caller hears)

Everything here is pure and synchronous so it can be unit-tested with
synthetic event sequences. Deltas that come out negative or implausibly
large (clock skew, dropped events, media/wall drift on long calls) are
discarded rather than reported — a missing stage reads as "—" downstream,
which is honest, where a fabricated number would not be.

STT is the one stage that mixes clocks: ``stt.stop_s`` is gradbot's
media-clock position when the caller stopped speaking, so we convert it to
wall time via the stream-start anchor. It is therefore an approximation and
is labelled as such; LLM, TTS, tool, and total are pure bridge wall-clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Sanity bounds (seconds). A stage delta outside its band is treated as
# unmeasurable for that turn rather than reported.
_MAX_STT_S = 6.0
_MAX_LLM_S = 60.0
_MAX_TTS_S = 30.0
_MAX_RESPONSE_S = 90.0


def _ms(delta_s: float | None, *, lo: float = 0.0, hi: float) -> float | None:
    """Convert a second-delta to rounded ms, or None if out of [lo, hi]."""
    if delta_s is None:
        return None
    if delta_s < lo or delta_s > hi:
        return None
    return round(delta_s * 1000.0, 1)


@dataclass
class _Turn:
    turn_idx: int
    stt_wall: float | None = None      # wall time the final transcript arrived
    stt_stop_s: float | None = None    # media-clock: caller stopped speaking
    first_text_wall: float | None = None
    first_audio_wall: float | None = None
    tools: list[dict] = field(default_factory=list)  # [{name, ms}]


class CallLatency:
    """Accumulates per-turn cascade timings for one call.

    All ``*_wall`` arguments are monotonic seconds (``time.monotonic()``);
    ``stream_started_wall`` is the same clock captured when the media stream
    opened, used to place ``stop_s`` (media clock) on the wall timeline.
    """

    def __init__(self, stream_started_wall: float | None = None) -> None:
        self._stream_start = stream_started_wall
        self._turns: dict[int, _Turn] = {}

    def _turn(self, turn_idx: int | None) -> _Turn:
        # gradbot omits turn_idx on some messages; bucket those as turn -1 so
        # they still contribute rather than being silently dropped.
        key = turn_idx if turn_idx is not None else -1
        t = self._turns.get(key)
        if t is None:
            t = _Turn(turn_idx=key)
            self._turns[key] = t
        return t

    # ── Event hooks (called from the bridge consumer loop) ──────────────

    def on_stt(self, turn_idx: int | None, wall: float, stop_s: float | None) -> None:
        """Final caller transcript for a turn arrived. First one per turn wins."""
        t = self._turn(turn_idx)
        if t.stt_wall is None:
            t.stt_wall = wall
            t.stt_stop_s = stop_s

    def on_agent_text(self, turn_idx: int | None, wall: float) -> None:
        """First agent text token of a turn arrived. First one per turn wins."""
        t = self._turn(turn_idx)
        if t.first_text_wall is None:
            t.first_text_wall = wall

    def on_first_audio(self, turn_idx: int | None, wall: float) -> None:
        """First agent audio frame of a turn arrived. First one per turn wins."""
        t = self._turn(turn_idx)
        if t.first_audio_wall is None:
            t.first_audio_wall = wall

    def on_tool(self, turn_idx: int | None, name: str, duration_ms: float) -> None:
        """A tool call the bridge dispatched for this turn completed."""
        self._turn(turn_idx).tools.append(
            {"name": name, "ms": round(max(0.0, duration_ms), 1)}
        )

    # ── Derivation ──────────────────────────────────────────────────────

    def _turn_row(self, t: _Turn) -> dict:
        stt_ms = None
        if t.stt_wall is not None and t.stt_stop_s is not None and self._stream_start is not None:
            stt_ms = _ms(t.stt_wall - (self._stream_start + t.stt_stop_s), hi=_MAX_STT_S)

        llm_ms = None
        if t.stt_wall is not None and t.first_text_wall is not None:
            llm_ms = _ms(t.first_text_wall - t.stt_wall, hi=_MAX_LLM_S)

        tts_ms = None
        if t.first_text_wall is not None and t.first_audio_wall is not None:
            tts_ms = _ms(t.first_audio_wall - t.first_text_wall, hi=_MAX_TTS_S)

        response_ms = None
        if t.stt_wall is not None and t.first_audio_wall is not None:
            response_ms = _ms(t.first_audio_wall - t.stt_wall, hi=_MAX_RESPONSE_S)

        tool_ms = round(sum(tool["ms"] for tool in t.tools), 1) if t.tools else None

        return {
            "turn": t.turn_idx,
            "stt_ms": stt_ms,
            "llm_ms": llm_ms,
            "tool_ms": tool_ms,
            "tools": list(t.tools),
            "tts_ms": tts_ms,
            "response_ms": response_ms,
        }

    def summary(self) -> dict:
        """Return per-turn rows + aggregate stats, JSON-safe.

        ``turns`` is ordered by turn index (the catch-all -1 bucket, if any,
        sorts first). ``aggregates`` gives median/p95/max per stage over the
        turns where that stage was measurable.
        """
        rows = [self._turn_row(t) for _, t in sorted(self._turns.items())]
        # Only real conversational turns (with a response) count toward the
        # headline aggregates; a stray tool-only bucket shouldn't skew them.
        agg = {
            stage: _stats([r[key] for r in rows])
            for stage, key in (
                ("stt", "stt_ms"),
                ("llm", "llm_ms"),
                ("tool", "tool_ms"),
                ("tts", "tts_ms"),
                ("response", "response_ms"),
            )
        }
        return {
            "turns": rows,
            "aggregates": agg,
            "turn_count": sum(1 for r in rows if r["response_ms"] is not None),
        }


def _stats(values: list[float | None]) -> dict | None:
    """median / p95 / max over the non-None values, or None if empty."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    return {
        "median": _percentile(xs, 50),
        "p95": _percentile(xs, 95),
        "max": xs[-1],
        "n": len(xs),
    }


def _percentile(sorted_xs: list[float], pct: float) -> float:
    """Nearest-rank percentile of a pre-sorted, non-empty list."""
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    # Nearest-rank: rank = ceil(pct/100 * n), 1-indexed.
    import math

    rank = max(1, math.ceil(pct / 100.0 * len(sorted_xs)))
    return sorted_xs[min(rank, len(sorted_xs)) - 1]
