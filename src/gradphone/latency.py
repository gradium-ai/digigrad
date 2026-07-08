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

    def has_response(self) -> bool:
        return self.first_text_wall is not None or self.first_audio_wall is not None


class CallLatency:
    """Accumulates per-turn cascade timings for one call.

    Pairing is by SEQUENCE, not by turn_idx: gradbot numbers the caller's
    transcript and the agent's response with different turn_idx values, so a
    caller transcript OPENS a turn and the next agent text/audio CLOSES it.
    turn_idx is kept only as a display label.

    All ``*_wall`` arguments are monotonic seconds (``time.monotonic()``);
    ``stream_started_wall`` is the same clock captured when the media stream
    opened, used to place ``stop_s`` (media clock) on the wall timeline.
    """

    def __init__(self, stream_started_wall: float | None = None) -> None:
        self._stream_start = stream_started_wall
        self._turns: list[_Turn] = []   # closed turns, in order
        self._cur: _Turn | None = None  # open turn awaiting the agent's response

    # ── Event hooks (called from the bridge consumer loop) ──────────────

    def on_stt(self, turn_idx: int | None, wall: float, stop_s: float | None) -> None:
        """A caller transcript arrived.

        gradbot streams partial transcripts (one per word-ish), so several
        arrive before the agent responds — they're all the SAME caller turn.
        Coalesce: while the current turn has no agent response yet, keep
        advancing its transcript timestamp (the latest partial is closest to
        when the caller actually stopped). Only once the agent has responded
        does a new transcript open a fresh turn."""
        if self._cur is not None and not self._cur.has_response():
            self._cur.stt_wall = wall
            if stop_s is not None:
                self._cur.stt_stop_s = stop_s
            return
        if self._cur is not None:
            self._turns.append(self._cur)
        self._cur = _Turn(
            turn_idx=turn_idx if turn_idx is not None else len(self._turns),
            stt_wall=wall, stt_stop_s=stop_s,
        )

    def on_agent_text(self, turn_idx: int | None, wall: float) -> None:
        """First agent text token of the response — closes the LLM stage.
        Ignored when no caller turn is open (e.g. the greeting/opener)."""
        if self._cur is not None and self._cur.first_text_wall is None:
            self._cur.first_text_wall = wall

    def on_first_audio(self, turn_idx: int | None, wall: float) -> None:
        """First agent audio frame of the response — closes TTS + the turn.
        Ignored when no caller turn is open (the agent-speaks-first opener)."""
        if self._cur is not None and self._cur.first_audio_wall is None:
            self._cur.first_audio_wall = wall

    def on_tool(self, turn_idx: int | None, name: str, duration_ms: float) -> None:
        """A tool call the bridge dispatched completed — attach to the open turn."""
        target = self._cur if self._cur is not None else (self._turns[-1] if self._turns else None)
        if target is not None:
            target.tools.append({"name": name, "ms": round(max(0.0, duration_ms), 1)})

    # ── Derivation ──────────────────────────────────────────────────────

    def _turn_row(self, t: _Turn) -> dict:
        stt_ms = None
        if t.stt_wall is not None and t.stt_stop_s is not None and self._stream_start is not None:
            stt_ms = _ms(t.stt_wall - (self._stream_start + t.stt_stop_s), hi=_MAX_STT_S)

        # First agent output of the turn — gradbot may emit the audio frame
        # slightly before or after the tts_text event, so take whichever came
        # first as the moment the caller starts hearing a response.
        outputs = [w for w in (t.first_text_wall, t.first_audio_wall) if w is not None]
        first_output = min(outputs) if outputs else None

        # LLM "think" time: transcript → first output. This is the model's
        # time-to-first-token from the caller's perspective (includes any tool
        # round-trip in the same turn).
        llm_ms = None
        if t.stt_wall is not None and first_output is not None:
            llm_ms = _ms(first_output - t.stt_wall, hi=_MAX_LLM_S)

        # Total response gap: the silence the caller hears (== llm here, since
        # first output is when audio starts; kept as its own field for the HUD).
        response_ms = None
        if t.stt_wall is not None and first_output is not None:
            response_ms = _ms(first_output - t.stt_wall, hi=_MAX_RESPONSE_S)

        tool_ms = round(sum(tool["ms"] for tool in t.tools), 1) if t.tools else None

        return {
            "turn": t.turn_idx,
            "stt_ms": stt_ms,
            "llm_ms": llm_ms,
            "tool_ms": tool_ms,
            "tools": list(t.tools),
            "response_ms": response_ms,
        }

    def summary(self) -> dict:
        """Return per-turn rows + aggregate stats, JSON-safe.

        ``turns`` is ordered by turn index (the catch-all -1 bucket, if any,
        sorts first). ``aggregates`` gives median/p95/max per stage over the
        turns where that stage was measurable.
        """
        all_turns = self._turns + ([self._cur] if self._cur is not None else [])
        rows = [self._turn_row(t) for t in all_turns]
        # Only real conversational turns (with a response) count toward the
        # headline aggregates; a stray tool-only bucket shouldn't skew them.
        agg = {
            stage: _stats([r[key] for r in rows])
            for stage, key in (
                ("stt", "stt_ms"),
                ("llm", "llm_ms"),
                ("tool", "tool_ms"),
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
