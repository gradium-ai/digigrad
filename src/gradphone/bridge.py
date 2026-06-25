"""Gradbot-backed outbound caller — substitute for the LiveKit SIP path.

Architecture (mirrors `gradium-ai/ticatag`'s Twilio bridge, adapted for
outbound):

    [make_call(orchestrator='gradbot')]
                  │
                  │ Twilio REST: client.calls.create(
                  │     to=…, from_=TWILIO_PHONE_NUMBER,
                  │     url=PUBLIC_HTTP_URL/twilio/voice?room=ROOM)
                  ▼
    [Twilio dials the destination]
                  │
                  │ Pickup → Twilio fetches TwiML from /twilio/voice
                  ▼
    [/twilio/voice returns <Connect><Stream url="…/twilio/stream?room=ROOM"/>]
                  │
                  ▼
    [Twilio opens Media Streams WS to /twilio/stream]
                  │
                  │ μ-law 8 kHz audio in JSON frames, base64-encoded
                  ▼
    [/twilio/stream bridge calls gradbot.run() and shuttles audio]

The gradbot session is configured with the same business prompt + tools
as the LiveKit path (`build_business_prompt`, save_business_result, …).

Audio is tee'd to mixed.wav under
~/.openclaw/workspace/call-recordings/<room>/ so the dashboard
auto-discovers the call exactly like LiveKit calls. A `framework.txt`
marker is written so the dashboard tags it as "gradbot" instead of the
default "livekit".

Public URL plumbing: this module needs the Mac to be reachable from
Twilio. Set `PUBLIC_HTTP_URL=https://<your-tunnel>` and
`PUBLIC_WS_URL=wss://<your-tunnel>` in env. Cloudflare tunnel,
ngrok, or any equivalent works.

Run the bridge with:
    uvicorn gradphone.bridge:app --host 127.0.0.1 --port 8082
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Lazy-import gradbot at use time — the rest of the agent must keep
# compiling and running even if gradbot isn't installed.
try:  # pragma: no cover
    import gradbot  # type: ignore
    from gradbot.schemas import sanitize  # type: ignore
    HAS_GRADBOT = True
except ImportError:
    gradbot = None  # type: ignore[assignment]
    sanitize = lambda x: x  # type: ignore[assignment]
    HAS_GRADBOT = False

import fastapi
from fastapi.responses import PlainTextResponse

from . import email_inbox
from . import memory as memory_mod
from . import db as db_mod
from . import tenants
from . import websearch
from .business_agent import (
    BusinessCallSpec,
    agent_name_for_language,
    build_assistant_prompt,
    build_business_prompt,
    build_opener_text,
    build_receptionist_prompt,
    contains_personal_detail,
    redact_personal_details,
)
from .config import cfg
from .voicemail import is_machine, voicemail_twiml

__all__ = ["app", "dispatch_gradbot_call"]

# Route the app's own logs to the platform log stream (Render captures
# stdout/stderr). uvicorn configures only its own loggers; without this the
# bridge's log.info/warning are invisible — which hid the WS rejection below.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger(__name__)

# Recording root — same path the LiveKit path writes to, so the dashboard
# auto-discovery picks both up.
WORKSPACE = Path.home() / ".openclaw" / "workspace"
RECORDINGS_ROOT = WORKSPACE / "call-recordings"
TRANSCRIPTS_ROOT = WORKSPACE / "call-transcripts"

# Voice IDs per language. These are Gradium TTS voice IDs (the same ones
# used by the LiveKit path's VOICE_PRESETS). gradbot's Rust core forwards
# voice_id straight through to Gradium TTS, so non-flagship IDs work as
# long as the LLM doesn't emit mixed-language text — Gradium TTS used to
# auto-swap voices when it detected a language change, which was what
# caused the "voice switching mid-call" symptom under Qwen. With Haiku
# 4.5 holding the language steady, Gradium-side IDs stay pinned.
#   en → Arthur     (LiveKit en preset)
#   fr → Constance  (LiveKit fr preset)
#   pt → Rafael     (LiveKit pt preset)
_VOICE_ID = {
    "en": "3jUdJyOi9pgbxBTK",   # Arthur
    "fr": "Y4iYxS8PBX-bazgX",   # Constance
    "pt": "KpDAXeGeen7P9Uri",   # Rafael
}

# Per-language Lang enum value name in gradbot. Resolved lazily.
_LANG_NAME = {"en": "En", "fr": "Fr", "pt": "Pt"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Backchannel "filler" phrase, synthesized once per voice and replayed during
# the think-gap between the caller finishing and the agent's first audio. See
# the filler block in twilio_stream. Off unless ENABLE_FILLERS is set.
_FILLER_PHRASE = os.environ.get("FILLER_PHRASE", "Mm, let me see.")
_FILLER_DELAY_S = _env_float("FILLER_DELAY_S", 0.7)
_FILLER_CACHE: dict[str, bytes] = {}


async def _render_filler_ulaw(voice_id: str) -> bytes | None:
    """Synthesize the filler phrase in ``voice_id`` and return it as 8 kHz
    μ-law (Twilio's wire format). Cached per voice, so it costs exactly one
    Gradium TTS call per voice for the life of the process. Best-effort —
    returns None on any failure (missing key, TTS error) so a filler never
    breaks a call."""
    if not voice_id:
        return None
    if voice_id in _FILLER_CACHE:
        return _FILLER_CACHE[voice_id]
    try:
        import audioop
        import io
        import wave as wave_mod

        import gradium

        api_key = os.environ.get("GRADIUM_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.environ.get("GRADIUM_BASE_URL", "").strip() or None
        client = (
            gradium.GradiumClient(api_key=api_key, base_url=base_url)
            if base_url else gradium.GradiumClient(api_key=api_key)
        )
        setup = gradium.TTSSetup(model_name="default", voice_id=voice_id, output_format="wav")
        result = await client.tts(setup, _FILLER_PHRASE)
        wav = getattr(result, "raw_data", None)
        if not wav:
            return None
        with wave_mod.open(io.BytesIO(wav), "rb") as w:
            rate, channels = w.getframerate(), w.getnchannels()
            pcm = w.readframes(w.getnframes())
        if channels == 2:
            pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
        if rate != TWILIO_ULAW_RATE:
            pcm, _ = audioop.ratecv(pcm, 2, 1, rate, TWILIO_ULAW_RATE, None)
        ulaw = audioop.lin2ulaw(pcm, 2)
        _FILLER_CACHE[voice_id] = ulaw
        return ulaw
    except Exception as exc:  # noqa: BLE001
        log.warning("filler render failed: %s", exc)
        return None


# --- Tool defs -----------------------------------------------------------
#
# Mirror the LiveKit tools (agent.py:tool_save_business_result, etc.) with
# matching names + JSON schemas so the prompt's "Completion" rules apply
# unchanged. Each gradbot tool call is dispatched in `on_tool_call` below.

def _hang_up_tool_def() -> Any:
    """Shared tool: end the call. Used by assistant + receptionist modes."""
    return gradbot.ToolDef(
        name="hang_up",
        description="End the call after a brief warm closing. Call ONCE when the caller is done.",
        parameters_json=json.dumps({
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": [],
        }),
    )


def _memory_tool_defs() -> list[Any]:
    """Memory tools shared by assistant + owner-inbound modes."""
    return [
        gradbot.ToolDef(
            name="remember",
            description=(
                "Save a durable fact about the caller for future calls. Use when "
                "they tell you something worth remembering (a preference, a name, a "
                "plan, a contact). Keep it to one short sentence."
            ),
            parameters_json=json.dumps({
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "One concise fact to remember."}},
                "required": ["fact"],
            }),
        ),
        gradbot.ToolDef(
            name="recall",
            description=(
                "Look up what you already know about the caller. Use when they refer "
                "to something from a past call, or ask what you remember. Returns "
                "matching facts; pass a topic to narrow, or leave blank for recent ones."
            ),
            parameters_json=json.dumps({
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Topic to search for; optional."}},
                "required": [],
            }),
        ),
    ]


def _web_search_tool_def() -> Any:
    """Live web search (Linkup sourced-answer). Assistant mode only."""
    return gradbot.ToolDef(
        name="web_search",
        description=(
            "Search the live web for current facts that may be outside your training "
            "knowledge — today's news, weather, recent events, prices, sports scores, "
            "anything time-sensitive or freshly changed. Returns a short sourced answer. "
            "Use it whenever the caller asks something you're not confident is current; "
            "do NOT guess at recent facts. Takes a second or two."
        ),
        parameters_json=json.dumps({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language question. Be specific; include the entity, date, or place when known.",
                },
            },
            "required": ["query"],
        }),
    )


def _place_call_tool_def() -> Any:
    """Let the assistant dispatch a NEW outbound call on the caller's behalf.

    This is what turns "call the cafe and order a matcha latte" into an
    actual phone call: a second gradbot agent dials the business in the
    caller's cloned voice, carries out the task, and the result is texted
    back to the caller's Telegram by _post_call_followups. The assistant
    does NOT stay on the line for it — it fires the call and moves on.
    """
    return gradbot.ToolDef(
        name="place_call",
        description=(
            "Place a NEW outbound phone call on the caller's behalf and have a "
            "second agent — speaking in the caller's own cloned voice — carry out "
            "a task on that call. Use this WHENEVER the caller asks you to 'call "
            "X and …' (e.g. call a cafe to order a matcha latte, call a restaurant "
            "to ask about availability, call a shop to check stock). If you don't "
            "already know the number, look it up FIRST with web_search (search for "
            "the business name plus 'phone number'), then call this. You do NOT "
            "stay on the line — the call runs in the background and its result is "
            "texted to the caller on Telegram. After calling this tool, tell the "
            "caller you're placing the call now and will text them the result."
        ),
        parameters_json=json.dumps({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": (
                        "Destination phone number in E.164 (e.g. +14155551234). "
                        "Look it up with web_search first if you don't have it."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "What the calling agent should accomplish, in one or two "
                        "sentences — e.g. 'Ask if they have matcha lattes and order "
                        "one for pickup under the name Pratim.'"
                    ),
                },
                "business_name": {
                    "type": "string",
                    "description": "Name of the business being called, if known (e.g. 'Blue Bottle Coffee').",
                },
                "allow_booking": {
                    "type": "boolean",
                    "description": (
                        "Set true when the task involves placing an order, booking, "
                        "or reserving. Leave false for information-only calls."
                    ),
                },
                "language": {
                    "type": "string",
                    "description": "Call language as an ISO code (en, fr, pt). Defaults to this call's language.",
                },
            },
            "required": ["to", "task"],
        }),
    )


def _assistant_tool_defs() -> list[Any]:
    """Tools for assistant mode (free personal call): email + memory + web +
    place outbound call + hang up."""
    if not HAS_GRADBOT:
        return []
    return _memory_tool_defs() + [
        _web_search_tool_def(),
        _place_call_tool_def(),
        gradbot.ToolDef(
            name="get_email_summary",
            description=(
                "Fetch the caller's recent email so you can summarize it aloud. "
                "Use when they ask about their inbox, email, or messages. Returns a "
                "list of recent messages (sender, subject, date, snippet), newest first."
            ),
            parameters_json=json.dumps({
                "type": "object",
                "properties": {
                    "days": {"type": "integer", "description": "How many days back to look (1 = today, 7 = this week). Default 7."},
                    "max_results": {"type": "integer", "description": "Max messages to return (default 25, cap 50)."},
                },
                "required": [],
            }),
        ),
        _hang_up_tool_def(),
    ]


def _receptionist_tool_defs() -> list[Any]:
    """Tools for inbound receptionist mode: take a message + hang up.
    No data-access tools — the caller is a stranger."""
    if not HAS_GRADBOT:
        return []
    return [
        gradbot.ToolDef(
            name="take_message",
            description=(
                "Record a message for the owner and deliver it to them immediately. "
                "Use once you know who is calling and what they want. Confirm the "
                "message back to the caller before or after calling this."
            ),
            parameters_json=json.dumps({
                "type": "object",
                "properties": {
                    "caller_name": {"type": "string", "description": "Who is calling, as they said it."},
                    "message": {"type": "string", "description": "The message for the owner, one or two sentences."},
                },
                "required": ["message"],
            }),
        ),
        _hang_up_tool_def(),
    ]


def _tool_defs() -> list[Any]:
    if not HAS_GRADBOT:
        return []
    return [
        gradbot.ToolDef(
            name="press_dtmf",
            description=(
                "Press touch-tone (DTMF) digits to navigate an automated phone menu. "
                "Use ONLY when you hear an explicit IVR menu like 'press 1 for English'. "
                "Pass the digit(s) for ONE menu choice as a string."
            ),
            parameters_json=json.dumps({
                "type": "object",
                "properties": {
                    "digits": {"type": "string", "description": "0-9 / * / # — single digit per call"},
                },
                "required": ["digits"],
            }),
        ),
        gradbot.ToolDef(
            name="wait_silently",
            description=(
                "Set the call state to IVR/holding/transfer and stay silent. "
                "Use ONLY after press_dtmf, or when hearing hold music or a transfer announcement."
            ),
            parameters_json=json.dumps({
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "state": {"type": "string", "enum": ["ivr", "holding", "transfer", "voicemail"]},
                },
                "required": [],
            }),
        ),
        gradbot.ToolDef(
            name="save_business_result",
            description=(
                "Save the outcome of this constrained business call. "
                "Use AFTER you have asked the task question and received a clear answer."
            ),
            parameters_json=json.dumps({
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["answered", "voicemail", "unclear", "no_answer"]},
                    "answer": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "follow_up_needed": {"type": "boolean"},
                },
                "required": ["status", "answer"],
            }),
        ),
        gradbot.ToolDef(
            name="end_business_call",
            description=(
                "End the call after thanking the callee. "
                "Call ONCE at the end, after save_business_result."
            ),
            parameters_json=json.dumps({
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                },
                "required": [],
            }),
        ),
    ]


# --- Per-call state ------------------------------------------------------

@dataclass
class _CallState:
    spec: BusinessCallSpec
    room_name: str
    rec_dir: Path
    transcript_path: Path
    business_result: dict = field(default_factory=dict)
    business_state_phase: str = "setup"
    introduced_once: bool = False
    first_user_seen: bool = False
    first_agent_seen: bool = False
    fillers_used: set = field(default_factory=set)
    end_requested: bool = False
    stream_sid: str | None = None
    twilio_call_sid: str = ""
    tenant_id: int | None = None
    voice_id_override: str | None = None
    started_at: float = field(default_factory=time.time)
    # Plain caller/agent turns captured for post-call memory extraction.
    transcript_turns: list = field(default_factory=list)
    # WAV writers — opened lazily on first audio frame.
    wav_caller: wave.Wave_write | None = None
    wav_agent: wave.Wave_write | None = None
    # First-frame anchors (monotonic seconds since session start). The
    # caller side starts as soon as Twilio sends media (= t≈0). The agent
    # side starts when gradbot produces its first TTS chunk (typically 1-2s
    # later). _build_mixed_wav uses (agent - caller) to pre-pad the agent
    # track with silence so the stereo mix stays time-aligned.
    first_caller_audio_at: float | None = None
    first_agent_audio_at: float | None = None
    # Rolling buffer of structured events (last ~30) for the dashboard's
    # Live Calls feed. Each entry: {t, type, ...fields}. type ∈
    # {agent_text, caller_text, state_change, tool_call, system, ringing,
    # connected, ended}. Capped to keep memory bounded for long calls.
    event_log: list[dict] = field(default_factory=list)
    # Don't forward caller audio to gradbot until this long after the agent
    # starts speaking. Gradbot's barge-in threshold is hardcoded in the Rust
    # core (inactivity_prob < 0.4 → cancel TTS), so we can't tune it
    # directly — but blinding gradbot to caller audio during the opener is
    # equivalent: it can't decide to stop if it doesn't hear anything.
    opener_guard_seconds: float = 4.0
    # Running sample counts written to each wav writer. We use these to
    # gap-fill the agent side with silence whenever it falls behind
    # wall-clock — gradbot only emits TTS frames while speaking, so without
    # gap-filling the agent track collapses (a 3-second pause becomes zero
    # samples) and falls progressively out of sync with the caller track.
    caller_samples_written: int = 0
    agent_samples_written: int = 0
    # web_search dedup: gemma sometimes fires the same query many times in a
    # burst. We serialize + cache per-call so identical queries hit Linkup once
    # and every duplicate gets the SAME result (no contradictory answers).
    search_cache: dict = field(default_factory=dict)  # query_key -> (monotonic_ts, payload)
    search_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_EVENT_LOG_CAP = 40
# How long a cached web_search result is reused for an identical query (s).
_SEARCH_CACHE_TTL = 30.0


def _log_event(state: _CallState, type_: str, **fields: object) -> None:
    """Push a structured event into the per-call rolling buffer.

    Read by /calls/live so the dashboard can render a live feed of what
    the agent is currently doing — transcript snippets, state changes,
    tool calls. Capped at ``_EVENT_LOG_CAP`` to keep memory bounded.
    """
    entry = {"t": time.time(), "type": type_, **fields}
    state.event_log.append(entry)
    if len(state.event_log) > _EVENT_LOG_CAP:
        # Drop the oldest entries in-place; keep the list reference stable.
        del state.event_log[: len(state.event_log) - _EVENT_LOG_CAP]


def _timeline_event(state: _CallState, name: str, **extra: object) -> None:
    """Append a speech-pipeline event to the per-call timeline.

    Mirrors agent.py:_timeline_event so the dashboard's discovery
    (`is_business_call` looks for `business_prompt_build` or
    `business_state_change`) and per-call analysis pick up gradbot calls
    exactly like LiveKit calls. Also mirrors interesting events into
    the live-feed buffer so the operator console can render them
    in real time.
    """
    try:
        record: dict[str, object] = {"t_ns": time.monotonic_ns(), "event": name}
        record.update(extra)
        state.rec_dir.mkdir(parents=True, exist_ok=True)
        with (state.rec_dir / "timeline.json.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass
    # Surface meaningful events to the live feed. Skip the noisier ones
    # (business_first_*) that are already implied by agent_text /
    # caller_text entries.
    feed_types = {
        "business_state_change": "state_change",
        "save_business_result": "tool_call",
        "end_business_call": "tool_call",
        "business_result_blocked": "tool_call",
        "business_prompt_build": "system",
        "email_summary": "tool_call",
        "web_search": "tool_call",
        "place_call": "tool_call",
        "take_message": "tool_call",
        "remember": "tool_call",
        "recall": "tool_call",
        "hang_up": "tool_call",
    }
    feed_type = feed_types.get(name)
    if feed_type:
        _log_event(state, feed_type, name=name, **{k: v for k, v in extra.items() if isinstance(v, (str, int, float, bool))})


# Pending calls keyed by room_name. The Twilio Media Streams handler
# looks the call up via the `room` query parameter we put in the TwiML.
# Calls live in _PENDING from /dial dispatch until the WS handler picks
# them up; then they move to _ACTIVE for the lifetime of the WS, and
# are removed in the WS finally. Both dicts are read by /calls/live.
_PENDING: dict[str, _CallState] = {}
_ACTIVE: dict[str, _CallState] = {}
_PENDING_LOCK = asyncio.Lock()

# Concurrency cap on simultaneous gradbot sessions. Gradium STT enforces
# a per-account session limit; without a bridge-side gate, calls 4+
# during a parallel batch fail with "Concurrency limit exceeded" and the
# call is wasted. With the semaphore, late callers wait politely for a
# slot to free.
#
# Tune via GRADBOT_MAX_CONCURRENT env. 0 (or unset) disables the cap.
_GRADBOT_MAX_CONCURRENT = int(os.environ.get("GRADBOT_MAX_CONCURRENT", "0") or "0")
_GRADBOT_SEMAPHORE: asyncio.Semaphore | None = (
    asyncio.Semaphore(_GRADBOT_MAX_CONCURRENT) if _GRADBOT_MAX_CONCURRENT > 0 else None
)


# --- Session config builder ---------------------------------------------

def _make_session_config(
    spec: BusinessCallSpec,
    voice_id_override: str | None = None,
    memory_digest: str = "",
):
    if not HAS_GRADBOT:
        raise RuntimeError("gradbot is not installed — `pip install gradbot`")
    code = (spec.language or "en").lower()
    name = spec.agent_name or agent_name_for_language(code)
    voice_id = voice_id_override or _VOICE_ID.get(code, _VOICE_ID["en"])
    lang = getattr(gradbot.Lang, _LANG_NAME.get(code, "En"))
    mode = (spec.mode or "business").lower()
    if mode == "assistant":
        instructions = build_assistant_prompt(spec, memory_digest=memory_digest)
        tools = _assistant_tool_defs()
    elif mode == "receptionist":
        owner = os.environ.get("OPERATOR_NAME", "").strip()
        instructions = build_receptionist_prompt(spec, owner_name=owner)
        tools = _receptionist_tool_defs()
    else:
        instructions = build_business_prompt(spec, opener_already_spoken=False)
        tools = _tool_defs()
    return gradbot.SessionConfig(
        voice_id=voice_id,
        instructions=instructions,
        language=lang,
        tools=tools,
        # Turn-taking — post-assistant RE-PROMPT nudge. IMPORTANT: despite what
        # earlier comments here claimed, silence_timeout_s does NOT close the
        # caller's turn. Per gradbot's multiplex.rs it is the seconds of silence
        # AFTER the assistant finishes before gradbot injects "..." to make the
        # model prompt continuation — i.e. the "are you still there?" nudge. At
        # 2–3s it fired during a caller's normal think-gap, so the clone nagged
        # while the caller was just gathering a thought. 8.0s lets the caller
        # breathe before the agent checks in, while still recovering from a
        # genuine silence. (The caller's ACTUAL end-of-turn is gradbot's
        # hardcoded VAD cutoff `inactivity_prob > 0.8`; we make it less eager
        # without touching gradbot via stt_extra_config padding_bonus below.)
        # flush_duration_s controls how promptly partial transcripts flush.
        # Both env-tunable, no redeploy.
        silence_timeout_s=_env_float("GRADBOT_SILENCE_TIMEOUT_S", 8.0),
        flush_duration_s=_env_float("GRADBOT_FLUSH_DURATION_S", 0.5),
        # Speak ~20% faster than default. Two knobs:
        #   - padding_bonus negative shrinks inter-token pause time
        #     (LiveKit's business mode uses -1.0 for "slightly brisker";
        #     -2.0 here is more aggressive).
        #   - tts_extra_config.speed is Gradium TTS's rate multiplier
        #     (1.2 = 20% faster word-rate).
        padding_bonus=-2.0,
        tts_extra_config=json.dumps({"speed": 1.1}),
        # STT-side endpointing padding (Gradium ASR json_config). Per Gradium's
        # "Transcription Settings" docs, padding_bonus (range -4..4) biases the
        # model to finalize the caller's turn SOONER (negative) or LATER
        # (positive). A positive value makes a mid-sentence pause (a breath,
        # gathering a thought) less likely to be read as end-of-turn and cut the
        # caller off. This is the STT counterpart to the TTS padding_bonus field
        # above — different engine, and we want the opposite sign here. It must
        # go through stt_extra_config because gradbot's own SessionConfig
        # .padding_bonus is (mis)wired to the TTS stream, not STT.
        # Env-tunable: raise toward 2–3 if still cutting; lower if replies lag.
        stt_extra_config=json.dumps(
            {"padding_bonus": _env_float("GRADBOT_STT_PADDING_BONUS", 1.5)}
        ),
        # First-turn behaviour: assistant speaks first so we play the
        # courtesy opener before the callee says anything. The prompt's
        # turn-1 rules tell the LLM to lead with the brief courteous
        # opener exactly like the LiveKit path's cached opener.
        assistant_speaks_first=True,
    )


# --- Audio recording (μ-law → 16-bit PCM, written to mixed.wav) ---------

def _ulaw_to_pcm16(ulaw: bytes) -> bytes:
    """Decode 8-bit μ-law to 16-bit signed little-endian PCM."""
    import audioop
    return audioop.ulaw2lin(ulaw, 2)


# gradbot's Python binding hardcodes its μ-law sample rates (gradbot_py/src/lib.rs):
# input is 24 kHz, output is 48 kHz. Twilio Media Streams is fixed at 8 kHz mono
# μ-law. Without resampling, agent audio arrives at the phone 6× too slow
# ("animal noises") and caller audio fed into gradbot's STT is 3× too fast.
# We hold per-direction ratecv state so the resample is continuous across frames.
GRADBOT_INPUT_RATE = 24000   # what gradbot expects we send it
GRADBOT_OUTPUT_RATE = 48000  # what gradbot sends us
TWILIO_ULAW_RATE = 8000


def _resample_ulaw(ulaw_in: bytes, src_rate: int, dst_rate: int, state):
    """Resample μ-law bytes between sample rates.

    Returns ``(ulaw_out, new_state)`` so the caller can preserve ratecv state
    across successive chunks. ``state`` is None on the first call.
    """
    if not ulaw_in or src_rate == dst_rate:
        return ulaw_in, state
    import audioop
    pcm = audioop.ulaw2lin(ulaw_in, 2)
    pcm_out, new_state = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, state)
    return audioop.lin2ulaw(pcm_out, 2), new_state


def _open_wav(path: Path, sample_rate: int = 8000) -> wave.Wave_write:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = wave.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    return w


def _close_wav(w: wave.Wave_write | None) -> None:
    if w is None:
        return
    try:
        w.close()
    except Exception:  # noqa: BLE001
        pass


def _build_mixed_wav(rec_dir: Path, agent_lead_seconds: float = 0.0) -> bool:
    """Combine sip_caller.wav (L) + tts_direct.wav (R) into a stereo mixed.wav.

    Both inputs are PCM16 mono at 8 kHz (the rate Twilio Media Streams uses).

    ``agent_lead_seconds`` is the offset between the first caller media
    frame and the first agent TTS frame (positive = agent started later).
    We pre-pad the late track with silence so the stereo mix preserves the
    real-time relationship between the two streams.

    Returns True on success.
    """
    caller_path = rec_dir / "sip_caller.wav"
    agent_path = rec_dir / "tts_direct.wav"
    out_path = rec_dir / "mixed.wav"
    if not (caller_path.exists() and agent_path.exists()):
        return False
    try:
        with wave.open(str(caller_path), "rb") as wc, wave.open(str(agent_path), "rb") as wa:
            if wc.getsampwidth() != 2 or wa.getsampwidth() != 2:
                return False
            rate = wc.getframerate()
            if wa.getframerate() != rate:
                return False
            caller_pcm = wc.readframes(wc.getnframes())
            agent_pcm = wa.readframes(wa.getnframes())
        # Pre-pad the track that started later with silence. Each silence
        # sample is 2 bytes (16-bit PCM); samples = rate * seconds.
        offset_bytes = int(round(abs(agent_lead_seconds) * rate)) * 2
        if offset_bytes > 0:
            silence = b"\x00" * offset_bytes
            if agent_lead_seconds > 0:
                agent_pcm = silence + agent_pcm
            else:
                caller_pcm = silence + caller_pcm
        # Pad whichever is shorter with trailing silence.
        if len(caller_pcm) < len(agent_pcm):
            caller_pcm += b"\x00" * (len(agent_pcm) - len(caller_pcm))
        elif len(agent_pcm) < len(caller_pcm):
            agent_pcm += b"\x00" * (len(caller_pcm) - len(agent_pcm))
        # Interleave into stereo: L=caller, R=agent.
        import audioop
        stereo = audioop.tostereo(caller_pcm, 2, 1.0, 0.0)
        right_only = audioop.tostereo(agent_pcm, 2, 0.0, 1.0)
        mixed = audioop.add(stereo, right_only, 2)
        with wave.open(str(out_path), "wb") as wo:
            wo.setnchannels(2)
            wo.setsampwidth(2)
            wo.setframerate(rate)
            wo.writeframes(mixed)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("mixed.wav build failed: %s", e)
        return False


# --- Operator notifications ----------------------------------------------

async def _notify_telegram(chat_id: str, text: str) -> bool:
    """Send a Telegram message via the bot API. Returns False (and logs) on
    any failure — a call must never die because a notification didn't send."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = str(chat_id or "").strip()
    if not token or not chat_id:
        return False
    import aiohttp

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status != 200:
                    log.warning("telegram notify failed: HTTP %s", r.status)
                    return False
    except aiohttp.ClientError as exc:
        log.warning("telegram notify failed: %s", exc)
        return False
    return True


async def _notify_operator(text: str) -> bool:
    """Relay to the operator's chat: OPERATOR_TELEGRAM_ID, else first id in
    ALLOWED_TELEGRAM_IDS. Used by receptionist mode for taken messages."""
    chat_id = os.environ.get("OPERATOR_TELEGRAM_ID", "").strip()
    if not chat_id:
        chat_id = (os.environ.get("ALLOWED_TELEGRAM_IDS", "").split(",") or [""])[0].strip()
    if not chat_id:
        log.warning("operator notify skipped: no OPERATOR_TELEGRAM_ID / ALLOWED_TELEGRAM_IDS")
        return False
    return await _notify_telegram(chat_id, text)


async def _post_call_followups(state: _CallState) -> None:
    """After a call: extract durable memories and push a summary to the
    tenant's Telegram. Entirely best-effort — never raises into the caller."""
    if state.tenant_id is None:
        return
    mode = (state.spec.mode or "business").lower()
    learned = 0
    try:
        if mode in ("assistant", "interview"):
            learned = await memory_mod.extract_and_store(
                state.tenant_id, state.transcript_turns, room=state.room_name
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("post-call extraction failed: %s", exc)

    try:
        tenant = await tenants.get_tenant_by_id(state.tenant_id)
        chat_id = tenant and tenant.get("telegram_id")
        if not chat_id:
            return
        dur = int(max(0.0, time.time() - state.started_at))
        if mode == "business":
            br = state.business_result or {}
            summary = (f"☎️ Call to {state.spec.destination or 'number'} ended ({dur}s).\n"
                       f"Result: {br.get('status', 'unclear')} — {br.get('answer', '')}".strip())
        else:
            summary = f"☎️ {mode.capitalize()} call ended ({dur}s)."
            if learned:
                summary += f" Remembered {learned} new thing{'s' if learned != 1 else ''}."
        await _notify_telegram(str(chat_id), summary)
    except Exception as exc:  # noqa: BLE001
        log.warning("post-call summary failed: %s", exc)


# --- Tool-call handler --------------------------------------------------

async def _handle_tool_call(handle, state: _CallState, send_event) -> None:
    """Dispatch a single gradbot tool call.

    Mirrors the guards from the LiveKit path:
      - save_business_result(answered) requires both first_user_final_seen
        and first_agent_text_seen (we approximate via business_state_phase
        having moved out of "setup").
      - wait_silently is honoured only when the call is in a transfer/IVR
        context — otherwise we refuse and tell the LLM to keep speaking.
      - end_business_call sets end_requested; the consumer loop will close
        the websocket on the next pass.
    """
    name = handle.name
    args = dict(getattr(handle, "args", {}) or {})

    if name == "press_dtmf":
        digit = (args.get("digits") or "").strip()
        if digit:
            await send_event({"type": "dtmf", "digits": digit, "room": state.room_name})
        await handle.send_json({"success": True, "pressed": digit})
        return

    if name == "wait_silently":
        phase = (args.get("state") or "holding").strip().lower()
        if phase not in {"ivr", "holding", "transfer", "voicemail"}:
            phase = "holding"
        old = state.business_state_phase
        if old != phase:
            _timeline_event(state, "business_state_change", old=old, new=phase, reason=args.get("reason") or "")
        # Honour without the same hard gate as LiveKit — gradbot doesn't
        # have an explicit mute mechanism, but the model self-suppresses
        # when this tool succeeds (per the prompt rules).
        state.business_state_phase = phase
        await handle.send_json({"success": True, "state": phase})
        return

    if name == "save_business_result":
        status = (args.get("status") or "unknown").strip().lower()
        answer = args.get("answer") or ""
        if status == "answered" and state.business_state_phase == "setup":
            _timeline_event(state, "business_result_blocked", reason="setup phase, no conversation yet")
            await handle.send_error(
                "Refused: cannot save 'answered' before any conversation. "
                "Ask the task question first."
            )
            return
        redacted = redact_personal_details(answer)
        if contains_personal_detail(answer):
            log.info("gradbot: redacted personal details from save_business_result")
            _timeline_event(state, "business_result_redacted", status=status)
        state.business_result = {
            "status": status,
            "answer": redacted.strip(),
            "confidence": args.get("confidence") or "medium",
            "follow_up_needed": "true" if args.get("follow_up_needed") else "false",
        }
        _timeline_event(state, "save_business_result", status=status, has_answer=bool(redacted.strip()))
        # Append to transcript like the LiveKit path
        await _append_transcript(state, "BUSINESS-RESULT", json.dumps(state.business_result, ensure_ascii=False))
        await handle.send_json({"success": True})
        return

    if name == "end_business_call":
        state.end_requested = True
        reason = args.get("reason") or ""
        _timeline_event(state, "end_business_call", reason=reason[:120])
        await _append_transcript(state, "BUSINESS-END", reason or "agent ended call")
        await handle.send_json({"success": True})
        return

    if name == "remember":
        fact = (args.get("fact") or "").strip()
        if state.tenant_id is None:
            await handle.send_json({"saved": False, "reason": "no account on this call"})
            return
        saved = await memory_mod.add_memory(
            state.tenant_id, fact, source="remember_tool", room=state.room_name
        )
        _timeline_event(state, "remember", saved=saved)
        await handle.send_json({"saved": saved})
        return

    if name == "recall":
        if state.tenant_id is None:
            await handle.send_json({"facts": []})
            return
        facts = await memory_mod.search_memories(state.tenant_id, args.get("query") or "")
        _timeline_event(state, "recall", count=len(facts))
        await handle.send_json({"facts": facts})
        return

    if name == "get_email_summary":
        days = args.get("days") or 7
        max_results = args.get("max_results") or 25
        try:
            messages = await asyncio.to_thread(
                email_inbox.fetch_recent, days=days, max_results=max_results
            )
        except email_inbox.EmailNotConfigured:
            # Keep the model-facing text generic; never echo credential/host
            # details into the spoken reply or the transcript.
            _timeline_event(state, "email_summary", error="not_configured")
            await handle.send_json({"error": "Email is not set up yet.", "configured": False})
            return
        except email_inbox.EmailFetchError as exc:
            log.warning("get_email_summary failed: %s", exc)
            _timeline_event(state, "email_summary", error="fetch_failed")
            await handle.send_json({"error": "Could not read email right now.", "configured": True})
            return
        _timeline_event(state, "email_summary", count=len(messages), days=int(days))
        await handle.send_json({"count": len(messages), "days": int(days), "messages": messages})
        return

    if name == "web_search":
        query = (args.get("query") or "").strip()
        if not query:
            await handle.send_error("Refused: empty search query — ask what to look up.")
            return
        # Receipt so the search is debuggable from Render logs: the query, then
        # its outcome (timeout / not_configured / failed / ok + latency).
        log.info("call %s | web_search query=%r", state.room_name, query)
        key = query.lower()
        # Serialize per-call so a burst of identical queries collapses to one
        # Linkup call; every duplicate returns the SAME cached payload, so the
        # model can't get two different results to answer twice with.
        async with state.search_lock:
            cached = state.search_cache.get(key)
            if cached and (time.monotonic() - cached[0]) < _SEARCH_CACHE_TTL:
                log.info("call %s | web_search CACHED query=%r", state.room_name, query)
                await handle.send_json(cached[1])
                return
            t0 = time.time()
            try:
                # Hard cap the wait: a live call can't tolerate long silence. If
                # Linkup is slow, fall back gracefully rather than hang the turn.
                result = await asyncio.wait_for(
                    asyncio.to_thread(websearch.search, query), timeout=8.0
                )
            except asyncio.TimeoutError:
                log.warning("call %s | web_search TIMEOUT after %.1fs query=%r", state.room_name, time.time() - t0, query)
                _timeline_event(state, "web_search", query=query[:120], error="timeout")
                await handle.send_json({"error": "The search took too long — tell the caller you couldn't pull it up right now.", "configured": True})
                return
            except websearch.WebSearchNotConfigured:
                log.warning("call %s | web_search not configured (LINKUP_API_KEY missing)", state.room_name)
                _timeline_event(state, "web_search", error="not_configured")
                await handle.send_json({"error": "Web search isn't set up.", "configured": False})
                return
            except websearch.WebSearchError as exc:
                log.warning("call %s | web_search FAILED query=%r: %s", state.room_name, query, exc)
                _timeline_event(state, "web_search", query=query[:120], error="failed")
                await handle.send_json({"error": "Couldn't reach the web right now.", "configured": True})
                return
            n_sources = len(result.get("sources", []))
            log.info("call %s | web_search OK in %.1fs sources=%d query=%r", state.room_name, time.time() - t0, n_sources, query)
            _timeline_event(state, "web_search", query=query[:120], sources=n_sources)
            payload = {"answer": result["answer"], "sources": result["sources"]}
            state.search_cache[key] = (time.monotonic(), payload)
            await handle.send_json(payload)
        return

    if name == "take_message":
        caller_name = (args.get("caller_name") or "").strip()
        message = (args.get("message") or "").strip()
        if not message:
            await handle.send_error("Refused: message is empty — ask the caller what to pass on.")
            return
        who = caller_name or state.spec.destination or "unknown caller"
        delivered = await _notify_operator(f"📞 Message from {who} ({state.spec.destination}):\n{message}")
        _timeline_event(state, "take_message", caller=who[:80], delivered=delivered)
        await _append_transcript(state, "MESSAGE-TAKEN", f"{who}: {message}")
        await handle.send_json({"success": True, "delivered": delivered})
        return

    if name == "place_call":
        raw_to = (args.get("to") or "").strip()
        had_plus = raw_to.startswith("+")
        digits = "".join(ch for ch in raw_to if ch.isdigit())
        # The model / web-search results often give a US number with no country
        # code (e.g. "415-362-8342"), which would build the invalid "+4153628342".
        # Coerce a bare 10-digit number to +1. A leading '+' means a full
        # international number was supplied — trust it as-is.
        if not had_plus and len(digits) == 10:
            digits = "1" + digits
        task = (args.get("task") or "").strip()
        business = (args.get("business_name") or "").strip()
        # Receipt so place_call is debuggable from the logs (like web_search):
        # the request, then its outcome (no_number / no_task / not_allowed /
        # dispatch_error / ok + room).
        log.info(
            "call %s | place_call to=%r business=%r task=%r",
            state.room_name, raw_to, business[:80], task[:120],
        )
        if not digits:
            log.warning("call %s | place_call REFUSED: no valid phone number (to=%r)",
                        state.room_name, raw_to)
            await handle.send_error(
                "Refused: no valid phone number. Look the business up with web_search "
                "first, then call place_call with its number in E.164 (e.g. +1…)."
            )
            return
        if not task:
            log.warning("call %s | place_call REFUSED: no task", state.room_name)
            await handle.send_error("Refused: no task — say what the call should accomplish.")
            return
        to = "+" + digits
        if not _outbound_destination_allowed(to):
            log.warning(
                "call %s | place_call REFUSED: %s not allowed (set OUTBOUND_ALLOWLIST "
                "or ALLOW_ARBITRARY_OUTBOUND)", state.room_name, to,
            )
            _timeline_event(state, "place_call", to=to, error="not_allowed")
            await handle.send_json({
                "success": False,
                "error": "That number isn't on the allowed list for outbound calls — "
                         "tell the caller you can't dial it right now.",
            })
            return
        lang = (args.get("language") or state.spec.language or "en").lower()
        sub_spec = BusinessCallSpec(
            task=task,
            language=lang,
            business_name=business,
            destination=to,
            allow_booking=bool(args.get("allow_booking", False)),
            mode="business",
        )
        out = await dispatch_gradbot_call(to=to, spec=sub_spec, tenant_id=state.tenant_id)
        if isinstance(out, str) and out.startswith("Error:"):
            log.warning("call %s | place_call DISPATCH FAILED to=%s: %s",
                        state.room_name, to, out)
            _timeline_event(state, "place_call", to=to, error=out[:120])
            await handle.send_json({
                "success": False,
                "error": "Couldn't place the call — tell the caller it didn't go through.",
            })
            return
        log.info("call %s | place_call OK to=%s room=%s", state.room_name, to, out)
        _timeline_event(
            state, "place_call", to=to, room=out,
            business=business[:80] or task[:80],
        )
        await _append_transcript(state, "PLACE-CALL", f"{to} — {task}")
        await handle.send_json({
            "success": True,
            "room": out,
            "note": "Call placed; it's running in the background and the result will be "
                    "texted to the caller on Telegram. Tell the caller it's on its way.",
        })
        return

    if name == "hang_up":
        state.end_requested = True
        reason = args.get("reason") or ""
        _timeline_event(state, "hang_up", reason=reason[:120])
        await _append_transcript(state, "ASSISTANT-END", reason or "caller ended call")
        await handle.send_json({"success": True})
        return

    await handle.send_error(f"Unknown tool: {name}")


# --- Transcript writer (matches the LiveKit format) ---------------------

async def _append_transcript(state: _CallState, kind: str, text: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] **{kind}:** {text}\n\n"
    try:
        state.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        # File I/O is small; the cost of running it on the event loop is
        # negligible compared to the audio path.
        with state.transcript_path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        log.warning("transcript write failed: %s", e)
    # Mirror into the live-feed buffer. Map the markdown ROLE label to a
    # compact event type the dashboard can style consistently.
    type_map = {
        "AGENT": "agent_text",
        "CALLER": "caller_text",
        "SYSTEM": "system",
        "BUSINESS-RESULT": "result_saved",
        "BUSINESS-END": "ended",
    }
    _log_event(state, type_map.get(kind, "system"), text=text[:240])


# --- FastAPI app: /twilio/voice + /twilio/stream ------------------------

app = fastapi.FastAPI(title="gradphone bridge")


@app.on_event("startup")
async def _on_startup() -> None:
    """Create the DB + tables (SQLite/dev) or rely on Alembic (Postgres). Idempotent."""
    await tenants.init_db()
    log.info("gradphone bridge ready — DB at %s", db_mod.url_for_log())


# ─── Web UI (operator's switchboard) ─────────────────────────────────
# Mount the static + router at /ui. Imports deferred so the bridge can
# still boot in environments without jinja2/itsdangerous (the JSON API
# remains usable; only the UI degrades).
try:
    from pathlib import Path as _Path
    from fastapi.staticfiles import StaticFiles
    from .web import router as _web_router
    _STATIC_DIR = _Path(__file__).parent / "static"
    if _STATIC_DIR.exists():
        app.mount("/ui/static", StaticFiles(directory=str(_STATIC_DIR)), name="ui-static")
    app.include_router(_web_router)
    log.info("gradphone web UI available at /ui")
except Exception as e:  # noqa: BLE001
    log.warning("Web UI not mounted: %s", e)


# --- Auth ----------------------------------------------------------------
#
# The bridge is exposed to the public internet (Twilio needs to reach the
# TwiML and WS endpoints). Three endpoint classes, each with its own
# auth model:
#
# - /dial: callable by us (and eventually the gizmogrid portal). Requires
#   a long random bearer token in `BRIDGE_API_KEY` env. If the env var is
#   unset, requests are allowed but a startup warning is logged — useful
#   for local dev where the bridge is only reachable via 127.0.0.1.
#
# - /twilio/voice + /twilio/stream: callable by Twilio. Verified via
#   Twilio's HMAC-SHA1 signature on `X-Twilio-Signature` using the
#   account auth token. If `TWILIO_AUTH_TOKEN` is unset, validation is
#   skipped with a startup warning.
#
# - /healthz: open. Exposes only "is gradbot installed" + a count.

# Explicit, opt-in escape hatch for purely-local development without the real
# secrets. It must be set deliberately; the mere ABSENCE of a secret never
# disables auth (that would make every misconfigured fork wide open). When this
# is off (the default), missing secrets fail CLOSED.
_ALLOW_INSECURE_LOCAL = os.environ.get("ALLOW_INSECURE_LOCAL", "").strip().lower() in (
    "1", "true", "yes",
)

_BRIDGE_API_KEY = os.environ.get("BRIDGE_API_KEY", "").strip()
if not _BRIDGE_API_KEY:
    if _ALLOW_INSECURE_LOCAL:
        log.warning(
            "BRIDGE_API_KEY unset and ALLOW_INSECURE_LOCAL=1 — control endpoints "
            "are UNAUTHENTICATED. Never do this on a public host."
        )
    else:
        log.error(
            "BRIDGE_API_KEY unset — control endpoints will reject all requests. "
            "Set BRIDGE_API_KEY (or ALLOW_INSECURE_LOCAL=1 for localhost-only dev)."
        )

_TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
if not _TWILIO_AUTH_TOKEN:
    if _ALLOW_INSECURE_LOCAL:
        log.warning(
            "TWILIO_AUTH_TOKEN unset and ALLOW_INSECURE_LOCAL=1 — Twilio signature "
            "verification is SKIPPED. Never do this on a public host."
        )
    else:
        log.error(
            "TWILIO_AUTH_TOKEN unset — /twilio/* will reject unverified requests. "
            "Set TWILIO_AUTH_TOKEN (or ALLOW_INSECURE_LOCAL=1 for localhost-only dev)."
        )


def _require_bearer(request: fastapi.Request) -> None:
    """FastAPI dependency: enforce `Authorization: Bearer <BRIDGE_API_KEY>`,
    OR a valid OPERATOR session cookie.

    Tenant sessions don't unlock these endpoints — they hit /ui/* wrappers
    that scope by tenant_id instead. Fails CLOSED when `BRIDGE_API_KEY` is
    unset (unless ALLOW_INSECURE_LOCAL=1 is explicitly set for local dev).
    """
    if not _BRIDGE_API_KEY:
        if _ALLOW_INSECURE_LOCAL:
            return
        raise fastapi.HTTPException(
            status_code=503,
            detail="server misconfigured: BRIDGE_API_KEY is not set",
        )
    import hmac
    header = request.headers.get("authorization", "")
    expected = f"Bearer {_BRIDGE_API_KEY}"
    if hmac.compare_digest(header, expected):
        return
    # Fall through: only OPERATOR session cookies count here.
    from .sessions import COOKIE_NAME, decode_role, is_operator
    if is_operator(decode_role(request.cookies.get(COOKIE_NAME))):
        return
    raise fastapi.HTTPException(status_code=401, detail="invalid or missing credentials")


def _verify_twilio_signature(
    method: str,
    url: str,
    params: dict,
    signature: str,
) -> bool:
    """Validate Twilio's HMAC-SHA1 signature on a request.

    Returns True if the signature matches. Twilio's algorithm:
      - sort params by key, concatenate key+value pairs, append to URL
      - HMAC-SHA1 the result with the auth token, base64 encode
    For WebSocket upgrade requests, Twilio signs the URL with no params
    (the ``params`` dict is empty).

    Fails CLOSED when TWILIO_AUTH_TOKEN is unset (returns False), unless
    ALLOW_INSECURE_LOCAL=1 is explicitly set for local dev without Twilio.
    """
    if not _TWILIO_AUTH_TOKEN:
        return _ALLOW_INSECURE_LOCAL
    if not signature:
        return False
    import base64 as _b64
    import hashlib
    import hmac
    payload = url
    for k in sorted(params.keys()):
        payload += k + (params[k] or "")
    digest = hmac.new(
        _TWILIO_AUTH_TOKEN.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    expected = _b64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature)


# --- Routes --------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "gradbot_installed": HAS_GRADBOT,
        "pending_calls": len(_PENDING),
        "active_calls": len(_ACTIVE),
    }


def _count_outbound_sockets() -> dict:
    """Count outbound TCP sockets owned by THIS Python process, grouped
    by remote host. Used by /diagnostics so the dashboard can show how
    many WebSockets gradbot is actually holding open right now.

    Reads /proc/self/net/tcp + /proc/self/net/tcp6 and matches socket
    inodes against /proc/self/fd/* to find sockets owned by this pid.
    Falls back to an empty result on non-Linux or on parse errors.
    """
    import socket as _sock

    # Map fd -> socket inode for THIS process.
    try:
        fd_dir = Path("/proc/self/fd")
        inode_set: set[str] = set()
        for fd in fd_dir.iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("socket:["):
                inode_set.add(target[8:-1])
    except Exception:  # noqa: BLE001
        return {"error": "/proc/self/fd unreadable", "by_host": {}, "total": 0}

    by_host: dict[str, int] = {}
    total = 0

    def _read(path: str, ipv6: bool) -> None:
        nonlocal total
        try:
            with open(path) as f:
                next(f)  # header
                for line in f:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    state = parts[3]
                    if state != "01":  # ESTABLISHED
                        continue
                    inode = parts[9]
                    if inode not in inode_set:
                        continue
                    rem = parts[2]
                    addr_hex, port_hex = rem.rsplit(":", 1)
                    port = int(port_hex, 16)
                    try:
                        if ipv6:
                            raw = bytes.fromhex(addr_hex)
                            # /proc gives IPv6 in 32-hex little-endian-by-word
                            ip = _sock.inet_ntop(_sock.AF_INET6, bytes(raw[i ^ 3] for i in range(16)))
                        else:
                            raw = bytes.fromhex(addr_hex)
                            ip = ".".join(str(b) for b in reversed(raw))
                    except Exception:  # noqa: BLE001
                        ip = addr_hex
                    key = f"{ip}:{port}"
                    by_host[key] = by_host.get(key, 0) + 1
                    total += 1
        except FileNotFoundError:
            pass

    _read("/proc/self/net/tcp", ipv6=False)
    _read("/proc/self/net/tcp6", ipv6=True)
    return {"total": total, "by_host": by_host}


@app.get("/diagnostics", dependencies=[fastapi.Depends(_require_bearer)])
async def diagnostics() -> dict:
    """Visibility for operators: how many calls in flight, how many
    outbound sockets the bridge is holding, and a guess at how those
    map to Gradium STT/TTS + LLM.
    """
    sockets = _count_outbound_sockets()
    # Heuristic upstream classification by /proc-resolved IP — Gradium STT
    # and TTS run on different hosts but resolve to the same edges in
    # practice; LLM is the Anthropic IP block.
    classes = {"gradium": 0, "anthropic": 0, "twilio": 0, "loopback": 0, "other": 0}
    for host, count in sockets.get("by_host", {}).items():
        # Strip the trailing port for classification.
        ip = host.rsplit(":", 1)[0]
        if ip.startswith("127.") or ip == "::1":
            classes["loopback"] += count       # gizmogrid → bridge polling, etc.
        elif ip.startswith("160.79."):
            classes["anthropic"] += count      # Anthropic API range
        elif "twilio" in host:
            classes["twilio"] += count
        else:
            # Bridge's only other outbound hop is gradbot→Gradium.
            classes["gradium"] += count
    return {
        "ok": True,
        "pending_calls": len(_PENDING),
        "active_calls": len(_ACTIVE),
        "sockets_total": sockets.get("total", 0),
        "sockets_by_host": sockets.get("by_host", {}),
        "sockets_by_class": classes,
        "concurrency_limit": _GRADBOT_MAX_CONCURRENT or "unset",
    }


def _snapshot_call(state: _CallState, *, phase_override: str | None = None) -> dict:
    """Render a _CallState as a JSON-safe row for /calls/live."""
    now = time.time()
    # Tail of the rolling event buffer — last 20 entries are plenty for a
    # live feed; older entries already wrote to the transcript file.
    events = state.event_log[-20:] if state.event_log else []
    return {
        "room": state.room_name,
        "tenant_id": state.tenant_id,
        "destination": state.spec.destination,
        "business_name": state.spec.business_name,
        "language": state.spec.language,
        "phase": phase_override or state.business_state_phase,
        "started_at": state.started_at,
        "age_seconds": round(now - state.started_at, 1),
        "first_user_seen": bool(state.first_user_seen),
        "first_agent_seen": bool(state.first_agent_seen),
        "stream_sid": state.stream_sid,
        "result_saved": bool(state.business_result),
        "events": events,
    }


@app.get("/result/{room}", dependencies=[fastapi.Depends(_require_bearer)])
async def get_result(room: str) -> dict:
    """Return the call's final outcome once the recording dir has metrics.json.

    Status values:
      - "pending"   — bridge knows the room (still ringing or active).
      - "missing"   — bridge has no record of the room AND no metrics file
                      exists. Either the room name is wrong, or the call
                      completed before this process started (e.g. bridge
                      restarted mid-call).
      - "complete"  — metrics.json present. The "result" field carries
                      the business_result the agent saved + framework +
                      duration + an optional twilio_call_status fed in
                      by the Twilio status callback.
    """
    room = _safe_room(room)
    rec_dir = RECORDINGS_ROOT / room
    metrics_path = rec_dir / "metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            return {"status": "error", "error": f"metrics read failed: {e}", "room": room}
        return {"status": "complete", "room": room, "result": data}
    async with _PENDING_LOCK:
        in_flight = room in _PENDING or room in _ACTIVE
    if in_flight:
        return {"status": "pending", "room": room}
    # DB fallback — the call may have completed before this bridge process
    # started (restart mid-call) or the recording dir was cleaned up.
    db_row = await tenants.get_call(room)
    if db_row and db_row.get("ended_at"):
        return {
            "status": "complete",
            "room": room,
            "result": {
                "call_id": room,
                "framework": "gradbot",
                "duration_seconds": db_row.get("duration_seconds") or 0.0,
                "business_result": {
                    "status": db_row.get("status") or "unclear",
                    "answer": db_row.get("answer") or "",
                    "confidence": db_row.get("confidence") or "",
                },
                "twilio_call_status": db_row.get("twilio_call_status") or "",
                "answered_by": db_row.get("answered_by") or "",
                "source": "db",
            },
        }
    return {"status": "missing", "room": room}


@app.post("/twilio/status")
async def twilio_status(request: fastapi.Request):
    """Twilio status callback — fires on ringing/answered/completed/failed.

    We use this to:
      - Detect calls that never reach the Media Streams handler (busy /
        no-answer / failed) so /result/<room> can return them cleanly.
      - Surface AnsweredBy (from machine_detection) so the agent's
        framework metadata records whether it spoke to a human, a
        voicemail, or a fax.

    Twilio signs status callbacks the same way as TwiML requests. We
    verify and then write a sidecar `twilio_status.json` under the
    recording dir so the result endpoint can fold it into the response.
    """
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.url.netloc
    signed_url = f"{proto}://{host}{request.url.path}"
    if request.url.query:
        signed_url += f"?{request.url.query}"
    form = await request.form()
    params = {k: v for k, v in form.multi_items()}
    if not _verify_twilio_signature(
        "POST",
        signed_url,
        params,
        request.headers.get("x-twilio-signature", ""),
    ):
        raise fastapi.HTTPException(status_code=403, detail="invalid Twilio signature")

    room = request.query_params.get("room", "")
    if not room:
        return {"ok": True}
    room = _safe_room(room)

    call_status = params.get("CallStatus", "")
    answered_by = params.get("AnsweredBy", "")
    duration = params.get("CallDuration", "")
    sid = params.get("CallSid", "")

    rec_dir = RECORDINGS_ROOT / room
    rec_dir.mkdir(parents=True, exist_ok=True)
    sidecar = rec_dir / "twilio_status.json"
    try:
        existing = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    except (OSError, json.JSONDecodeError):
        existing = {}
    existing.update({
        "call_sid": sid,
        "call_status": call_status,
        "answered_by": answered_by,
        "duration": duration,
    })
    try:
        sidecar.write_text(json.dumps(existing, indent=2))
    except OSError as e:
        log.warning("twilio_status sidecar write failed: %s", e)

    # If the call ended without ever reaching /twilio/stream (busy /
    # no-answer / failed), emit a minimal metrics.json so /result/<room>
    # has something to return instead of "pending" forever.
    terminal = {"completed", "busy", "no-answer", "failed", "canceled"}
    if call_status in terminal:
        biz_status = _twilio_status_to_business_status(call_status, answered_by)
        metrics_path = rec_dir / "metrics.json"
        if not metrics_path.exists():
            try:
                metrics_path.write_text(json.dumps({
                    "call_id": room,
                    "framework": "gradbot",
                    "duration_seconds": float(duration or 0.0),
                    "business_result": {
                        "status": biz_status,
                        "answer": "",
                        "confidence": "high",
                        "follow_up_needed": "false",
                    },
                    "twilio_call_status": call_status,
                    "answered_by": answered_by,
                }, indent=2))
            except OSError as e:
                log.warning("twilio_status synthetic metrics write failed: %s", e)
        # Persist to the calls table (idempotent — only updates rows still
        # marked 'pending').
        async with _PENDING_LOCK:
            _PENDING.pop(room, None) or _ACTIVE.pop(room, None)
        try:
            await tenants.record_call_end(
                room=room,
                status=biz_status,
                twilio_call_status=call_status,
                answered_by=answered_by,
                duration_seconds=float(duration or 0.0),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("record_call_end failed for room=%s: %s", room, e)
    return {"ok": True}


def _twilio_status_to_business_status(call_status: str, answered_by: str) -> str:
    if answered_by.startswith("machine") or answered_by == "fax":
        return "voicemail"
    if call_status in {"busy", "no-answer", "canceled"}:
        return "no_answer"
    if call_status == "failed":
        return "unclear"
    return "unclear"


@app.get("/calls/live", dependencies=[fastapi.Depends(_require_bearer)])
async def calls_live() -> dict:
    """Snapshot of every call currently in flight.

    `phase = "ringing"` means the bridge has dispatched the Twilio call
    but the WebSocket stream hasn't connected yet (line is still
    ringing). After connect, `phase` reflects the live business-call
    state machine: setup / human / ivr / holding / transfer / voicemail.
    """
    async with _PENDING_LOCK:
        ringing = [_snapshot_call(s, phase_override="ringing") for s in _PENDING.values()]
        active = [_snapshot_call(s) for s in _ACTIVE.values()]
    calls = ringing + active
    calls.sort(key=lambda c: c["started_at"], reverse=True)
    return {"ok": True, "count": len(calls), "calls": calls}


import re as _re

_ROOM_RE = _re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_room(room: str) -> str:
    """Validate a room name before it is used to build a filesystem path.

    Room names are always server-generated as ``[A-Za-z0-9_-]+``; rejecting
    anything else stops a request-supplied ``room`` (path or query param) from
    escaping RECORDINGS_ROOT via ``..`` / separators (path traversal)."""
    if not room or not _ROOM_RE.match(room):
        raise fastapi.HTTPException(status_code=400, detail="invalid room")
    return room


def _outbound_destination_allowed(to: str) -> bool:
    """Workshop guard: outbound dialing is restricted to OUTBOUND_ALLOWLIST
    (comma-separated E.164) unless ALLOW_ARBITRARY_OUTBOUND=true. With
    neither set, every destination is refused — attendees must not be able
    to dial arbitrary numbers in a cloned voice."""
    if os.environ.get("ALLOW_ARBITRARY_OUTBOUND", "").strip().lower() in ("1", "true", "yes"):
        return True
    allowlist = {
        "+" + "".join(ch for ch in n if ch.isdigit())
        for n in os.environ.get("OUTBOUND_ALLOWLIST", "").split(",")
        if n.strip()
    }
    return to in allowlist


@app.post("/dial", dependencies=[fastapi.Depends(_require_bearer)])
async def dial(spec_payload: dict) -> dict:
    """In-process dial endpoint.

    The bridge's `_PENDING` registry only lives in this process — calling
    `dispatch_gradbot_call` from a separate Python (e.g. the
    `make_call(..., orchestrator='gradbot')` caller) would register state
    in the wrong process. Wrap the dispatch in this HTTP endpoint so the
    state lands here, where the WS handler can find it.

    Request JSON:
      {"to": "+1…", "reason": "…", "language": "fr", "mode": "business",
       "business_name": "", "allow_booking": false, "tenant_id": 42}
    Response: {"room": "outbound-…"}  or  {"error": "…"}

    ``tenant_id`` is the owner's id. When present, the call is recorded in
    the ``calls`` table; operator-mode (no tenant_id) calls skip history.
    """
    raw_to = spec_payload.get("to", "")
    normalised_to = "+" + "".join(ch for ch in str(raw_to) if ch.isdigit())
    if normalised_to == "+":
        return {"error": "Error: 'to' must contain digits"}

    if not _outbound_destination_allowed(normalised_to):
        log.warning("dial refused: %s not in OUTBOUND_ALLOWLIST", normalised_to)
        return {
            "error": "Error: destination not allowed. Ask the operator to add it to "
                     "OUTBOUND_ALLOWLIST (or set ALLOW_ARBITRARY_OUTBOUND=true)."
        }

    tenant_id = spec_payload.get("tenant_id")
    if tenant_id is not None:
        try:
            tenant_id = int(tenant_id)
        except (TypeError, ValueError):
            return {"error": "Error: tenant_id must be an integer"}

    # /dial only places OUTBOUND calls. "receptionist" is inbound-only and is
    # set directly in _register_inbound_call, so it's intentionally not a valid
    # /dial mode — anything unexpected falls back to a normal business call.
    mode = (spec_payload.get("mode") or "business").lower()
    if mode not in ("business", "assistant"):
        mode = "business"
    spec = BusinessCallSpec(
        task=spec_payload.get("reason", ""),
        language=(spec_payload.get("language") or "en").lower(),
        business_name=spec_payload.get("business_name", ""),
        destination=normalised_to,
        allow_booking=bool(spec_payload.get("allow_booking", False)),
        mode=mode,
    )
    out = await dispatch_gradbot_call(to=normalised_to, spec=spec, tenant_id=tenant_id)
    if out.startswith("Error:"):
        return {"error": out}
    return {"room": out}


@app.get("/history/{tenant_id}", dependencies=[fastapi.Depends(_require_bearer)])
async def history(tenant_id: int, limit: int = 10) -> dict:
    """Return the last ``limit`` calls placed by this tenant, most recent first."""
    limit = max(1, min(limit, 100))
    rows = await tenants.list_calls(tenant_id, limit=limit)
    return {"ok": True, "tenant_id": tenant_id, "count": len(rows), "calls": rows}


@app.get("/tenants/{telegram_id}", dependencies=[fastapi.Depends(_require_bearer)])
async def get_tenant(telegram_id: int) -> dict:
    row = await tenants.get_tenant_by_telegram(telegram_id)
    if not row:
        return {"ok": False, "error": "not registered"}
    return {"ok": True, "tenant": row}


@app.post("/tenants", dependencies=[fastapi.Depends(_require_bearer)])
async def register(payload: dict) -> dict:
    telegram_id = payload.get("telegram_id")
    name = (payload.get("name") or "").strip()
    if not isinstance(telegram_id, int) or not name:
        return {"ok": False, "error": "telegram_id (int) and name (string) required"}
    tenant_id = await tenants.register_tenant(telegram_id, name)
    return {"ok": True, "tenant_id": tenant_id}


async def _register_inbound_call(params: dict) -> str:
    """Build receptionist call state for an inbound call and register it in
    _PENDING. Returns the generated room name to embed in the TwiML stream.

    Inbound calls hit /twilio/voice with no ``room`` query param — Twilio
    sends CallSid / From / To instead. We mint state on the fly, keyed by
    CallSid, so the WS handler can find it like an outbound call.
    """
    call_sid = (params.get("CallSid") or os.urandom(6).hex()).replace("/", "_")
    caller = params.get("From", "")
    room_name = f"inbound-{call_sid}"
    rec_dir = RECORDINGS_ROOT / room_name
    rec_dir.mkdir(parents=True, exist_ok=True)
    (rec_dir / "framework.txt").write_text("gradbot\n")
    transcript_path = TRANSCRIPTS_ROOT / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{room_name}.md"

    # Caller ID = identity. If the caller's number belongs to a registered
    # tenant, they reach their OWN assistant (their voice + memory). Anyone
    # else gets the receptionist (screen + take a message, no data access).
    lang = (os.environ.get("INBOUND_LANGUAGE", "en") or "en").lower()
    tenant = None
    try:
        tenant = await tenants.get_tenant_by_phone(caller)
    except Exception as exc:  # noqa: BLE001
        log.warning("caller-id lookup failed: %s", exc)

    if tenant:
        mode = "assistant"
        tenant_id = int(tenant["id"])
        voice_id_override = tenant.get("voice_id") or os.environ.get("INBOUND_VOICE_ID", "").strip() or None
    else:
        mode = "receptionist"
        tenant_id = None
        voice_id_override = os.environ.get("INBOUND_VOICE_ID", "").strip() or None

    spec = BusinessCallSpec(task="", language=lang, destination=caller, mode=mode)
    state = _CallState(
        spec=spec,
        room_name=room_name,
        rec_dir=rec_dir,
        transcript_path=transcript_path,
        tenant_id=tenant_id,
        voice_id_override=voice_id_override,
    )
    state.twilio_call_sid = params.get("CallSid", "")
    _log_event(state, "ringing", destination=caller, business_name=f"(inbound:{mode})")
    async with _PENDING_LOCK:
        _PENDING[room_name] = state
    log.info("inbound call: room=%s from=%s mode=%s tenant=%s", room_name, caller, mode, tenant_id)
    return room_name


@app.post("/twilio/voice", response_class=PlainTextResponse)
async def twilio_voice(request: fastapi.Request):
    """TwiML returned to Twilio after pickup — connects to /twilio/stream.

    Twilio strips query parameters from <Stream url=…> on the WS upgrade.
    Pass the room name via <Parameter name="room" value="…"/> inside
    <Stream> instead — Twilio delivers it in the WS `start` event under
    `start.customParameters.room`.
    """
    # Verify Twilio's signature so the public TwiML endpoint can't be
    # poked by random clients on the internet. Reconstruct the signed
    # URL from x-forwarded headers (Caddy/nginx terminate TLS).
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.url.netloc
    signed_url = f"{proto}://{host}{request.url.path}"
    if request.url.query:
        signed_url += f"?{request.url.query}"
    form = await request.form()
    params = {k: v for k, v in form.multi_items()}
    if not _verify_twilio_signature(
        "POST",
        signed_url,
        params,
        request.headers.get("x-twilio-signature", ""),
    ):
        log.warning("twilio_voice: signature mismatch from %s", request.client)
        raise fastapi.HTTPException(status_code=403, detail="invalid Twilio signature")
    room = request.query_params.get("room", "")

    # Inbound branch — no room means Twilio is routing an INCOMING call to us
    # (the number's Voice webhook points here). Gated behind ENABLE_INBOUND so
    # the line isn't live by accident. When on, mint receptionist state and
    # answer; when off, politely decline.
    if not room:
        if os.environ.get("ENABLE_INBOUND", "").strip().lower() not in ("1", "true", "yes"):
            log.info("inbound call rejected (ENABLE_INBOUND off) from %s", params.get("From"))
            return PlainTextResponse(
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Response><Say>Sorry, this line is not available right now. Goodbye.</Say>'
                '<Hangup/></Response>',
                media_type="application/xml",
            )
        room = await _register_inbound_call(params)

    # Voicemail branch — if Twilio's machine_detection caught a non-human
    # answerer, skip the Media Streams handshake entirely. Saves a
    # Gradium STT/TTS slot and (optionally) leaves a brief localized
    # voicemail message via Twilio Say. The originating call still gets
    # a metrics.json row from /twilio/status when it terminates.
    answered_by = params.get("AnsweredBy", "")
    if is_machine(answered_by):
        async with _PENDING_LOCK:
            state = _PENDING.get(room) or _ACTIVE.get(room)
        task = state.spec.task if state else ""
        language = state.spec.language if state else "en"
        log.info("voicemail branch: room=%s AnsweredBy=%s lang=%s", room, answered_by, language)
        return PlainTextResponse(
            voicemail_twiml(task=task, language=language),
            media_type="application/xml",
        )

    public_ws = os.environ.get("PUBLIC_WS_URL", "").rstrip("/")
    if not public_ws:
        # Fall back to the request host (works behind a single tunnel).
        host = request.headers.get("x-forwarded-host") or request.url.hostname
        public_ws = f"wss://{host}"
    ws_url = f"{public_ws}/twilio/stream"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Response>'
        '  <Connect>'
        f'    <Stream url="{ws_url}">'
        f'      <Parameter name="room" value="{room}"/>'
        '    </Stream>'
        '  </Connect>'
        '</Response>'
    )
    return PlainTextResponse(twiml, media_type="application/xml")


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: fastapi.WebSocket):
    """Bridge a Twilio Media Streams session to gradbot.run()."""
    if not HAS_GRADBOT:
        await websocket.close(code=1011, reason="gradbot not installed")
        return

    # Verify Twilio's signature on the WS upgrade. For WebSockets, Twilio
    # signs the URL only (no body params). Reconstruct it from the
    # x-forwarded headers Caddy adds.
    # Twilio signs the exact <Stream url=...> value from the TwiML, which we
    # built from PUBLIC_WS_URL. Verify against that canonical URL rather than
    # rebuilding it from x-forwarded-* headers — those vary by platform (Render
    # doesn't set them on a WS upgrade the way cloudflared did, which silently
    # failed verification and dropped every call on Render).
    public_ws = os.environ.get("PUBLIC_WS_URL", "").rstrip("/")
    if public_ws:
        signed_url = f"{public_ws}{websocket.url.path}"
    else:
        proto = "wss" if (websocket.headers.get("x-forwarded-proto") == "https"
                          or websocket.url.scheme == "wss") else "ws"
        host = websocket.headers.get("x-forwarded-host") or websocket.url.netloc
        signed_url = f"{proto}://{host}{websocket.url.path}"
    if websocket.url.query:
        signed_url += f"?{websocket.url.query}"
    if not _verify_twilio_signature(
        "GET",
        signed_url,
        {},
        websocket.headers.get("x-twilio-signature", ""),
    ):
        log.warning("twilio_stream: signature mismatch from %s", websocket.client)
        await websocket.close(code=4403, reason="invalid Twilio signature")
        return

    await websocket.accept()
    # Wait for the "start" event — that's where Twilio delivers the
    # custom <Parameter name="room" .../> from the TwiML, plus the
    # streamSid. URL query params are NOT preserved on the upgrade.
    room = ""
    stream_sid = ""
    try:
        while not (room and stream_sid):
            raw = await websocket.receive_text()
            evt = json.loads(raw)
            kind = evt.get("event", "")
            if kind == "start":
                stream_sid = evt["streamSid"]
                params = evt.get("start", {}).get("customParameters") or {}
                room = params.get("room", "") or websocket.query_params.get("room", "")
                break
            if kind in ("connected",):
                continue
    except (fastapi.WebSocketDisconnect, json.JSONDecodeError):
        await _safe_close(websocket)
        return

    async with _PENDING_LOCK:
        state = _PENDING.pop(room, None) if room else None
        if state is not None:
            _ACTIVE[room] = state
    if state is None:
        log.warning("twilio_stream: unknown room=%r — closing (have %d pending)", room, len(_PENDING))
        await _safe_close(websocket)
        return
    _log_event(state, "connected", stream_sid=stream_sid)

    state.stream_sid = stream_sid
    log.info("twilio_stream: bridge starting room=%s sid=%s", room, stream_sid)
    await _append_transcript(state, "SYSTEM", f"📤 Outbound call (gradbot) — room: {room}")
    # Emit business_prompt_build so the dashboard's discover_business_rooms()
    # sees this as a business call (it filters on this exact event name).
    _timeline_event(
        state, "business_prompt_build",
        mode="business", language=state.spec.language,
        business_name=state.spec.business_name or "",
    )

    # Open WAV writers under the call's recording dir. We record at 8 kHz to
    # match what's actually flowing on the wire (post-resample on the agent
    # side, pre-resample on the caller side).
    state.wav_caller = _open_wav(state.rec_dir / f"sip_caller.wav", sample_rate=TWILIO_ULAW_RATE)
    state.wav_agent = _open_wav(state.rec_dir / f"tts_direct.wav", sample_rate=TWILIO_ULAW_RATE)

    # Resample state — preserved across chunks for continuity.
    agent_resample_state = None  # 48k → 8k for outbound
    caller_resample_state = None  # 8k → 48k for inbound

    # Start the gradbot session — gated by the concurrency semaphore so
    # late entrants in a parallel batch wait for a slot to free instead
    # of hard-failing with "Concurrency limit exceeded".
    if _GRADBOT_SEMAPHORE is not None:
        if _GRADBOT_SEMAPHORE.locked():
            log.info("twilio_stream: room=%s waiting for gradbot slot", state.room_name)
        await _GRADBOT_SEMAPHORE.acquire()
    # Load what we know about this caller so the agent starts the call already
    # knowing them. Best-effort: never block or fail the call on memory.
    memory_digest = ""
    if state.tenant_id is not None:
        try:
            memory_digest = await memory_mod.render_digest(state.tenant_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("memory digest load failed: %s", exc)

    try:
        # gradbot.run() reads GRADIUM_API_KEY / OPENAI_API_KEY / LLM_BASE_URL /
        # LLM_MODEL from env when args are None — see gradbot_lib::GradbotClients::new.
        input_handle, output_handle = await gradbot.run(
            session_config=_make_session_config(
                state.spec,
                voice_id_override=state.voice_id_override,
                memory_digest=memory_digest,
            ),
            input_format=gradbot.AudioFormat.Ulaw,
            output_format=gradbot.AudioFormat.Ulaw,
        )
    except Exception:
        # If gradbot.run() itself raises (e.g. STT slot still wasn't
        # free even after the semaphore), release the slot so the next
        # caller can claim it. The semaphore release inside the call's
        # finally only runs on the success path.
        if _GRADBOT_SEMAPHORE is not None:
            _GRADBOT_SEMAPHORE.release()
        raise

    stop_event = asyncio.Event()
    pending_tool_tasks: set[asyncio.Task] = set()

    # --- Barge-in + backchannel fillers ---------------------------------
    # Barge-in (default on): when gradbot flags an agent audio frame as
    # `interrupted` (the caller started talking over the clone), tell Twilio to
    # drop whatever it still has buffered so the clone stops mid-sentence —
    # the same thing the gradbot browser demo does by resetting its playback
    # buffer. A kill-switch in case it ever misbehaves on a real call.
    barge_in_on = _env_flag("BARGE_IN", True)

    # Fillers (default OFF — experimental): gradbot streams a whole agent turn
    # at once, so the gap between the caller finishing and the first agent
    # audio is dead air. When ENABLE_FILLERS=1 we play a short clip in the
    # clone's own voice during that gap. This pokes audio into Twilio
    # out-of-band from gradbot, so it wants a real-call shakedown before being
    # turned on for the workshop.
    fillers_on = _env_flag("ENABLE_FILLERS", False)
    call_voice_id = state.voice_id_override or _VOICE_ID.get(
        (state.spec.language or "en").lower(), _VOICE_ID["en"]
    )
    awaiting_response = {"v": False}      # caller finished; agent hasn't replied yet
    filler_used_this_turn = {"v": False}  # at most one filler per think-gap
    filler_tasks: set[asyncio.Task] = set()
    # Barge-in guard: ignore interruptions within this many seconds of an agent
    # turn starting, so brief noise/echo right as the clone begins speaking
    # doesn't twitchily cut it off. Raise it if barge-in feels too sensitive,
    # lower toward 0 to make it more eager.
    barge_guard_s = _env_float("BARGE_IN_GUARD_S", 1.0)
    agent_turn = {"start": 0.0, "last": 0.0}  # monotonic timestamps for turn tracking
    # Tool window: while a tool (e.g. web_search) is running and briefly after,
    # gradbot flags the answer-turn's first audio frame as `interrupted` because
    # the new generation replaces the preamble/filler. Without this guard the
    # barge-in handler would read that as a caller talking over the clone and
    # flush Twilio's buffer, cutting the preamble off mid-word before the result
    # is spoken. Suppressing barge-in during this window lets the current TTS
    # finish smoothly and the result follow. Covers web_search's 8s timeout + a
    # margin for the answer's first frames.
    tool_window = {"until": 0.0}
    tool_window_s = _env_float("TOOL_BARGE_SUPPRESS_S", 12.0)
    # Duplicate-answer guard: gemma sometimes emits a SECOND answer turn right
    # after the first post-tool answer (e.g. two different weather readings from
    # one web_search). We arm when a tool finishes, count agent turns, and drop
    # the audio of any extra turn that starts within dup_answer_window_s of the
    # first answer — so the caller hears exactly one answer. A genuine follow-up
    # ("anything else?") comes later, outside the window, and is unaffected.
    post_tool = {"armed": False, "armed_at": 0.0, "answers": 0, "first_answer_at": 0.0}
    dup_drop = {"v": False}
    dup_answer_window_s = _env_float("DUP_ANSWER_WINDOW_S", 6.0)
    dup_arm_max_s = _env_float("DUP_ANSWER_ARM_MAX_S", 15.0)

    async def _maybe_play_filler() -> None:
        await asyncio.sleep(_FILLER_DELAY_S)
        if not awaiting_response["v"] or filler_used_this_turn["v"] or stop_event.is_set():
            return
        ulaw = await _render_filler_ulaw(call_voice_id)
        # Re-check after the render (a first-time cache miss is a full TTS
        # round-trip): if the real answer has started meanwhile, skip so we
        # never talk over the clone.
        if not ulaw or not awaiting_response["v"] or filler_used_this_turn["v"]:
            return
        filler_used_this_turn["v"] = True
        try:
            await websocket.send_text(json.dumps({
                "event": "media",
                "streamSid": state.stream_sid,
                "media": {"payload": base64.b64encode(ulaw).decode("ascii")},
            }))
            _log_event(state, "filler")
        except Exception:  # noqa: BLE001
            pass

    if fillers_on:
        # Pre-warm so the first gap can use it too. Fire-and-forget.
        _pw = asyncio.create_task(_render_filler_ulaw(call_voice_id))
        filler_tasks.add(_pw)
        _pw.add_done_callback(filler_tasks.discard)

    async def emit_event(payload: dict) -> None:
        # Hook for surfacing dtmf/tool events to a side-channel later
        # (Telegram, dashboard websocket, …). No-op for now.
        log.debug("gradbot event: %s", payload)

    async def _close_input_handle() -> None:
        """Idempotent best-effort close of the gradbot input handle.

        Both consumer and producer (and the outer finally) call this on
        their way out so the Gradium STT WebSocket releases its session
        slot the moment ANY of the three exits, not just the producer.
        Multiple calls are safe — the second is a no-op.
        """
        try:
            await input_handle.close()
        except Exception:  # noqa: BLE001
            pass

    async def consumer() -> None:
        """Read from gradbot, write back to Twilio + tee to wav."""
        try:
            while not stop_event.is_set():
                msg = await output_handle.receive()
                if msg is None:
                    return
                kind = getattr(msg, "msg_type", None)

                if kind == "audio":
                    nonlocal agent_resample_state
                    now = time.monotonic()
                    interrupted = barge_in_on and getattr(msg, "interrupted", False)
                    # Don't treat the agent's own post-tool continuation as a
                    # caller barge-in (see tool_window note above).
                    in_tool_window = now < tool_window["until"]
                    if (interrupted and not in_tool_window and agent_turn["start"]
                            and (now - agent_turn["start"]) >= barge_guard_s):
                        # Genuine barge-in (past the guard window): drop the
                        # agent audio Twilio still has queued so the clone stops,
                        # and reset resample state for a clean next turn.
                        try:
                            await websocket.send_text(json.dumps({
                                "event": "clear",
                                "streamSid": state.stream_sid,
                            }))
                        except Exception:  # noqa: BLE001
                            pass
                        agent_resample_state = None
                        awaiting_response["v"] = False
                        filler_used_this_turn["v"] = False
                        agent_turn["start"] = 0.0
                        _log_event(state, "barge_in")
                        continue
                    # Otherwise play normally — no interruption, or one suppressed
                    # inside the guard window. Track turn boundaries: a >0.5s gap
                    # since the last agent frame marks the start of a new turn.
                    new_turn = (not agent_turn["start"]) or (now - agent_turn["last"]) > 0.5
                    if new_turn:
                        agent_turn["start"] = now
                        # Duplicate-answer classification (see post_tool note).
                        if post_tool["armed"] and (now - post_tool["armed_at"]) <= dup_arm_max_s:
                            if post_tool["answers"] == 0:
                                post_tool["answers"] = 1
                                post_tool["first_answer_at"] = now
                                dup_drop["v"] = False
                            elif (now - post_tool["first_answer_at"]) < dup_answer_window_s:
                                dup_drop["v"] = True  # extra answer turn → drop it
                                log.info("call %s | suppressed duplicate post-tool answer turn",
                                         state.room_name)
                                _log_event(state, "dup_answer_suppressed")
                            else:
                                post_tool["armed"] = False
                                dup_drop["v"] = False
                        else:
                            post_tool["armed"] = False
                            dup_drop["v"] = False
                    agent_turn["last"] = now
                    # Drop the audio of a suppressed duplicate answer turn so the
                    # caller never hears the second, conflicting reply.
                    if dup_drop["v"]:
                        continue
                    awaiting_response["v"] = False
                    filler_used_this_turn["v"] = False
                    if state.first_agent_audio_at is None:
                        state.first_agent_audio_at = now
                    ulaw_8k, agent_resample_state = _resample_ulaw(
                        msg.data, GRADBOT_OUTPUT_RATE, TWILIO_ULAW_RATE, agent_resample_state,
                    )
                    payload_b64 = base64.b64encode(ulaw_8k).decode("ascii")
                    await websocket.send_text(json.dumps({
                        "event": "media",
                        "streamSid": state.stream_sid,
                        "media": {"payload": payload_b64},
                    }))
                    if state.wav_agent is not None:
                        try:
                            # Gap-fill: emit silence to bring the wav up to
                            # the wall-clock position before writing this
                            # chunk, so between-turn pauses are preserved.
                            now = time.monotonic()
                            expected = int((now - state.first_agent_audio_at) * TWILIO_ULAW_RATE)
                            gap_samples = expected - state.agent_samples_written
                            if gap_samples > 0:
                                state.wav_agent.writeframes(b"\x00\x00" * gap_samples)
                                state.agent_samples_written += gap_samples
                            chunk_pcm = _ulaw_to_pcm16(ulaw_8k)
                            state.wav_agent.writeframes(chunk_pcm)
                            state.agent_samples_written += len(chunk_pcm) // 2
                        except Exception as e:  # noqa: BLE001
                            log.debug("wav agent write failed: %s", e)

                elif kind == "stt_text":
                    text = (getattr(msg, "text", "") or "").strip()
                    if text:
                        # Stdout receipt so the live conversation is tailable in
                        # Render logs (no disk access needed to debug a call).
                        log.info("call %s | CALLER: %s", state.room_name, text)
                        await _append_transcript(state, "CALLER", f"[{state.spec.language}] {text}")
                        state.transcript_turns.append(("caller", text))
                        # New caller turn — disarm the duplicate-answer guard so a
                        # fresh question's answer is never mistaken for a duplicate.
                        post_tool["armed"] = False
                        dup_drop["v"] = False
                        # Caller just spoke — agent reply is pending. Arm a
                        # filler to cover the think-gap (no-op unless enabled).
                        if fillers_on:
                            awaiting_response["v"] = True
                            _ft = asyncio.create_task(_maybe_play_filler())
                            filler_tasks.add(_ft)
                            _ft.add_done_callback(filler_tasks.discard)
                        if not state.first_user_seen:
                            state.first_user_seen = True
                            _timeline_event(state, "business_first_final_transcript", text=text[:160])
                        # Phase moves out of setup on first user turn.
                        if state.business_state_phase == "setup":
                            _timeline_event(state, "business_state_change", old="setup", new="human", reason="first user turn")
                            state.business_state_phase = "human"

                elif kind == "tts_text":
                    text = sanitize(getattr(msg, "text", "") or "")
                    if text:
                        log.info("call %s | AGENT: %s", state.room_name, text)
                        await _append_transcript(state, "AGENT", text)
                        state.transcript_turns.append(("agent", text))
                        if not state.first_agent_seen:
                            state.first_agent_seen = True
                            _timeline_event(state, "business_first_agent_text", text=text[:160])
                        # Track first agent text for guards similar to LiveKit's.
                        if "introduce" in text.lower() and state.introduced_once:
                            log.info("gradbot: opener already spoken — guard logged")
                        state.introduced_once = state.introduced_once or any(
                            kw in text.lower() for kw in ("my name is", "je m'appelle", "calling on behalf", "je vous appelle")
                        )

                elif kind == "tool_call":
                    handle = gradbot.ToolHandle(msg.tool_call_handle, msg.tool_call)
                    # Suppress barge-in around the tool so the answer-turn doesn't
                    # flush the in-progress preamble/filler (see tool_window note).
                    tool_window["until"] = time.monotonic() + tool_window_s
                    task = asyncio.create_task(_handle_tool_call(handle, state, emit_event))
                    pending_tool_tasks.add(task)
                    task.add_done_callback(pending_tool_tasks.discard)

                    # Arm the duplicate-answer guard: count agent turns from here
                    # so a second, conflicting answer to the same tool result is
                    # dropped (see post_tool note).
                    def _arm_post_tool(_t: asyncio.Task) -> None:
                        nowm = time.monotonic()
                        # Arm ONCE per burst: gemma fires the same tool many
                        # times, and resetting the answer counter on each
                        # completion would let every answer look like the
                        # "first" and defeat the guard. Keep the existing count
                        # if we're still armed within the window.
                        if post_tool["armed"] and (nowm - post_tool["armed_at"]) <= dup_arm_max_s:
                            return
                        post_tool["armed"] = True
                        post_tool["armed_at"] = nowm
                        post_tool["answers"] = 0
                        post_tool["first_answer_at"] = 0.0
                    task.add_done_callback(_arm_post_tool)

                # If the LLM called end_business_call, stop the consumer too.
                if state.end_requested:
                    return
        except Exception:  # noqa: BLE001
            log.exception("gradbot consumer error")
        finally:
            stop_event.set()
            # Release the Gradium STT/TTS session slot the moment the
            # consumer exits. Without this the session leaks until the
            # producer's WS to Twilio drops, which can take many seconds
            # of dead-air on a real call.
            await _close_input_handle()

    async def producer() -> None:
        """Read from Twilio, push into gradbot."""
        try:
            while not stop_event.is_set():
                raw = await websocket.receive_text()
                evt = json.loads(raw)
                kind = evt.get("event")
                if kind == "media":
                    nonlocal caller_resample_state
                    if state.first_caller_audio_at is None:
                        state.first_caller_audio_at = time.monotonic()
                    audio_8k = base64.b64decode(evt["media"]["payload"])
                    audio_24k, caller_resample_state = _resample_ulaw(
                        audio_8k, TWILIO_ULAW_RATE, GRADBOT_INPUT_RATE, caller_resample_state,
                    )
                    # Guards on forwarding caller audio to gradbot (the WAV
                    # recording below still captures everything):
                    #  • opener guard — stay deaf to the caller during the
                    #    agent's opening turn.
                    #  • per-turn barge-in guard — for a short window after the
                    #    agent STARTS each turn, withhold caller audio so noise/
                    #    echo as the clone begins speaking can't trip a false
                    #    interruption at gradbot's (core-hardcoded) VAD. This is
                    #    the real lever for "barge-in too aggressive": the engine
                    #    cannot interrupt on audio it never receives. agent_turn
                    #    is set by the consumer when a new agent turn begins.
                    now_p = time.monotonic()
                    opener_guard = (
                        state.first_agent_audio_at is None
                        or (now_p - state.first_agent_audio_at) < state.opener_guard_seconds
                    )
                    turn_guard = bool(
                        agent_turn["start"]
                        and (now_p - agent_turn["start"]) < barge_guard_s
                    )
                    if not opener_guard and not turn_guard:
                        await input_handle.send_audio(audio_24k)
                    if state.wav_caller is not None:
                        try:
                            state.wav_caller.writeframes(_ulaw_to_pcm16(audio_8k))
                        except Exception as e:  # noqa: BLE001
                            log.debug("wav caller write failed: %s", e)
                elif kind == "stop":
                    log.info("twilio stop received")
                    return
                elif kind == "mark":
                    pass  # we don't currently use marks
        except fastapi.WebSocketDisconnect:
            log.info("twilio websocket disconnected")
        except Exception:  # noqa: BLE001
            log.exception("gradbot producer error")
        finally:
            stop_event.set()
            await _close_input_handle()

    try:
        await asyncio.gather(consumer(), producer(), return_exceptions=True)
    finally:
        # Backstop close — covers the path where both consumer and
        # producer raised before reaching their own finallies. Idempotent.
        await _close_input_handle()
        # Release the concurrency slot once the session is fully torn down.
        if _GRADBOT_SEMAPHORE is not None:
            _GRADBOT_SEMAPHORE.release()
        for t in filler_tasks:
            t.cancel()
        for t in pending_tool_tasks:
            t.cancel()
        if pending_tool_tasks:
            await asyncio.gather(*pending_tool_tasks, return_exceptions=True)
        _close_wav(state.wav_caller)
        _close_wav(state.wav_agent)
        # Compute the agent-vs-caller start offset so the stereo mix stays
        # time-aligned. Positive value = agent started speaking after the
        # caller line was already up.
        if state.first_caller_audio_at is not None and state.first_agent_audio_at is not None:
            agent_lead = state.first_agent_audio_at - state.first_caller_audio_at
        else:
            agent_lead = 0.0
        if _build_mixed_wav(state.rec_dir, agent_lead_seconds=agent_lead):
            log.info(
                "twilio_stream: mixed.wav written for room=%s (agent_lead=%.2fs)",
                state.room_name, agent_lead,
            )
        await _emit_completion(state)
        # Extract durable memories + push a Telegram summary. Best-effort.
        await _post_call_followups(state)
        # Persist the WS-side outcome into the calls table. Only writes if
        # the row is still 'pending', so a Twilio-side terminal status
        # that fired first wins.
        try:
            br = state.business_result or {}
            duration = max(0.0, time.time() - state.started_at)
            await tenants.record_call_end(
                room=state.room_name,
                status=br.get("status") or "unclear",
                answer=br.get("answer") or "",
                confidence=br.get("confidence") or "",
                duration_seconds=duration,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("record_call_end failed for room=%s: %s", state.room_name, e)
        await _safe_close(websocket)
        # Drop from the live registry so /calls/live stops listing it.
        async with _PENDING_LOCK:
            _ACTIVE.pop(state.room_name, None)


async def _safe_close(websocket: fastapi.WebSocket) -> None:
    try:
        await websocket.close()
    except Exception:  # noqa: BLE001
        pass


async def _emit_completion(state: _CallState) -> None:
    """Write a final sidecar JSON so the dashboard's auto-discovery treats
    this call as a real completed call (matches the LiveKit recorder's
    timeline.json.jsonl format approximately).
    """
    timeline_path = state.rec_dir / "timeline.json.jsonl"
    metrics_path = state.rec_dir / "metrics.json"
    duration = max(0.0, time.time() - state.started_at)

    # Append minimal events so detect_framework() (build_dashboard.py) tags
    # this call as gradbot via its event-name heuristic.
    try:
        with timeline_path.open("a", encoding="utf-8") as f:
            ts0 = int(state.started_at * 1e9)
            for ev in [
                {"t_ns": ts0, "event": "gradbot_session_started"},
                {"t_ns": int(time.time() * 1e9), "event": "gradbot_call_dispatched", "room": state.room_name},
                {"t_ns": int(time.time() * 1e9), "event": "gradbot_turn_complete", "result": state.business_result},
                {"t_ns": int(time.time() * 1e9), "event": "call_ended"},
            ]:
                f.write(json.dumps(ev) + "\n")
    except OSError as e:
        log.warning("timeline write failed: %s", e)

    try:
        metrics_path.write_text(json.dumps({
            "call_id": state.room_name,
            "framework": "gradbot",
            "duration_seconds": duration,
            "target_duration_ms": duration * 1000,
            "business_result": state.business_result or None,
        }, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("metrics write failed: %s", e)

    # mixed.wav is built by `_build_mixed_wav()` in the websocket finally
    # block (true stereo mix: caller=L, agent=R). Don't overwrite it here.


# --- Outbound dispatch (called from outbound.make_call when
#     orchestrator='gradbot') ------------------------------------------

async def dispatch_gradbot_call(
    *,
    to: str,
    spec: BusinessCallSpec,
    public_http_url: str | None = None,
    tenant_id: int | None = None,
) -> str:
    """Dial outbound via Twilio REST. Returns the room name (= recording dir).

    Required env (for the bridge server itself to be reachable):
      PUBLIC_HTTP_URL=https://your-tunnel.example.com
      PUBLIC_WS_URL=wss://your-tunnel.example.com

    ``tenant_id`` is recorded with the call row and used to release the
    rate-limit slot when the call ends. Operator-mode calls pass None.
    """
    public_http_url = (
        public_http_url
        or os.environ.get("PUBLIC_HTTP_URL", "").rstrip("/")
    )
    if not public_http_url:
        return "Error: PUBLIC_HTTP_URL not set — gradbot bridge needs a publicly reachable URL"

    # Normalise the destination to strict E.164 (digits + leading '+').
    # The dial form sometimes receives human-formatted numbers like
    # "+33 1 40 60 44 32"; without scrubbing, the spaces leak into the
    # room name → into the TwiML URL → Twilio rejects with HTTP 400
    # ("Url is not a valid URL").
    digits = "".join(ch for ch in (to or "") if ch.isdigit())
    if not digits:
        return "Error: 'to' must contain at least one digit"
    to = "+" + digits
    # Enforce the destination allowlist at the choke point every outbound path
    # funnels through (so /dial and the place_call tool are both covered).
    # Default-closed: see _outbound_destination_allowed.
    if not _outbound_destination_allowed(to):
        log.warning("dispatch refused: %s not allowed by OUTBOUND_ALLOWLIST", to)
        return "Error: destination not allowed (set OUTBOUND_ALLOWLIST or ALLOW_ARBITRARY_OUTBOUND)"
    # 16 random bytes so a pending room name can't be guessed and used to attach
    # to the Media Streams WebSocket before the real call connects.
    room_name = f"outbound-{digits}_{os.urandom(16).hex()}"
    rec_dir = RECORDINGS_ROOT / room_name
    rec_dir.mkdir(parents=True, exist_ok=True)
    # Mark for the dashboard's framework detector.
    (rec_dir / "framework.txt").write_text("gradbot\n")

    transcript_path = TRANSCRIPTS_ROOT / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{room_name}.md"

    # If the tenant has a cloned voice, use it. Operator calls (tenant_id=None)
    # fall through to the default _VOICE_ID per-language.
    voice_id_override: str | None = None
    if tenant_id is not None:
        try:
            t = await tenants.get_tenant_by_id(tenant_id)
            voice_id_override = (t or {}).get("voice_id") or None
        except Exception as e:  # noqa: BLE001
            log.warning("tenant lookup for voice_id failed: %s", e)

    state = _CallState(
        spec=spec,
        room_name=room_name,
        rec_dir=rec_dir,
        transcript_path=transcript_path,
        tenant_id=tenant_id,
        voice_id_override=voice_id_override,
    )
    _log_event(state, "ringing", destination=spec.destination, business_name=spec.business_name)
    async with _PENDING_LOCK:
        _PENDING[room_name] = state

    twiml_url = f"{public_http_url}/twilio/voice?room={room_name}"
    status_url = f"{public_http_url}/twilio/status?room={room_name}"

    # Twilio REST API call. Use the SDK already pinned in the project.
    try:
        from twilio.rest import Client as TwilioClient  # type: ignore
    except ImportError:
        return "Error: twilio SDK not installed"

    # Prefer Twilio API Key auth (TWILIO_API_KEY_SID + TWILIO_API_KEY_SECRET)
    # — that's what the Gradbots Infisical project provisions and what
    # production should use anyway. Fall back to the legacy
    # account_sid + auth_token if no API key is set in env.
    api_key_sid = os.environ.get("TWILIO_API_KEY_SID")
    api_key_secret = os.environ.get("TWILIO_API_KEY_SECRET")
    if api_key_sid and api_key_secret:
        twilio = TwilioClient(api_key_sid, api_key_secret, cfg.twilio_account_sid)
    else:
        twilio = TwilioClient(cfg.twilio_account_sid, cfg.twilio_auth_token)

    # Phone number: env override wins (Gradbots project has its own number),
    # otherwise fall back to the gizmo-voice-agent's TWILIO_PHONE_NUMBER.
    from_number = os.environ.get("TWILIO_FROM_NUMBER") or cfg.twilio_phone_number

    # Voicemail detection — Twilio decides synchronously (~3s) whether the
    # answerer is a human or machine, and surfaces the verdict via the
    # AnsweredBy form field on the TwiML POST + status callbacks. Disable
    # by setting TWILIO_MACHINE_DETECTION=disable.
    machine_detection = os.environ.get("TWILIO_MACHINE_DETECTION", "Enable")

    try:
        call = twilio.calls.create(
            to=to,
            from_=from_number,
            url=twiml_url,
            method="POST",
            status_callback=status_url,
            status_callback_method="POST",
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            machine_detection=machine_detection if machine_detection.lower() != "disable" else None,
        )
    except Exception as e:  # noqa: BLE001
        # Roll back the pending entry so we don't leak state.
        async with _PENDING_LOCK:
            _PENDING.pop(room_name, None)
        return f"Error: {e}"

    # Track the Twilio CallSid so the watchdog can hang up the call via
    # REST when the max-duration deadline is reached.
    state.twilio_call_sid = getattr(call, "sid", "")

    # Persist to the calls table so /history and /result-fallback work.
    try:
        await tenants.record_call_start(
            room=room_name,
            tenant_id=tenant_id,
            twilio_call_sid=state.twilio_call_sid,
            destination=to,
            task=spec.task,
            language=spec.language,
            business_name=spec.business_name,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("record_call_start failed for room=%s: %s", room_name, e)

    log.info(
        "gradbot dispatch: twilio_sid=%s room=%s to=%s tenant=%s twiml_url=%s",
        state.twilio_call_sid, room_name, to, tenant_id, twiml_url,
    )

    # Max-duration watchdog. gradbot has no Twilio-side cap; without this a
    # runaway call (e.g. the model gets stuck in a loop) racks up minutes.
    max_seconds = int(os.environ.get("MAX_CALL_DURATION_SECONDS", "600"))
    if max_seconds > 0 and state.twilio_call_sid:
        asyncio.create_task(_call_watchdog(state.twilio_call_sid, room_name, max_seconds, twilio))

    return room_name


async def _call_watchdog(call_sid: str, room: str, max_seconds: int, twilio_client) -> None:
    """Hang up via Twilio REST if the call outlives MAX_CALL_DURATION_SECONDS.

    Polls _ACTIVE to detect early termination — if the room is gone from
    the registry, the WS handler already cleaned up and there's nothing
    to do. Twilio's own max_call_duration is a per-trunk SIP setting and
    doesn't apply to PSTN-direct calls, so this is the actual hard limit.
    """
    deadline = asyncio.get_event_loop().time() + max_seconds
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(15.0)
        async with _PENDING_LOCK:
            still_in_flight = room in _PENDING or room in _ACTIVE
        if not still_in_flight:
            return
    log.warning("watchdog: max duration reached for room=%s — hanging up call %s", room, call_sid)
    try:
        await asyncio.to_thread(
            lambda: twilio_client.calls(call_sid).update(status="completed")
        )
    except Exception as e:  # noqa: BLE001
        log.warning("watchdog: hangup failed for %s: %s", call_sid, e)
