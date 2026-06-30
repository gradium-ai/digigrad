"""Voice-note chat with your clone, over Telegram — no phone call needed.

Pipeline per voice note:
    Telegram OGG/Opus
        → ffmpeg → PCM16 mono 16k → Gradium STT → user text
        → LLM chat (system prompt carries the tenant's memory digest)
        → reply text
        → Gradium TTS (the tenant's cloned voice) → WAV
        → ffmpeg → OGG/Opus → Telegram voice note

Memory: the digest is injected so the clone "knows" the caller; after each
exchange we run the post-call extractor so it keeps learning. This reuses the
same memory + LLM plumbing as the live phone path — it's the phone-free way to
exercise and demo the clone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

import aiohttp

from . import audio_io
from . import memory as memory_mod
from . import places
from . import websearch
from .business_agent import language_name

log = logging.getLogger(__name__)

_STT_SAMPLE_RATE = 24000  # Gradium STT operates at 24 kHz
_MAX_HISTORY_TURNS = 12  # rolling user+assistant messages kept for context
_LLM_TIMEOUT = 30
_MAX_TOOL_ROUNDS = 3  # cap tool → LLM round-trips per reply
_WEB_SEARCH_TIMEOUT = 10.0  # seconds for one Linkup call
_FIND_BUSINESS_TIMEOUT = 8.0  # seconds for one Google Places call


def _gradium_client():
    import gradium

    api_key = os.environ.get("GRADIUM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GRADIUM_API_KEY is required for voice chat")
    base_url = os.environ.get("GRADIUM_BASE_URL", "").strip() or None
    if base_url:
        return gradium.GradiumClient(api_key=api_key, base_url=base_url)
    return gradium.GradiumClient(api_key=api_key)


async def transcribe(ogg_bytes: bytes) -> str:
    """Voice note → text via Gradium STT (24 kHz int16 samples)."""
    import gradium
    import numpy as np

    pcm = await asyncio.to_thread(audio_io.ogg_to_pcm16, ogg_bytes, _STT_SAMPLE_RATE)
    samples = np.frombuffer(pcm, dtype=np.int16)
    client = _gradium_client()
    setup = gradium.STTSetup(model_name="default", input_format="pcm")
    result = await client.stt(setup, samples, sample_rate=_STT_SAMPLE_RATE)
    return (getattr(result, "text", "") or "").strip()


async def synthesize(text: str, voice_id: str) -> bytes:
    """Reply text → OGG/Opus voice note in the tenant's cloned voice."""
    import gradium

    client = _gradium_client()
    setup = gradium.TTSSetup(model_name="default", voice_id=voice_id, output_format="wav")
    result = await client.tts(setup, text)
    wav = getattr(result, "raw_data", None)
    if not wav:
        raise RuntimeError("Gradium TTS returned no audio")
    return await asyncio.to_thread(audio_io.wav_to_ogg_opus, wav)


def _system_prompt(
    name: str,
    memory_digest: str,
    language: str = "en",
    channel: str = "voice",
    web_search_enabled: bool = False,
    places_enabled: bool = False,
) -> str:
    lang = language_name(language)
    block = (
        "\nWHAT YOU ALREADY KNOW ABOUT THEM (from past chats — use it naturally, "
        "don't recite):\n" + memory_digest + "\n"
        if memory_digest.strip() else ""
    )
    is_text = channel == "text"
    if is_text:
        medium = "casual text chat"
        style = (
            "You're texting, so reply in clear written text. You MAY use short lists "
            "and include specifics like names and addresses when they're what's asked for."
        )
    else:
        medium = "casual voice chat"
        style = (
            "Be warm, concise, and natural — one or two sentences per reply, like a "
            "quick voice message. This will be read aloud by text-to-speech, so never "
            "use markdown, bullet points, emoji, or stage directions."
        )
    # Tool guidance — applies to BOTH channels. For voice, keep answers brief and
    # link-free since they're spoken; for text, links and lists are fine.
    if web_search_enabled:
        style += (
            " You can look things up on the live web with the web_search tool. Use it "
            "whenever they ask about current facts that may be outside your training "
            "knowledge — today's news, weather, recent events, prices, scores, anything "
            "time-sensitive — instead of guessing."
        )
        style += (
            " After searching, answer in your own words and add the source links on "
            "their own lines." if is_text
            else " After searching, give the answer briefly in your own words — do NOT "
            "read out URLs or source links."
        )
    if places_enabled:
        style += (
            " To find places or businesses nearby — restaurants, cafes, shops — or look "
            "up a place's details or phone number, use the find_business tool (NOT "
            "web_search). Finding and listing options is a valid request on its own."
        )
        style += (
            " List the top few with their rating and address." if is_text
            else " Read back the top two or three by name and rating, briefly."
        )
    return (
        f"You are {name}'s personal assistant, speaking as their voice clone in a "
        f"{medium}. {style}\n"
        f"{block}"
        "\nMEMORY: When they tell you something durable about themselves — a "
        "preference, dietary restriction, allergy, name, relationship, plan, or "
        "recurring need — call the remember tool with one concise fact, AND confirm "
        "briefly that you've saved it. If they ask you to remember something, you MUST "
        "call remember; saying 'I'll keep that in mind' without calling it does NOT "
        "save it. If you're unsure what you already know, call recall.\n"
        "\nAPPLY WHAT YOU KNOW: Before recommending food, drinks, places, or products, "
        "check what you know about them and use it. If a recommendation conflicts with a "
        "known restriction or allergy (e.g. they're lactose intolerant and you're "
        "suggesting ice cream), say so and steer them to a suitable option — e.g. flag "
        "dairy-free choices. Acting on a known constraint is being helpful, not "
        "'reciting' — do it proactively.\n"
        "\nIMPORTANT — answer the request directly in THIS reply. You have no way to "
        "send anything separately, 'put together' a list later, or follow up in another "
        "message; there is no background task. If they ask for a list or details, give "
        "it now. If you don't actually have the information (e.g. live or current data), "
        "say so plainly — never promise to send or compile something you can't deliver "
        "right here.\n"
        f"\nReply in {lang} unless they switch languages."
    )


def web_search_available() -> bool:
    """True when Linkup is configured, so the text assistant can search the web."""
    return bool(os.environ.get("LINKUP_API_KEY", "").strip())


def _web_search_tool_schema() -> dict:
    """OpenAI-style function schema for the web_search tool (text chat)."""
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web for current facts that may be outside your "
                "training knowledge — today's news, weather, recent events, prices, "
                "sports scores, anything time-sensitive or freshly changed. Returns a "
                "short sourced answer. Use it whenever the user asks something you're "
                "not confident is current; do NOT guess at recent facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language question. Be specific; include the "
                            "entity, date, or place when known."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }


async def _run_web_search(query: str) -> dict:
    """Execute one web_search tool call; never raises — returns a dict the LLM
    can read, with an `error` key when the search couldn't run."""
    query = (query or "").strip()
    if not query:
        return {"error": "Empty search query — ask what to look up."}
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(websearch.search, query), timeout=_WEB_SEARCH_TIMEOUT
        )
    except asyncio.TimeoutError:
        log.warning("text chat: web_search timed out for %r", query[:120])
        return {"error": "The search took too long — say you couldn't pull it up."}
    except websearch.WebSearchNotConfigured:
        return {"error": "Web search isn't set up."}
    except websearch.WebSearchError as exc:
        log.warning("text chat: web_search failed: %s", exc)
        return {"error": "The search failed — say you couldn't look it up right now."}


def find_business_available() -> bool:
    """True when Google Places is configured, so the assistant can find places."""
    return places.available()


def _find_business_tool_schema() -> dict:
    """OpenAI-style function schema for the find_business (Google Places) tool."""
    return {
        "type": "function",
        "function": {
            "name": "find_business",
            "description": (
                "Find real businesses or places nearby and their details — name, "
                "rating, address, and phone number — via Google Places. Use this for "
                "ANY 'find / recommend / what's a good X near Y' request, e.g. 'find "
                "good Indian restaurants nearby' or 'a coffee shop near my hotel'. "
                "Returns a ranked list, best first. Use this (NOT web_search) to find "
                "places and their phone numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to find, with the area/landmark — e.g. 'highly "
                            "rated Indian restaurant near Clancy Hotel, San Francisco'."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }


async def _run_find_business(query: str) -> dict:
    """Execute one find_business tool call; never raises — returns a dict the LLM
    can read, with an `error` key when the lookup couldn't run."""
    query = (query or "").strip()
    if not query:
        return {"error": "Empty query — ask what place to find."}
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(places.find_businesses, query),
            timeout=_FIND_BUSINESS_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning("chat: find_business timed out for %r", query[:120])
        return {"error": "The lookup took too long — say you couldn't pull it up."}
    except places.PlacesNotConfigured:
        return {"error": "Place lookup isn't set up."}
    except places.PlacesError as exc:
        log.warning("chat: find_business failed: %s", exc)
        return {"error": "The lookup failed — say you couldn't find it right now."}
    return {"results": results}


def _remember_tool_schema() -> dict:
    """OpenAI-style schema for the remember tool — same store the call uses."""
    return {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save ONE durable fact about the user for future chats and calls — a "
                "preference, dietary restriction, name, relationship, plan, or recurring "
                "need. Call this whenever the user tells you something worth keeping or "
                "explicitly asks you to remember it (e.g. 'I'm lactose intolerant', "
                "'remember my wife's name is Mei'). One concise fact per call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "One short, self-contained fact, e.g. 'The user is lactose intolerant.'",
                    },
                },
                "required": ["fact"],
            },
        },
    }


def _recall_tool_schema() -> dict:
    """OpenAI-style schema for the recall tool — searches saved memory."""
    return {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Look up what you already know about the user from past chats and calls. "
                "Call this when they refer to something from before, ask what you know, "
                "or when a known preference/constraint (diet, allergy, budget, location) "
                "could affect your answer — e.g. before recommending food or places."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional topic to filter on (e.g. 'diet'). Omit to get recent facts.",
                    },
                },
                "required": [],
            },
        },
    }


async def _run_remember(tenant_id: int, fact: str, room: str) -> dict:
    """Persist one fact; never raises. Same memory store as the call path."""
    fact = (fact or "").strip()
    if not fact:
        return {"error": "Empty fact — nothing to remember."}
    try:
        saved = await memory_mod.add_memory(
            tenant_id, fact, source="remember_tool", room=room
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("chat: remember failed: %s", exc)
        return {"error": "Couldn't save that right now."}
    return {"saved": saved, "fact": fact}


async def _run_recall(tenant_id: int, topic: str) -> dict:
    """Search saved memory; never raises."""
    try:
        facts = await memory_mod.search_memories(tenant_id, topic or "")
    except Exception as exc:  # noqa: BLE001
        log.warning("chat: recall failed: %s", exc)
        return {"facts": []}
    return {"facts": facts}


async def _chat(messages: list[dict], tools: list[dict] | None = None,
                max_tokens: int = 300) -> dict:
    """One LLM completion against the configured OpenAI-compatible endpoint.

    Returns the raw assistant message dict (so callers can inspect tool_calls);
    pass ``tools`` to expose function calling.
    """
    base = os.environ.get("LLM_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("LLM_MODEL", "").strip()
    if not base or not model:
        raise RuntimeError("LLM_BASE_URL / LLM_MODEL not set")
    api_key = (os.environ.get("OPENAI_API_KEY", "").strip()
               or os.environ.get("GRADIUM_API_KEY", "").strip())
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict = {
        "model": model, "messages": messages,
        "temperature": 0.6, "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            f"{base}/chat/completions", json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=_LLM_TIMEOUT),
        ) as r:
            r.raise_for_status()
            data = await r.json()
    # An OpenAI-compatible endpoint can return an error/quota body (or an
    # empty choices list) with a 2xx — guard before indexing so a bad
    # response surfaces as a handled error, not a KeyError/IndexError that
    # kills the whole voice reply.
    choices = (data or {}).get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM returned no choices: {str(data)[:300]}")
    return choices[0]["message"]


async def _complete_with_tools(messages: list[dict], tools: list[dict] | None,
                               max_tokens: int, tenant_id: int = 0,
                               room: str = "") -> str:
    """Run the LLM, servicing tool calls, until it returns prose.

    ``messages`` is mutated in place with the assistant + tool turns. Without
    tools this is a single completion. ``tenant_id``/``room`` scope the
    remember/recall tools to the same memory store the call path uses.
    """
    msg: dict = {}
    for _ in range(_MAX_TOOL_ROUNDS if tools else 1):
        msg = await _chat(messages, tools=tools, max_tokens=max_tokens)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            break
        messages.append(msg)  # assistant turn that requested the tools
        for call in tool_calls:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {}
            if fn.get("name") == "web_search":
                result = await _run_web_search(args.get("query", ""))
            elif fn.get("name") == "find_business":
                result = await _run_find_business(args.get("query", ""))
            elif fn.get("name") == "remember":
                result = await _run_remember(tenant_id, args.get("fact", ""), room)
            elif fn.get("name") == "recall":
                result = await _run_recall(tenant_id, args.get("topic", ""))
            else:
                result = {"error": f"Unknown tool {fn.get('name')!r}."}
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id"),
                "content": json.dumps(result),
            })
    else:
        # Exhausted the round budget while still asking for tools — force a
        # final prose answer with what we have.
        msg = await _chat(messages, max_tokens=max_tokens)
    return (msg.get("content") or "").strip()


async def reply(
    tenant: dict,
    history: list[dict],
    user_text: str,
    channel: str = "voice",
) -> str:
    """Produce the clone's reply to user_text, given rolling history.

    Reads the tenant's memory digest into the system prompt; mutates ``history``
    in place (appends the user + assistant turns, trimmed). Fire-and-forget
    memory growth happens in the caller after the reply is sent.
    """
    tenant_id = int(tenant["id"])
    digest = await memory_mod.render_digest(tenant_id)
    # Tools work on BOTH channels (typed text and voice notes). For voice the
    # system prompt steers answers to be brief and link-free, since they're
    # read aloud by TTS.
    use_web = web_search_available()
    use_places = find_business_available()
    system = _system_prompt(
        tenant.get("name") or "the user", digest,
        channel=channel, web_search_enabled=use_web, places_enabled=use_places,
    )
    messages = [{"role": "system", "content": system}, *history,
                {"role": "user", "content": user_text}]
    # remember/recall are always available — same memory store the call path
    # uses — so "remember X" saves identically whether typed, sent as a voice
    # note, or said on a call. Channel only changes phrasing, never capability.
    tools: list[dict] = [_remember_tool_schema(), _recall_tool_schema()]
    if use_web:
        tools.append(_web_search_tool_schema())
    if use_places:
        tools.append(_find_business_tool_schema())
    room = "telegram-text" if channel == "text" else "telegram-voice"
    answer = await _complete_with_tools(
        messages, tools, max_tokens=500 if channel == "text" else 300,
        tenant_id=tenant_id, room=room,
    )
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    del history[:-_MAX_HISTORY_TURNS]  # keep the tail only
    return answer


async def learn_from_exchange(tenant_id: int, user_text: str, reply_text: str) -> int:
    """Grow memory from one exchange (best-effort)."""
    return await memory_mod.extract_and_store(
        tenant_id, [("caller", user_text), ("agent", reply_text)], room="telegram-voice"
    )


# Intents the Telegram bot can route a typed message to (besides plain chat).
# Each maps to an existing command handler in bot.py.
INTENTS = ("translate", "callme", "call", "history", "status", "voice", "clear_voice", "web", "chat")

_INTENT_SYSTEM = (
    "You are the intent router for a personal voice-assistant Telegram bot. "
    "Read the user's message and reply with EXACTLY ONE word from this list, "
    "nothing else:\n"
    "- translate: they want to translate an audio clip / voice note into another language.\n"
    "- callme: they want the assistant to phone THEM.\n"
    "- call: they want to place an outbound call to someone or a business.\n"
    "- history: they want to see their past or recent calls.\n"
    "- status: they want to see calls currently live / in progress.\n"
    "- voice: they're asking about their cloned-voice status.\n"
    "- clear_voice: they want to remove or re-clone their voice.\n"
    "- web: they want the web dashboard link.\n"
    "- chat: ANYTHING else — normal conversation, questions, or requests to "
    "write/explain/summarize/etc.\n"
    "When in doubt, answer chat."
)


async def classify_intent(text: str) -> str:
    """Map a typed message to one bot intent (see INTENTS), defaulting to 'chat'.

    Best-effort: any LLM error or unrecognized output falls back to 'chat' so
    normal conversation always works even if routing misfires.
    """
    text = (text or "").strip()
    if not text:
        return "chat"
    try:
        msg = await _chat(
            [{"role": "system", "content": _INTENT_SYSTEM},
             {"role": "user", "content": text}],
            max_tokens=4,
        )
        raw = (msg.get("content") or "").strip().lower()
        tokens = re.findall(r"[a-z_]+", raw)
        word = tokens[0] if tokens else ""
        return word if word in INTENTS else "chat"
    except Exception:  # noqa: BLE001 - routing must never break chat
        log.warning("intent classify failed; defaulting to chat", exc_info=True)
        return "chat"


_CALL_LANGS = ("en", "fr", "pt")

_CALL_PARSE_SYSTEM = (
    "Extract outbound-call details from the user's request. Reply with ONLY a "
    "JSON object (no prose, no code fences) with these keys:\n"
    '  "to": the phone number in E.164 form (e.g. "+33618286290"), or "" if none is given.\n'
    '  "task": a short instruction describing what to say on the call, in the third '
    'person (e.g. "Let them know I will be 20 minutes late for dinner").\n'
    '  "language": one of en, fr, pt — the language to speak on the call; default '
    '"en" unless the request clearly implies another.\n'
    'Example → {"to": "+33618286290", "task": "Let them know I will be late for dinner", "language": "en"}'
)


async def parse_call_request(text: str) -> dict:
    """Extract {to, task, language} from a natural-language call request.

    Returns {} if no usable phone number is found or on any error, so the caller
    can fall back to asking for it explicitly.
    """
    try:
        msg = await _chat(
            [{"role": "system", "content": _CALL_PARSE_SYSTEM},
             {"role": "user", "content": text or ""}],
            max_tokens=120,
        )
        content = (msg.get("content") or "").strip()
        m = re.search(r"\{.*\}", content, re.S)  # tolerate stray fences/prose
        data = json.loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001
        log.warning("call-request parse failed", exc_info=True)
        return {}
    to = "+" + "".join(ch for ch in str(data.get("to", "")) if ch.isdigit())
    if to == "+":
        return {}
    lang = str(data.get("language") or "en").strip().lower()
    return {
        "to": to,
        "task": str(data.get("task") or "").strip(),
        "language": lang if lang in _CALL_LANGS else "en",
    }
