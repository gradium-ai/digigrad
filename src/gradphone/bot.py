"""Telegram trigger for gradphone — self-serve registration + dashboard link.

Run:
    python -m gradphone.bot

Required env:
    TELEGRAM_BOT_TOKEN     — from @BotFather
    GRADBOT_BRIDGE_URL     — default http://127.0.0.1:8082
    BRIDGE_API_KEY         — same value the bridge enforces on /dial
                             and used to sign magic-link tokens.
    PUBLIC_HTTP_URL        — needed for /web to generate a public URL.

Commands:
    /whoami    — show Telegram ID + registration state
    /register  — self-serve: anyone can register; rate-limited per tenant
    /call      — guided call flow
    /history   — last 10 calls placed by you
    /status    — currently in-flight calls (yours only)
    /web       — DM a magic link to the web dashboard (5-min expiry)
    /translate — real-time voice translation in your cloned voice
    /cancel    — abort a /call mid-way
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
from typing import Optional

import aiohttp
from dotenv import load_dotenv
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    TypeHandler,
    filters,
)
from telegram import Update as _Update

from . import memory as _memory
from . import tenants as _tenants_db
from . import translate as _translate
from . import voice_chat as _voice_chat
from . import voices as _voices
from .dial import _auth_headers, _format_result, dial, wait_for_result
from .sessions import make_magic_token

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gradphone.bot")

ASK_TO, ASK_TASK, ASK_LANG, CONFIRM = range(4)
LANGUAGES = ["en", "fr", "pt"]
MAX_HISTORY_DISPLAYED = 10
# Stock Gradium voice for the post-clone A/B sample (Arthur, the bridge's en
# default). Override via env if the account uses different built-ins.
_AB_DEFAULT_VOICE_ID = os.environ.get("AB_DEFAULT_VOICE_ID", "3jUdJyOi9pgbxBTK")


def _bridge_url() -> str:
    return os.environ.get("GRADBOT_BRIDGE_URL", "http://127.0.0.1:8082").rstrip("/")


async def _fetch_tenant(telegram_id: int) -> Optional[dict]:
    """Look up the tenant via the bridge's /tenants/{telegram_id} endpoint."""
    async with aiohttp.ClientSession() as sess:
        try:
            async with sess.get(
                f"{_bridge_url()}/tenants/{telegram_id}",
                headers=_auth_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                data = await r.json()
        except aiohttp.ClientError as e:
            log.warning("tenant lookup failed: %s", e)
            return None
    if not data.get("ok"):
        return None
    return data.get("tenant")


class _BridgeDown(Exception):
    """The bridge couldn't be reached or returned an unusable response."""


async def _bridge_json(method: str, path: str, **kw) -> dict:
    """Call the bridge and return parsed JSON, or raise _BridgeDown.

    Centralizes the failure handling the command handlers need: network
    errors, timeouts, 5xx, and non-JSON bodies all become _BridgeDown so a
    handler can show "couldn't reach the bridge" instead of crashing.
    """
    kw.setdefault("headers", _auth_headers())
    kw.setdefault("timeout", aiohttp.ClientTimeout(total=10))
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.request(method, f"{_bridge_url()}{path}", **kw) as r:
                if r.status >= 500:
                    raise _BridgeDown(f"bridge returned HTTP {r.status}")
                return await r.json()
    except _BridgeDown:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        raise _BridgeDown(f"network error: {e}") from e
    except ValueError as e:  # non-JSON body (json.JSONDecodeError ⊂ ValueError)
        raise _BridgeDown(f"unexpected response: {e}") from e


# ─── Command handlers ────────────────────────────────────

async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if tenant:
        has_voice = bool(tenant.get("voice_id"))
        lead = (
            "🎙️ Send me a voice note to talk to your clone, or /translate to hear "
            "yourself in another language."
            if has_voice else
            "🎙️ Send me a 20–30s voice note and I'll clone your voice."
        )
        await update.message.reply_text(
            f"Welcome back, {tenant['name']}.\n\n"
            f"{lead}\n\n"
            "/callme — your clone calls your phone\n"
            "/translate — hear yourself in another language\n"
            "/reset — wipe your data · /checkup — system status\n"
            "/voice /history /status /whoami"
        )
        return
    await update.message.reply_text(
        "Hi — I'm gradphone. I can clone your voice, chat and translate in it, "
        "and even answer your phone.\n\n"
        "1. Send /register to create your account.\n"
        "2. Then send me a 20–30 second voice note — I'll clone your voice and "
        "reply in it.\n\n"
        "/whoami shows your Telegram ID."
    )


async def whoami(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    lines = [
        f"Telegram ID: {user.id if user else 'unknown'}",
        f"Username:    @{user.username if user and user.username else '-'}",
        f"Registered:  {'yes (tenant_id=' + str(tenant['id']) + ')' if tenant else 'no — send /register'}",
    ]
    await update.message.reply_text("\n".join(lines))


async def register(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """One-time owner setup.

    Access is already restricted to the owner by the gatekeeper
    (ALLOWED_TELEGRAM_IDS), so this just creates the owner's profile row the
    first time and is idempotent thereafter.
    """
    user = update.effective_user
    if user is None:
        return
    existing = await _fetch_tenant(user.id)
    if existing:
        await update.message.reply_text(
            f"You're already registered (tenant_id={existing['id']}, "
            f"name: {existing['name']}). Send /call to place a call, "
            "or /web for the dashboard."
        )
        return
    name = user.full_name or user.username or f"user_{user.id}"
    try:
        data = await _bridge_json(
            "POST", "/tenants", json={"telegram_id": user.id, "name": name},
        )
    except _BridgeDown as e:
        log.warning("register: %s", e)
        await update.message.reply_text(
            "Couldn't reach the bridge — make sure it's running, then try /register again."
        )
        return
    if not data.get("ok"):
        await update.message.reply_text(f"Registration failed: {data.get('error', 'unknown')}")
        return
    contact_kb = ReplyKeyboardMarkup(
        [[KeyboardButton("📱 Share my number", request_contact=True)]],
        resize_keyboard=True, one_time_keyboard=True,
    )
    await update.message.reply_text(
        f"Registered as <code>{html.escape(name)}</code> "
        f"(tenant_id={data.get('tenant_id')}).\n\n"
        "Share your number so that when you call the agent it recognizes you "
        "and answers as your personal assistant (with your voice + memory). "
        "Tap below, or send a voice note to clone your voice.",
        parse_mode="HTML",
        reply_markup=contact_kb,
    )


async def save_contact(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Store the tenant's phone (shared via Telegram contact) for caller-ID
    identity on inbound calls. Only accepts the user's OWN contact."""
    user = update.effective_user
    contact = update.message.contact if update.message else None
    if user is None or contact is None:
        return
    if contact.user_id and contact.user_id != user.id:
        await update.message.reply_text(
            "Please share your own number, not someone else's.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return
    tenant = await _fetch_tenant(user.id)
    if not tenant:
        await update.message.reply_text("Run /register first.", reply_markup=ReplyKeyboardRemove())
        return
    await _tenants_db.set_tenant_phone(int(tenant["id"]), contact.phone_number)
    await update.message.reply_text(
        "Got it — when you call the agent from this number, it'll greet you as "
        "your own assistant. Send a voice note next to clone your voice.",
        reply_markup=ReplyKeyboardRemove(),
    )


async def web(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a signed magic link that logs the tenant into the web dashboard."""
    user = update.effective_user
    if user is None:
        return
    tenant = await _fetch_tenant(user.id)
    if not tenant:
        await update.message.reply_text("Run /register first.")
        return
    public = os.environ.get("PUBLIC_HTTP_URL", "").rstrip("/")
    if not public:
        await update.message.reply_text(
            "Web dashboard isn't reachable — PUBLIC_HTTP_URL not set on the bridge."
        )
        return
    token = make_magic_token(int(tenant["id"]))
    link = f"{public}/ui/auth?token={token}"
    await update.message.reply_text(
        "Open this link to access your dashboard:\n"
        f"{link}\n\n"
        "Valid for 5 minutes. Once signed in, the session lasts 7 days.",
        disable_web_page_preview=True,
    )


async def call_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await update.message.reply_text("Run /register first.")
        return ConversationHandler.END
    if not tenant.get("is_active", 1):
        await update.message.reply_text(
            "Your account is inactive. Contact the operator."
        )
        return ConversationHandler.END
    context.user_data.clear()
    context.user_data["tenant_id"] = tenant["id"]
    await update.message.reply_text(
        "What number should I call? E.164 format (e.g. +33144581010)."
    )
    return ASK_TO


async def got_to(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    raw = (update.message.text or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) < 7:
        await update.message.reply_text("That doesn't look like a phone number. Try again or /cancel.")
        return ASK_TO
    context.user_data["to"] = "+" + digits
    await update.message.reply_text(
        "Got it. What should I ask or do on the call? "
        "Be specific — the agent will follow these instructions verbatim."
    )
    return ASK_TASK


async def got_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    task = (update.message.text or "").strip()
    if len(task) < 5:
        await update.message.reply_text("Task too short. Give me a real instruction or /cancel.")
        return ASK_TASK
    context.user_data["task"] = task
    keyboard = [[InlineKeyboardButton(code.upper(), callback_data=f"lang:{code}") for code in LANGUAGES]]
    await update.message.reply_text(
        "Language for the call?",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return ASK_LANG


async def got_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    code = query.data.split(":", 1)[1]
    if code not in LANGUAGES:
        await query.edit_message_text("Unknown language. /call to start over.")
        return ConversationHandler.END
    context.user_data["language"] = code
    to = context.user_data["to"]
    task = context.user_data["task"]
    keyboard = [[
        InlineKeyboardButton("Place call", callback_data="confirm:yes"),
        InlineKeyboardButton("Cancel", callback_data="confirm:no"),
    ]]
    await query.edit_message_text(
        f"Ready to call:\n• To: {to}\n• Language: {code}\n• Task: {task}",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return CONFIRM


async def confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data != "confirm:yes":
        await query.edit_message_text("Cancelled.")
        return ConversationHandler.END

    to = context.user_data["to"]
    task = context.user_data["task"]
    language = context.user_data["language"]
    tenant_id = context.user_data["tenant_id"]
    await query.edit_message_text(f"Dialing {to}…")

    out = await dial(to=to, reason=task, language=language, tenant_id=tenant_id)
    if out.startswith("Error"):
        await query.message.reply_text(out)
        return ConversationHandler.END

    room = out
    await query.message.reply_text(
        f"Call placed (room: <code>{html.escape(room)}</code>). "
        "I'll post the result here when the call ends.",
        parse_mode="HTML",
    )
    _spawn_bg(_report_call_result(query.message, room))
    return ConversationHandler.END


async def cancel(update: Update, _: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


# Background tasks must be referenced or asyncio may garbage-collect them
# mid-flight. /callme spawns a poller per call; keep a strong ref until done.
_BG_TASKS: set[asyncio.Task] = set()


def _spawn_bg(coro) -> None:
    """Run a coroutine as a referenced background task (see _BG_TASKS)."""
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


async def _report_call_result(message, room: str) -> None:
    """Poll the bridge for a business call's result and post it to the chat.

    Runs as a background task: a call can take minutes, and awaiting it inside
    the confirm handler would hold that update slot (and before concurrent
    updates, the entire bot) hostage until the call ended."""
    try:
        data = await wait_for_result(room)
        formatted = _format_result(data)
    except Exception as exc:  # noqa: BLE001
        log.warning("call result poll failed for room=%s: %s", room, exc)
        return
    try:
        await message.reply_text(
            f"<pre>{html.escape(formatted)}</pre>", parse_mode="HTML"
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("call result notify failed for room=%s: %s", room, exc)

# A /callme call can't outlive the bridge's MAX_CALL_DURATION_SECONDS (180 for
# the workshop). 220s comfortably covers a full-length connected call plus the
# ring/no-answer window, so we always see a terminal /result.
CALLME_RESULT_DEADLINE = 220.0


def _callme_outcome_message(to: str, data: dict) -> Optional[str]:
    """Turn a /result payload into a user-facing /callme outcome line, or None
    if the call is still in progress (timeout/missing) and we should stay quiet
    rather than send a misleading message."""
    if data.get("status") != "complete":
        return None
    result = data.get("result") or {}
    tcs = (result.get("twilio_call_status") or "").lower()
    answered_by = (result.get("answered_by") or "").lower()
    esc = html.escape(to)
    if tcs in {"busy", "no-answer", "canceled"}:
        return (
            f"📵 Couldn't reach you at <code>{esc}</code> — the line was busy or "
            "there was no answer. Run /callme to try again."
        )
    if tcs == "failed":
        return (
            f"⚠️ The call to <code>{esc}</code> failed — the number may be "
            "unreachable. Check it's correct and in E.164 (e.g. +14155551234)."
        )
    if answered_by.startswith("machine") or answered_by == "fax":
        return (
            f"📭 Reached voicemail at <code>{esc}</code>, not you. Run /callme "
            "again when you can pick up."
        )
    # Call connected and ran. Stay silent here: the bridge's _post_call_followups
    # already DMs the tenant ("☎️ Assistant call ended …") on a connected call.
    # This poller exists to cover the cases the bridge never reaches — busy /
    # no-answer / voicemail / failed — so it must not double-notify on success.
    return None


async def _report_callme_outcome(
    context: ContextTypes.DEFAULT_TYPE, chat_id: int, to: str, room: str
) -> None:
    """Poll the bridge for the call's outcome and DM the user only if it didn't
    connect (busy / no answer / voicemail / failed). Connected calls are left to
    the bridge's own post-call summary, so we never double-notify."""
    try:
        data = await wait_for_result(room, deadline_seconds=CALLME_RESULT_DEADLINE)
    except Exception as exc:  # noqa: BLE001
        log.warning("callme result poll failed for room=%s: %s", room, exc)
        return
    message = _callme_outcome_message(to, data)
    if not message:
        return
    try:
        await context.bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
    except Exception as exc:  # noqa: BLE001
        log.warning("callme outcome notify failed for chat=%s: %s", chat_id, exc)


async def callme(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Call the user in assistant mode: the clone phones them and converses
    freely (and can summarize their email).

    With no argument, rings the number you saved by sharing your contact —
    "my agent, call me". Pass a number (/callme +14155551234) to ring a
    different phone just this once."""
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await update.message.reply_text("Run /register first.")
        return
    if not tenant.get("is_active", 1):
        await update.message.reply_text("Your account is inactive. Contact the operator.")
        return
    args = context.args or []
    digits = "".join(ch for ch in " ".join(args) if ch.isdigit())
    if digits:
        # Explicit override — ring a different phone this once.
        if len(digits) < 7:
            await update.message.reply_text(
                "That doesn't look like a phone number. Send /callme to ring "
                "your saved number, or /callme +14155551234 for a different one."
            )
            return
        to = "+" + digits
    else:
        # No argument — ring the number saved when you shared your contact.
        to = (tenant.get("phone") or "").strip()
        if not to:
            contact_kb = ReplyKeyboardMarkup(
                [[KeyboardButton("📱 Share my number", request_contact=True)]],
                resize_keyboard=True, one_time_keyboard=True,
            )
            await update.message.reply_text(
                "I don't have your number yet. Tap below to share it — then just "
                "send /callme and I'll ring you. (Or /callme +14155551234 to use "
                "a one-off number.)",
                reply_markup=contact_kb,
            )
            return
    await _place_callme(update.message, context, tenant, to)


async def _place_callme(message, context: ContextTypes.DEFAULT_TYPE, tenant: dict, to: str) -> None:
    """Dial the tenant in assistant mode and watch the outcome in the
    background. Shared by the /callme command and the natural-language
    confirm-card path."""
    await message.reply_text(f"Calling you at {to} in assistant mode…")
    out = await dial(
        to=to,
        reason="personal assistant call",
        language="en",
        tenant_id=tenant["id"],
        mode="assistant",
    )
    if out.startswith("Error"):
        await message.reply_text(out)
        return
    await message.reply_text(
        f"Call placed (room: <code>{html.escape(out)}</code>). "
        "Pick up and say e.g. “summarize my emails this week.”",
        parse_mode="HTML",
    )
    # Poll for the outcome in the background so a busy / no-answer / voicemail
    # result is reported instead of leaving the user staring at "pick up" for a
    # call that never connected. Background so other commands aren't blocked.
    _spawn_bg(_report_callme_outcome(context, message.chat_id, to, out))


async def _callme_intent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Natural-language "call me": ask for one tap before dialing. The typed
    /callme command dials immediately (explicit intent), but a classifier
    guess on free text must not phone anyone without confirmation."""
    user = update.effective_user
    msg = update.effective_message
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await msg.reply_text("Run /register first.")
        return
    if not tenant.get("is_active", 1):
        await msg.reply_text("Your account is inactive. Contact the operator.")
        return
    to = ((tenant.get("phone")) or "").strip()
    if not to:
        # No saved number yet — /callme already handles the share-contact prompt.
        await callme(update, context)
        return
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("📞 Yes, call me", callback_data="callme:yes"),
        InlineKeyboardButton("Cancel", callback_data="callme:no"),
    ]])
    await msg.reply_text(
        f"Sounds like you'd like me to call you at {to} — should I?",
        reply_markup=kb,
    )


async def callme_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirm/cancel the natural-language call-me card."""
    query = update.callback_query
    await query.answer()
    if query.data != "callme:yes":
        await query.edit_message_text("Okay — no call.")
        return
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    to = (((tenant or {}).get("phone")) or "").strip()
    if not tenant or not to:
        await query.edit_message_text("I don't have your number anymore — send /callme.")
        return
    await query.edit_message_text(f"Okay — calling you at {to}.")
    await _place_callme(query.message, context, tenant, to)


async def history(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await update.message.reply_text("Run /register first.")
        return
    try:
        data = await _bridge_json(
            "GET", f"/history/{tenant['id']}", params={"limit": MAX_HISTORY_DISPLAYED},
        )
    except _BridgeDown as e:
        log.warning("history: %s", e)
        await update.message.reply_text("Couldn't reach the bridge to load your history.")
        return
    rows = data.get("calls") or []
    if not rows:
        await update.message.reply_text("No calls yet. /call to place one.")
        return
    lines = [f"Last {len(rows)} call{'s' if len(rows) != 1 else ''}:\n"]
    for c in rows:
        started = c.get("started_at", "")[:16].replace("T", " ")
        status = c.get("status") or "pending"
        dest = c.get("destination", "")
        dur = c.get("duration_seconds") or 0.0
        answer = (c.get("answer") or "").replace("\n", " ")[:80]
        lines.append(f"• {started}  {dest}  [{status}, {dur:.0f}s]")
        if answer:
            lines.append(f"    {answer}")
    await update.message.reply_text("\n".join(lines))


async def voice_status(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """`/voice` — show the tenant's current voice clone + how to replace it."""
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await update.message.reply_text("Run /register first.")
        return
    voice_id = tenant.get("voice_id")
    voice_name = tenant.get("voice_name") or "—"
    if voice_id:
        await update.message.reply_text(
            f"Your custom voice is active.\n"
            f"  uid: {voice_id}\n"
            f"  name: {voice_name}\n\n"
            "Send a fresh voice message or audio file (≥20s of clean speech) "
            "to replace it.\n"
            "Send /clear_voice to revert to the language default."
        )
    else:
        await update.message.reply_text(
            "No custom voice set — calls use the default per-language voice.\n\n"
            "To clone yours: record a voice message (or send an audio file) of "
            "≥20s of clean speech to this chat. I'll create the clone and "
            "use it on your future calls."
        )


async def clear_voice(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """`/clear_voice` — drop the tenant's clone, revert to default."""
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await update.message.reply_text("Run /register first.")
        return
    old = tenant.get("voice_id")
    if not old:
        await update.message.reply_text("No custom voice to clear.")
        return
    await _tenants_db.set_tenant_voice(int(tenant["id"]), "", "")
    try:
        await _voices.delete_voice(old)
    except Exception:  # noqa: BLE001
        pass
    await update.message.reply_text("Cleared. Future calls use the language default.")


async def reset(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """`/reset` — wipe everything this tenant has: memories, voice clone,
    saved phone, and chat history. Confirmation-gated. Ideal between demos."""
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await update.message.reply_text("Nothing to reset — you're not registered.")
        return
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("🗑️ Yes, wipe everything", callback_data="reset:yes"),
        InlineKeyboardButton("Cancel", callback_data="reset:no"),
    ]])
    await update.message.reply_text(
        "This clears your saved facts (memory), your voice clone, your saved "
        "phone number, and this chat's history. Your account stays registered. "
        "Proceed?",
        reply_markup=kb,
    )


async def reset_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.data != "reset:yes":
        await query.edit_message_text("Cancelled — nothing was deleted.")
        return
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await query.edit_message_text("You're not registered.")
        return
    tid = int(tenant["id"])
    facts = 0
    try:
        facts = await _memory.clear_memories(tid)
    except Exception as e:  # noqa: BLE001
        log.warning("reset: clear_memories failed: %s", e)
    old_voice = tenant.get("voice_id")
    try:
        await _tenants_db.set_tenant_voice(tid, "", "")
        if old_voice:
            await _voices.delete_voice(old_voice)
    except Exception as e:  # noqa: BLE001
        log.warning("reset: voice clear failed: %s", e)
    try:
        await _tenants_db.set_tenant_phone(tid, "")
    except Exception as e:  # noqa: BLE001
        log.warning("reset: phone clear failed: %s", e)
    context.user_data.pop("chat_history", None)
    await query.edit_message_text(
        f"✅ Reset done — removed {facts} fact{'s' if facts != 1 else ''}, your "
        "voice clone, saved number, and chat history. Send a voice note to "
        "clone again."
    )


async def checkup(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """`/checkup` — ask the bridge to verify every dependency (Gradium, LLM,
    Twilio, public URL, DB) and report a green/red panel. Run before a demo."""
    await update.message.chat.send_action("typing")
    try:
        data = await _bridge_json("GET", "/readyz")
    except _BridgeDown as e:
        log.warning("checkup: %s", e)
        await update.message.reply_text("❌ Couldn't reach the bridge to run the checkup.")
        return
    checks = data.get("checks") or {}
    if not checks:
        await update.message.reply_text("Checkup returned nothing — bridge may be an old build.")
        return
    lines = ["🩺 <b>System checkup</b>"]
    for name, c in checks.items():
        ok = c.get("ok")
        icon = "✅" if ok else "❌"
        ms = c.get("ms")
        detail = f" ({ms:.0f}ms)" if isinstance(ms, (int, float)) else ""
        if not ok and c.get("error"):
            detail += f" — {html.escape(str(c['error'])[:80])}"
        lines.append(f"{icon} {html.escape(name)}{detail}")
    overall = "✅ all systems go" if data.get("ok") else "⚠️ some checks failed"
    lines.append(f"\n<b>{overall}</b>")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def translate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/translate` — arm one-shot real-time translation. Pick a target language,
    then send a voice note: I'll speak it back translated, in your cloned voice."""
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await update.message.reply_text("Run /register first.")
        return
    langs = _translate.supported_languages()
    # Two per row keeps the keyboard compact.
    rows, row = [], []
    for i, lang in enumerate(langs, 1):
        row.append(InlineKeyboardButton(lang["name"], callback_data=f"xlate:{lang['code']}"))
        if i % 2 == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    await update.message.reply_text(
        "🌍 Translate to which language?\n"
        "Pick one, then send a voice note — I'll speak it back translated"
        + (" in your cloned voice." if tenant.get("voice_id") else
           " (clone your voice first to hear it in your own voice)."),
        reply_markup=InlineKeyboardMarkup(rows),
    )


async def translate_pick_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Store the chosen target language and arm the next voice note for translation."""
    query = update.callback_query
    await query.answer()
    code = query.data.split(":", 1)[1]
    name = _translate.LANGUAGE_NAMES.get(code, code)
    context.user_data["translate_to"] = code
    await query.edit_message_text(
        f"🌍 Translating your next voice note to {name}. Send it now."
    )


async def translate_turn(update: Update, context: ContextTypes.DEFAULT_TYPE, tenant: dict) -> None:
    """Translate a single voice/audio message into the armed target language and
    reply with the translated audio in the tenant's cloned voice."""
    target = context.user_data.pop("translate_to", None)
    if not target:
        return
    msg = update.message
    file_obj = msg.voice or msg.audio
    if file_obj is None:
        return
    suffix = ".ogg"
    if msg.audio:
        mime = (msg.audio.mime_type or "").lower()
        if "mp3" in mime or "mpeg" in mime:
            suffix = ".mp3"
        elif "wav" in mime:
            suffix = ".wav"
        elif "m4a" in mime or "mp4" in mime:
            suffix = ".m4a"
    name = _translate.LANGUAGE_NAMES.get(target, target)
    await msg.chat.send_action("record_voice")
    try:
        tg_file = await file_obj.get_file()
        audio = bytes(await tg_file.download_as_bytearray())
        text, ogg_out = await _translate.translate_voice_note(
            audio, target, voice_id=tenant.get("voice_id") or None, suffix=suffix,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("translate: failed")
        await msg.reply_text(f"Translation failed: {e}")
        return
    await msg.reply_voice(voice=ogg_out)
    if text:
        await msg.reply_text(f"🌍 {name}: {text}")


async def voice_chat_turn(update: Update, context: ContextTypes.DEFAULT_TYPE, tenant: dict) -> None:
    """One round of voice-note conversation with the clone: transcribe → reply
    in the cloned voice → grow memory. The phone-free way to talk to your clone."""
    import time as _time
    msg = update.message
    await msg.chat.send_action("record_voice")
    try:
        tg_file = await msg.voice.get_file()
        ogg = bytes(await tg_file.download_as_bytearray())
        _t0 = _time.monotonic()
        user_text = await _voice_chat.transcribe(ogg)
        stt_ms = (_time.monotonic() - _t0) * 1000
    except Exception as e:  # noqa: BLE001
        log.exception("voice chat: transcription failed")
        await msg.reply_text(f"Sorry, I couldn't hear that ({e}). Try again?")
        return
    if not user_text:
        await msg.reply_text("I didn't catch any speech in that — try again?")
        return

    history = context.user_data.setdefault("chat_history", [])
    try:
        _t1 = _time.monotonic()
        answer = await _voice_chat.reply(tenant, history, user_text)
        llm_ms = (_time.monotonic() - _t1) * 1000
        _t2 = _time.monotonic()
        ogg_out = await _voice_chat.synthesize(answer, tenant["voice_id"])
        tts_ms = (_time.monotonic() - _t2) * 1000
    except Exception as e:  # noqa: BLE001
        log.exception("voice chat: reply/synthesis failed")
        await msg.reply_text(f"Hit a snag generating my reply ({e}).")
        return

    await msg.reply_voice(voice=ogg_out)
    # Latency footer — the Gradium selling point, shown on every voice turn.
    footer = f"\n\n⧗ STT {stt_ms:.0f}ms · LLM {llm_ms:.0f}ms · TTS {tts_ms:.0f}ms"
    await msg.reply_text(f"🗣️ You: {user_text}\n🤖 {answer}{footer}")
    try:
        learned = await _voice_chat.learn_from_exchange(int(tenant["id"]), user_text, answer)
        if learned:
            await msg.reply_text(f"🧠 (remembered {learned} new thing{'s' if learned != 1 else ''})")
    except Exception as e:  # noqa: BLE001
        log.warning("voice chat: memory growth failed: %s", e)


async def _route_intent(intent: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Run the existing handler for a natural-language intent. Returns True if
    the message was handled as a command (so the caller skips plain chat)."""
    if intent == "translate":
        await translate(update, context)
    elif intent == "callme":
        await _callme_intent(update, context)
    elif intent == "history":
        await history(update, context)
    elif intent == "status":
        await status(update, context)
    elif intent == "voice":
        await voice_status(update, context)
    elif intent == "clear_voice":
        await clear_voice(update, context)
    elif intent == "web":
        await web(update, context)
    elif intent == "call":
        await _start_oneshot_call(update, context)
    else:
        return False
    return True


async def _start_oneshot_call(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Natural-language outbound call: parse the number + message from the
    request and ask for one tap to confirm before dialing (a misparse must not
    silently place a real call)."""
    msg = update.effective_message
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await msg.reply_text("Run /register first.")
        return
    parsed = await _voice_chat.parse_call_request(msg.text or "")
    if not parsed.get("to"):
        await msg.reply_text(
            "Sure — what number should I call? Say it all at once, e.g. "
            "\"call +1 555 123 4567 and tell them I'm running late\", or use /call."
        )
        return
    context.user_data["pending_oscall"] = {**parsed, "tenant_id": tenant["id"]}
    text, kb = _oscall_card(context.user_data["pending_oscall"])
    await msg.reply_text(text, reply_markup=kb)


# Languages the outbound call can be placed in (match business_agent support).
_CALL_LANG_LABELS = {"en": "English", "fr": "French", "pt": "Português"}


def _oscall_card(pending: dict) -> tuple[str, InlineKeyboardMarkup]:
    """Render the one-shot call confirmation: details + a language row (current
    one ticked, tap to change) + Place/Cancel."""
    lang = pending.get("language", "en")
    task = pending.get("task") or "(no message — I'll just introduce the call)"
    lang_row = [
        InlineKeyboardButton(
            ("✅ " if code == lang else "") + label,
            callback_data=f"oscall:lang:{code}",
        )
        for code, label in _CALL_LANG_LABELS.items()
    ]
    kb = InlineKeyboardMarkup([
        lang_row,
        [InlineKeyboardButton("📞 Place call", callback_data="oscall:yes"),
         InlineKeyboardButton("Cancel", callback_data="oscall:no")],
    ])
    text = (
        f"Ready to call:\n"
        f"• To: {pending['to']}\n"
        f"• Language: {_CALL_LANG_LABELS.get(lang, lang)}  (tap below to change)\n"
        f"• Message: {task}"
    )
    return text, kb


async def oscall_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Confirm/cancel a natural-language one-shot call, then dial + report —
    mirrors the /call ConversationHandler's confirm step."""
    query = update.callback_query
    await query.answer()
    # Language switch: update the pending call and re-render the card; don't dial.
    if query.data.startswith("oscall:lang:"):
        pending = context.user_data.get("pending_oscall")
        if not pending:
            await query.edit_message_text("This call setup expired — ask again.")
            return
        code = query.data.rsplit(":", 1)[-1]
        if code in _CALL_LANG_LABELS and code != pending.get("language"):
            pending["language"] = code
            text, kb = _oscall_card(pending)
            try:
                await query.edit_message_text(text, reply_markup=kb)
            except Exception:  # noqa: BLE001 - ignore "message not modified"
                pass
        return
    pending = context.user_data.pop("pending_oscall", None)
    if query.data != "oscall:yes" or not pending:
        await query.edit_message_text("Cancelled.")
        return
    to = pending["to"]
    task = pending.get("task") or ""
    language = pending.get("language", "en")
    tenant_id = pending.get("tenant_id")
    await query.edit_message_text(f"Dialing {to}…")
    out = await dial(to=to, reason=task, language=language, tenant_id=tenant_id)
    if out.startswith("Error"):
        await query.message.reply_text(out)
        return
    room = out
    await query.message.reply_text(
        f"Call placed (room: <code>{html.escape(room)}</code>). "
        "I'll post the result here when the call ends.",
        parse_mode="HTML",
    )
    _spawn_bg(_report_call_result(query.message, room))


async def handle_text_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Plain typed message → text reply from the clone. The text counterpart to
    voice_chat_turn: same LLM + memory + shared chat_history, but replies with
    text (voice notes still get voice replies). Registered after the /call
    ConversationHandler so it doesn't hijack that flow.

    Before chatting, the message is run through a lightweight intent classifier:
    if it maps to a command (e.g. "translate this clip" → the /translate flow),
    that handler runs instead of a plain reply. Defaults to chat on any doubt."""
    user = update.effective_user
    msg = update.message
    if not msg or not msg.text:
        return
    intent = await _voice_chat.classify_intent(msg.text)
    if intent != "chat" and await _route_intent(intent, update, context):
        return
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await msg.reply_text("Send /register first, then we can chat.")
        return
    await msg.chat.send_action("typing")
    history = context.user_data.setdefault("chat_history", [])
    try:
        answer = await _voice_chat.reply(tenant, history, msg.text, channel="text")
    except Exception as e:  # noqa: BLE001
        log.exception("text chat: reply failed")
        await msg.reply_text(f"Hit a snag generating my reply ({e}).")
        return
    await msg.reply_text(answer)
    try:
        learned = await _voice_chat.learn_from_exchange(int(tenant["id"]), msg.text, answer)
        if learned:
            await msg.reply_text(f"🧠 (remembered {learned} new thing{'s' if learned != 1 else ''})")
    except Exception as e:  # noqa: BLE001
        log.warning("text chat: memory growth failed: %s", e)


async def handle_audio_sample(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Voice notes/audio. With a clone already set, a voice note is a CHAT turn;
    otherwise it's a sample to clone (re-clone via /clear_voice first)."""
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await update.message.reply_text("Send /register first, then I'll clone your voice.")
        return

    # Armed by /translate → translate this clip instead of cloning/chatting.
    if context.user_data.get("translate_to") and (update.message.voice or update.message.audio):
        await translate_turn(update, context, tenant)
        return

    # Clone exists + this is a voice note → talk to the clone.
    if update.message.voice and tenant.get("voice_id"):
        await voice_chat_turn(update, context, tenant)
        return

    msg = update.message
    file_obj = None
    suffix = ".ogg"
    if msg.voice:
        file_obj = msg.voice
        suffix = ".ogg"
    elif msg.audio:
        file_obj = msg.audio
        # Telegram audio uploads keep the original mime — common: mp3, m4a, wav, ogg.
        mime = (msg.audio.mime_type or "").lower()
        if "mp3" in mime or "mpeg" in mime:
            suffix = ".mp3"
        elif "wav" in mime:
            suffix = ".wav"
        elif "m4a" in mime or "mp4" in mime:
            suffix = ".m4a"
        elif "ogg" in mime or "opus" in mime:
            suffix = ".ogg"

    if file_obj is None:
        return  # not an audio message — let other handlers pick it up

    # Consent gate: never clone a voice note silently. The sample could be
    # anyone's voice — require an explicit "it's my own voice" confirmation.
    context.user_data["pending_clone"] = {"file_id": file_obj.file_id, "suffix": suffix}
    keyboard = [[
        InlineKeyboardButton("✅ Yes, clone my voice", callback_data="clone_consent:yes"),
        InlineKeyboardButton("❌ Cancel", callback_data="clone_consent:no"),
    ]]
    await msg.reply_text(
        "Before I clone this: please confirm this recording is YOUR OWN voice "
        "and you consent to creating a synthetic clone of it for this agent.",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def clone_consent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Runs the actual clone once the user confirms the sample is their own voice."""
    query = update.callback_query
    await query.answer()
    pending = context.user_data.pop("pending_clone", None)
    if query.data != "clone_consent:yes":
        await query.edit_message_text("Cancelled — your voice was not cloned.")
        return
    if not pending:
        await query.edit_message_text("That confirmation expired. Send the voice note again.")
        return
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await query.edit_message_text("Send /register first, then I'll clone your voice.")
        return

    msg = query.message
    suffix = pending["suffix"]
    await query.edit_message_text("Thanks — cloning your voice via Gradium…")
    try:
        tg_file = await context.bot.get_file(pending["file_id"])
        audio_bytes = await tg_file.download_as_bytearray()
    except Exception as e:  # noqa: BLE001
        await msg.reply_text(f"Couldn't fetch your audio from Telegram: {e}")
        return

    try:
        result = await _voices.clone_from_bytes(
            bytes(audio_bytes),
            name=f"gradphone:{tenant['name']}",
            suffix=suffix,
            description=f"Telegram clone for tenant_id={tenant['id']}",
        )
    except ValueError as e:
        await msg.reply_text(f"{e}")
        return
    except Exception as e:  # noqa: BLE001
        log.exception("voice clone failed")
        await msg.reply_text(f"Cloning failed: {e}")
        return

    uid = result.get("uid") or result.get("voice_id") or result.get("id")
    if not uid:
        await msg.reply_text(f"Gradium returned an unexpected response: {result}")
        return

    await _tenants_db.set_tenant_voice(
        int(tenant["id"]), uid, voice_name=f"gradphone:{tenant['name']}"
    )
    await msg.reply_text(
        f"Done. Your voice clone is active.\n"
        f"uid: <code>{html.escape(uid)}</code>\n\n"
        "🎙️ Send me a <b>voice note</b> anytime to talk to your clone — it replies "
        "in your voice and remembers what you tell it.\n"
        "/voice to inspect, /clear_voice to re-clone.",
        parse_mode="HTML",
    )
    # Before/after A/B: the same line in a stock voice, then in their clone —
    # makes the cloning quality immediately obvious.
    ab_line = "Hi! This is what I sound like. Pretty close, right?"
    try:
        default_ogg = await _voice_chat.synthesize(ab_line, _AB_DEFAULT_VOICE_ID)
        await msg.reply_voice(voice=default_ogg, caption="🔊 A generic Gradium voice")
        clone_ogg = await _voice_chat.synthesize(ab_line, uid)
        await msg.reply_voice(voice=clone_ogg, caption="🎯 Your clone — same words")
    except Exception as e:  # noqa: BLE001
        log.warning("voice A/B sample failed: %s", e)


async def status(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    tenant = await _fetch_tenant(user.id) if user else None
    if not tenant:
        await update.message.reply_text("Run /register first.")
        return
    try:
        data = await _bridge_json("GET", "/calls/live")
    except _BridgeDown as e:
        log.warning("status: %s", e)
        await update.message.reply_text("Couldn't reach the bridge to check live calls.")
        return
    mine = [c for c in (data.get("calls") or []) if c.get("tenant_id") == tenant["id"]]
    if not mine:
        await update.message.reply_text("No calls in flight.")
        return
    lines = [f"{len(mine)} call{'s' if len(mine) != 1 else ''} in flight:\n"]
    for c in mine:
        lines.append(
            f"• {c.get('destination', '?')}  phase={c.get('phase')}  "
            f"age={c.get('age_seconds')}s  room={c.get('room')}"
        )
    await update.message.reply_text("\n".join(lines))


def _allowed_telegram_ids() -> set[int]:
    """Parse ALLOWED_TELEGRAM_IDS (comma-separated Telegram user IDs)."""
    raw = os.environ.get("ALLOWED_TELEGRAM_IDS", "")
    ids: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
    return ids


async def _gatekeeper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Group -1 guard that runs before every handler.

    Single-owner model (fails CLOSED for forks):
      - ALLOWED_TELEGRAM_IDS set → only the owner's Telegram ID may use the bot.
      - Else → the bot refuses everyone, unless ALLOW_INSECURE_LOCAL=1 is
        explicitly set for local dev. This stops a freshly-forked bot from
        being open to the entire internet by default.
    """
    allowed = _allowed_telegram_ids()
    user = update.effective_user
    uid = user.id if user else None

    if allowed:
        if uid in allowed:
            return
        # Log the rejected id so "not authorized" is self-diagnosable — the
        # usual cause is messaging from a different Telegram account than the
        # one in ALLOWED_TELEGRAM_IDS.
        log.warning("gatekeeper denied uid=%s (allowed=%s)", uid, sorted(allowed))
        denied = "Not authorized. This is a personal assistant for its owner only."
    elif os.environ.get("ALLOW_INSECURE_LOCAL", "").strip().lower() in ("1", "true", "yes"):
        return
    else:
        denied = "This bot isn't configured yet (set ALLOWED_TELEGRAM_IDS to the owner's Telegram ID)."

    msg = update.effective_message
    if msg is not None:
        try:
            await msg.reply_text(denied)
        except Exception:  # noqa: BLE001 - never let the refusal crash the gate
            pass
    raise ApplicationHandlerStop


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Catch-all so one failing handler never silently drops an update.

    Logs the traceback and, when a chat is known, tells the user instead of
    leaving them with no reply. This is the backstop for the cases a handler
    doesn't anticipate — e.g. an update with no ``message`` (edited/channel
    posts), or an unexpected bridge/LLM error.
    """
    log.exception("unhandled error in handler", exc_info=context.error)
    chat = getattr(update, "effective_chat", None)
    if chat is not None:
        try:
            await context.bot.send_message(
                chat.id, "Something went wrong handling that — please try again."
            )
        except Exception:  # noqa: BLE001 - the error handler must never raise
            pass


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var is required")

    # concurrent_updates: without it PTB processes updates strictly one at a
    # time, so any handler that awaits something slow (an LLM reply, a result
    # poll) freezes the bot for every other user and command.
    app = Application.builder().token(token).concurrent_updates(True).build()
    app.add_handler(TypeHandler(_Update, _gatekeeper), group=-1)
    app.add_error_handler(_on_error)

    conv = ConversationHandler(
        entry_points=[CommandHandler("call", call_start)],
        states={
            ASK_TO: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_to)],
            ASK_TASK: [MessageHandler(filters.TEXT & ~filters.COMMAND, got_task)],
            ASK_LANG: [CallbackQueryHandler(got_language, pattern=r"^lang:")],
            CONFIRM: [CallbackQueryHandler(confirm, pattern=r"^confirm:")],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("register", register))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("web", web))
    app.add_handler(CommandHandler("callme", callme))
    app.add_handler(CommandHandler("voice", voice_status))
    app.add_handler(CommandHandler("clear_voice", clear_voice))
    app.add_handler(CommandHandler("translate", translate))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("checkup", checkup))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio_sample))
    app.add_handler(MessageHandler(filters.CONTACT, save_contact))
    app.add_handler(CallbackQueryHandler(clone_consent, pattern=r"^clone_consent:"))
    app.add_handler(CallbackQueryHandler(translate_pick_language, pattern=r"^xlate:"))
    app.add_handler(CallbackQueryHandler(oscall_confirm, pattern=r"^oscall:"))
    app.add_handler(CallbackQueryHandler(callme_confirm, pattern=r"^callme:"))
    app.add_handler(CallbackQueryHandler(reset_confirm, pattern=r"^reset:"))
    app.add_handler(conv)
    # Free-text chat — registered AFTER conv so the /call flow's text steps
    # take precedence while that conversation is active.
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_chat))

    async def _post_init(application: Application) -> None:
        # Populate the Telegram "/" command menu so features are discoverable.
        await application.bot.set_my_commands([
            BotCommand("register", "Create your account"),
            BotCommand("callme", "Have your clone call your phone"),
            BotCommand("translate", "Hear yourself in another language"),
            BotCommand("voice", "Voice-clone status"),
            BotCommand("clear_voice", "Remove your clone (then re-clone)"),
            BotCommand("history", "Your recent calls"),
            BotCommand("status", "Live call status"),
            BotCommand("checkup", "Verify all systems are green"),
            BotCommand("reset", "Wipe your data (fresh demo)"),
            BotCommand("web", "Open the web dashboard"),
            BotCommand("whoami", "Show your Telegram ID"),
        ])
    app.post_init = _post_init

    log.info("gradphone bot starting against bridge %s", _bridge_url())
    app.run_polling()


if __name__ == "__main__":
    main()
