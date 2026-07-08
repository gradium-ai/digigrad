# gradphone — demo runbook

The one-page guide for showing gradphone live. Read the **pre-room checklist**,
run the **5-minute script**, and keep the **recovery moves** handy.

Production URL: `https://gradphone.onrender.com` · Dashboard: `/ui`
Telegram bot: **@Gradphonebot** · Phone number: **+1 (336) 502-8926**

---

## Pre-room checklist (2 minutes)

1. **Run `/checkup` in Telegram.** You should get all ✅ — Gradium TTS, LLM,
   Twilio, public URL, database. If anything is ❌, fix it before you walk in
   (see recovery moves). This single command is your "all systems green".
2. **Confirm your voice clone is fresh.** Send `/voice`; if it says no clone or
   a call fails on it, `/clear_voice` then send a new 20–30s voice note.
3. **Paid Twilio + verified number.** `/callme` only reaches numbers Twilio can
   dial; on a trial account it must be a *verified* number.
4. **Dashboard up on the projector.** Open `/ui`, log in with `BRIDGE_API_KEY`
   (from Render → the service → Environment). Leave it on the "In flight" view.
5. **Phone on the table, ringer on.**

---

## The 5-minute script

1. **Onboard (0:00).** Telegram → `/register` → tap **Share my number**.
   *"That's the whole signup."*
2. **Clone (0:30).** Record a ~20s voice note → tap consent → clone active in
   seconds. It immediately plays a **generic voice vs. your clone** A/B of the
   same sentence. *"One Gradium API call, from 20 seconds of audio — and note
   the explicit consent gate."*
3. **Talk to yourself (1:15).** Send a voice-note question → it replies **in
   your voice**, with a latency footer (`⧗ STT · LLM · TTS`). Point at the STT
   transcript accuracy and the millisecond timings.
4. **The killer (2:15).** `/translate` → Spanish (or German) → send a voice
   note → you hear **yourself speaking a language you don't know**.
   *"Speech-to-speech, no LLM in this loop — a Gradium model end to end."*
5. **On the phone (3:15).** `/callme` → your clone rings you on speaker.
   **Interrupt it mid-sentence** (barge-in) and ask **"what do you remember
   about me?"** (memory carried from the chat minutes ago).
6. **Close (4:30).** On the dashboard, open the completed call row → show the
   **latency breakdown** (STT/LLM/Tool/TTS per turn) and play the recording.
   Then `/reset` so the next prospect starts clean.

---

## Conference mode — the Gradium voice agent

When `GRADIUM_CONFERENCE_MODE=true` (set on Render), **anyone who calls
+1 (336) 502-8926** — except you (the registered owner, who reaches your own
assistant) — is greeted by:

> "Hi, I'm Gradium's voice agent. What would you like to know about Gradium?"

It answers from a built-in knowledge base scraped from gradium.ai + the blog
(models, benchmarks, pricing, team, customers), searching the docs for
specifics. Hand attendees the number on a card and let them ask it anything
about Gradium — in a Gradium voice.

- Refresh the knowledge base anytime: `python3 scripts/scrape_gradium.py`,
  commit `src/gradphone/data/gradium_kb.json`, redeploy.
- To turn it off (back to the plain receptionist for strangers), set
  `GRADIUM_CONFERENCE_MODE=false`.

---

## Recovery moves

| Symptom | Fix |
|---|---|
| `/checkup` shows **LLM ❌ payment/HTTP** | LLM account out of credit — switch `LLM_BASE_URL`/`LLM_MODEL`/`OPENAI_API_KEY` on Render to a funded provider, redeploy. |
| Call **drops instantly** | Dead voice clone — the bridge now falls back to a default voice and DMs you; re-clone with `/clear_voice` + voice note. |
| Caller hears **"application error"** | Twilio webhook not pointing at the live URL — set the number's Voice webhook to `https://gradphone.onrender.com/twilio/voice` (POST). |
| **"Not authorized"** in Telegram | You're messaging from a different account than `ALLOWED_TELEGRAM_IDS`. Check the bridge log for `gatekeeper denied uid=…` and add that id. |
| Call **goes silent after a question** | Post-tool watchdog speaks a recovery line after ~5s; if it persists, the LLM is stalling — a stronger `LLM_MODEL` fixes it. |
| Everything looks wrong | Redeploy the service on Render (Manual Deploy → Clear build cache), then re-run `/checkup`. |
