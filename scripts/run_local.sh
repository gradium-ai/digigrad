#!/usr/bin/env bash
# gradphone — run everything locally (macOS + Linux)
#
# Starts the cloudflare tunnel, syncs the (ephemeral) tunnel URL into .env,
# points your Twilio number's voice webhook at it, then launches the bridge and
# the Telegram bot. Ctrl-C stops all three.
#
# Prereq: run scripts/setup.sh once first.
# Usage:  scripts/run_local.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE=".env"
VENV=".venv"
PORT=8082
TUN_LOG=/tmp/gradphone_tunnel.log
BRIDGE_LOG=/tmp/gradphone_bridge.log
BOT_LOG=/tmp/gradphone_bot.log

[ -f "$ENV_FILE" ]          || { echo "no .env — run scripts/setup.sh first"; exit 1; }
[ -x "$VENV/bin/python" ]   || { echo "no $VENV — run scripts/setup.sh first"; exit 1; }
VPY="$VENV/bin/python"

# Make the gradphone package importable regardless of how the editable install
# landed. On macOS some tools flag .venv as hidden, and CPython's site.py
# SILENTLY SKIPS hidden .pth files — which would make `import gradphone` fail at
# launch. Putting src on PYTHONPATH sidesteps that entirely and also wins over
# any stale namespace dir in site-packages.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CF="$(command -v cloudflared || true)"
[ -n "$CF" ] || CF="./.tools/cloudflared"
[ -x "$CF" ] || { echo "cloudflared not found — run scripts/setup.sh first"; exit 1; }

get_env(){ grep -E "^$1=" "$ENV_FILE" | head -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//'; }
set_env(){ # $1 key  $2 value  (portable, via python — no sed -i quirks)
  "$VPY" - "$ENV_FILE" "$1" "$2" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1]); k, v = sys.argv[2], sys.argv[3]; t = p.read_text()
pat = rf'(?m)^{re.escape(k)}=.*$'
t = re.sub(pat, f'{k}={v}', t) if re.search(pat, t) else t + f'\n{k}={v}\n'
p.write_text(t)
PY
}

pids=()
cleanup(){
  echo; echo "stopping…"
  for p in "${pids[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  pkill -f "cloudflared tunnel --url http://localhost:$PORT" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------- tunnel ----------
echo "==> starting cloudflare tunnel"
rm -f "$TUN_LOG"
"$CF" tunnel --url "http://localhost:$PORT" >"$TUN_LOG" 2>&1 &
pids+=($!)
URL=""
for _ in $(seq 1 30); do
  URL="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUN_LOG" | head -1 || true)"
  [ -n "$URL" ] && break
  sleep 1
done
[ -n "$URL" ] || { echo "tunnel URL never appeared (see $TUN_LOG)"; exit 1; }
HOST="${URL#https://}"
echo "    tunnel: $URL"
set_env PUBLIC_HTTP_URL "https://$HOST"
set_env PUBLIC_WS_URL   "wss://$HOST"
echo "    wrote PUBLIC_HTTP_URL / PUBLIC_WS_URL to .env"

# ---------- twilio webhook ----------
SID="$(get_env TWILIO_ACCOUNT_SID)"; TOK="$(get_env TWILIO_AUTH_TOKEN)"; NUM="$(get_env TWILIO_PHONE_NUMBER)"
if [ -n "$SID" ] && [ -n "$TOK" ] && [ -n "$NUM" ] && [ "$NUM" != "+1xxxxxxxxxx" ]; then
  echo "==> pointing Twilio $NUM voice webhook at the tunnel"
  PNSID="$(curl -s -u "$SID:$TOK" \
    "https://api.twilio.com/2010-04-01/Accounts/$SID/IncomingPhoneNumbers.json?PhoneNumber=$NUM" \
    | "$VPY" -c 'import sys,json; l=json.load(sys.stdin).get("incoming_phone_numbers",[]); print(l[0]["sid"] if l else "")')"
  if [ -n "$PNSID" ]; then
    curl -s -o /dev/null -u "$SID:$TOK" \
      -d "VoiceUrl=https://$HOST/twilio/voice" -d "VoiceMethod=POST" \
      "https://api.twilio.com/2010-04-01/Accounts/$SID/IncomingPhoneNumbers/$PNSID.json"
    echo "    $NUM -> https://$HOST/twilio/voice"
  else
    echo "    !! number $NUM not found on this Twilio account — webhook NOT set"
  fi
else
  echo "==> Twilio creds/number not set in .env — skipping webhook (Telegram chat still works)"
fi

# ---------- bridge ----------
echo "==> starting bridge on :$PORT"
rm -f "$BRIDGE_LOG"
"$VENV/bin/uvicorn" gradphone.bridge:app --host 0.0.0.0 --port "$PORT" >"$BRIDGE_LOG" 2>&1 &
pids+=($!)
healthy=0
for _ in $(seq 1 24); do
  code="$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://localhost:$PORT/healthz" || true)"
  [ "$code" = "200" ] && { healthy=1; break; }
  sleep 2
done
[ "$healthy" = "1" ] && echo "    bridge healthy" || { echo "    bridge failed to start (see $BRIDGE_LOG)"; exit 1; }

# ---------- telegram bot ----------
if [ -n "$(get_env TELEGRAM_BOT_TOKEN)" ]; then
  echo "==> starting Telegram bot"
  rm -f "$BOT_LOG"
  "$VPY" -m gradphone.bot >"$BOT_LOG" 2>&1 &
  pids+=($!)
  echo "    bot started"
else
  echo "==> TELEGRAM_BOT_TOKEN not set — skipping bot"
fi

cat <<EOF

gradphone is running.
  Public URL : $URL
  Twilio voice webhook : https://$HOST/twilio/voice
  Logs       : tunnel=$TUN_LOG  bridge=$BRIDGE_LOG  bot=$BOT_LOG

Press Ctrl-C to stop everything.
EOF

wait
