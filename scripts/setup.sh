#!/usr/bin/env bash
# gradphone — local setup bootstrap (macOS + Linux)
#
# One-time setup so you can run the agent locally. It:
#   • verifies/installs Python 3.12, ffmpeg, and cloudflared
#   • creates an isolated .venv and installs gradphone + deps (gradbot from PyPI)
#   • scaffolds .env (generates BRIDGE_API_KEY, sets local-dev defaults)
#
# Footprint, by design:
#   • Python packages go ONLY in ./.venv  (never global pip)
#   • cloudflared is downloaded into ./.tools  (no system change) unless you
#     already have it on PATH
#   • ffmpeg / Python 3.12 are installed via your OS package manager ONLY with
#     your explicit confirmation
#   • an existing .env is never overwritten
#
# After this, the only thing left to do is paste your external keys/IDs into
# .env (Telegram token, Gradium key, Twilio SID/token/number, LLM, …).
#
# Usage:  scripts/setup.sh [-y|--yes]    (-y answers "yes" to install prompts)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY_REQ="3.12"
VENV=".venv"
TOOLS_DIR=".tools"
ENV_FILE=".env"
ASSUME_YES=0

for a in "$@"; do
  case "$a" in
    -y|--yes) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "unknown arg: $a (try --help)"; exit 2 ;;
  esac
done

# ---------- pretty output ----------
if [ -t 1 ]; then
  c_reset=$'\033[0m'; c_b=$'\033[1m'; c_g=$'\033[32m'; c_y=$'\033[33m'; c_r=$'\033[31m'; c_c=$'\033[36m'
else
  c_reset=""; c_b=""; c_g=""; c_y=""; c_r=""; c_c=""
fi
step(){ printf "\n${c_b}${c_c}==>${c_reset} ${c_b}%s${c_reset}\n" "$1"; }
ok(){   printf "  ${c_g}OK${c_reset}  %s\n" "$1"; }
info(){ printf "  ${c_c}->${c_reset}  %s\n" "$1"; }
warn(){ printf "  ${c_y}!!${c_reset}  %s\n" "$1"; }
die(){  printf "  ${c_r}xx  %s${c_reset}\n" "$1" >&2; exit 1; }
confirm(){
  [ "$ASSUME_YES" = 1 ] && return 0
  [ -t 0 ] || return 1
  printf "  ${c_y}??${c_reset}  %s [y/N] " "$1"
  read -r ans </dev/tty || return 1
  case "$ans" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}
has(){ command -v "$1" >/dev/null 2>&1; }

# ---------- detect OS + package manager ----------
OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS" in
  Darwin) PLAT=macos ;;
  Linux)  PLAT=linux ;;
  *) die "unsupported OS: $OS (this script supports macOS and Linux)" ;;
esac
SUDO=""
[ "$PLAT" = linux ] && [ "$(id -u)" != 0 ] && SUDO="sudo"

PKG=""
if   has brew;    then PKG=brew
elif has apt-get; then PKG=apt
elif has dnf;     then PKG=dnf
elif has yum;     then PKG=yum
elif has pacman;  then PKG=pacman
elif has zypper;  then PKG=zypper
fi

pkg_install(){ # $@ = packages
  case "$PKG" in
    brew)   brew install "$@" ;;
    apt)    $SUDO apt-get update -y && $SUDO apt-get install -y "$@" ;;
    dnf)    $SUDO dnf install -y "$@" ;;
    yum)    $SUDO yum install -y "$@" ;;
    pacman) $SUDO pacman -Sy --noconfirm "$@" ;;
    zypper) $SUDO zypper install -y "$@" ;;
    *) return 1 ;;
  esac
}

step "Environment"
ok "platform: $PLAT ($ARCH)"
[ -n "$PKG" ] && ok "package manager: $PKG" || warn "no known package manager detected (brew/apt/dnf/yum/pacman/zypper)"

# ---------- Python 3.12 ----------
find_py(){
  if has "python${PY_REQ}"; then command -v "python${PY_REQ}"; return; fi
  if has python3 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)'; then
    command -v python3; return
  fi
  echo ""
}
step "Python ${PY_REQ}"
PYBIN="$(find_py)"
if [ -z "$PYBIN" ]; then
  warn "Python ${PY_REQ} not found (gradphone requires exactly 3.12)."
  if [ -n "$PKG" ] && confirm "Install Python ${PY_REQ} via ${PKG}?"; then
    case "$PKG" in
      brew)    pkg_install python@3.12 ;;
      apt)     pkg_install python3.12 python3.12-venv || warn "older Ubuntu may need the deadsnakes PPA for 3.12" ;;
      dnf|yum) pkg_install python3.12 ;;
      pacman)  pkg_install python ;;       # rolling; verified below
      zypper)  pkg_install python312 ;;
    esac
  fi
  PYBIN="$(find_py)"
fi
[ -n "$PYBIN" ] || die "Python ${PY_REQ} is required. Install it (https://www.python.org/downloads/release/python-3120/) and re-run."
ok "using $("$PYBIN" --version 2>&1) at $PYBIN"

# ---------- ffmpeg ----------
step "ffmpeg"
if has ffmpeg; then
  ok "ffmpeg present"
else
  warn "ffmpeg not found (needed to transcode Telegram voice notes)."
  if [ -n "$PKG" ] && confirm "Install ffmpeg via ${PKG}?"; then
    pkg_install ffmpeg || warn "ffmpeg install failed — install it manually."
  fi
  has ffmpeg && ok "ffmpeg installed" \
             || warn "ffmpeg still missing — text chat & calls work, but voice-note cloning/chat will fail until it's installed."
fi

# ---------- cloudflared (project-local, no system change) ----------
step "cloudflared (tunnel so Twilio can reach your machine)"
if has cloudflared; then
  CF="$(command -v cloudflared)"
  ok "cloudflared present at $CF (system)"
else
  mkdir -p "$TOOLS_DIR"
  CF="$TOOLS_DIR/cloudflared"
  if [ -x "$CF" ] && "$CF" --version >/dev/null 2>&1; then
    ok "cloudflared present at $CF (project-local)"
  else
    case "$ARCH" in
      x86_64|amd64)  cfarch=amd64 ;;
      aarch64|arm64) cfarch=arm64 ;;
      armv7l|armv6l) cfarch=arm ;;
      i386|i686)     cfarch=386 ;;
      *)             cfarch=amd64 ;;
    esac
    base="https://github.com/cloudflare/cloudflared/releases/latest/download"
    info "downloading cloudflared ($PLAT/$cfarch) into $TOOLS_DIR/ — nothing installed system-wide"
    if [ "$PLAT" = macos ]; then
      curl -fsSL "$base/cloudflared-darwin-${cfarch}.tgz" -o "$TOOLS_DIR/cf.tgz"
      tar -xzf "$TOOLS_DIR/cf.tgz" -C "$TOOLS_DIR"
      rm -f "$TOOLS_DIR/cf.tgz"
    else
      curl -fsSL "$base/cloudflared-linux-${cfarch}" -o "$CF"
    fi
    chmod +x "$CF"
    "$CF" --version >/dev/null 2>&1 && ok "cloudflared ready at $CF" || die "cloudflared download failed"
  fi
fi

# ---------- virtualenv + dependencies ----------
step "Virtualenv (${VENV}) + dependencies"
if [ -d "$VENV" ] && [ -x "$VENV/bin/python" ]; then
  ok "$VENV exists — reusing"
else
  "$PYBIN" -m venv "$VENV" \
    || die "venv creation failed. On Debian/Ubuntu: $SUDO apt-get install -y python3.12-venv"
  ok "created $VENV"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --upgrade pip >/dev/null
info "installing gradphone + dependencies (editable; pulls gradbot from PyPI)…"
"$VPY" -m pip install -e . >/dev/null
ok "dependencies installed"

# macOS: clear any UF_HIDDEN flag on the venv. Some macOS tools/daemons mark
# dot-dirs (.venv) hidden, and CPython's site.py SILENTLY SKIPS hidden .pth
# files — which makes the editable install's gradphone.* modules unimportable.
[ "$PLAT" = macos ] && chflags -R nohidden "$VENV" 2>/dev/null || true

# Verify the package imports the same way run_local.sh launches it (src on
# PYTHONPATH). We import the package, not gradphone.bridge — the latter loads
# config and needs the external keys you haven't added to .env yet.
if ! PYTHONPATH="$REPO_ROOT/src" "$VPY" -c "import gradphone" >/dev/null 2>&1; then
  die "gradphone is not importable after install. Debug with:
       PYTHONPATH=src $VENV/bin/python -c 'import gradphone'"
fi
ok "verified: gradphone imports cleanly"

# Non-fatal: if the editable install (the .pth) isn't honored but the check
# above passed, a background tool is flagging .venv hidden. run_local.sh sets
# PYTHONPATH=src so the app still runs; this just flags the shell-import gap.
if ! "$VPY" -c "import gradphone" >/dev/null 2>&1; then
  warn "editable .pth isn't being honored (a background tool keeps flagging"
  warn "$VENV hidden). run_local.sh handles this via PYTHONPATH and works fine."
  warn "to fix bare 'python -c \"import gradphone\"' too: chflags -R nohidden $VENV"
fi

# ---------- .env scaffold ----------
step "Configuration (${ENV_FILE})"
if [ -f "$ENV_FILE" ]; then
  ok ".env already exists — leaving your values untouched"
else
  cp .env.example "$ENV_FILE"
  ok "created .env from .env.example"
  "$VPY" - "$ENV_FILE" <<'PY'
import re, sys, secrets, pathlib
p = pathlib.Path(sys.argv[1]); t = p.read_text()
def setkv(t, k, v):
    pat = rf'(?m)^{re.escape(k)}=.*$'
    return re.sub(pat, f'{k}={v}', t) if re.search(pat, t) else t + f'\n{k}={v}\n'
t = setkv(t, 'BRIDGE_API_KEY', secrets.token_urlsafe(36))   # self-generated, not an external key
t = setkv(t, 'TWILIO_MACHINE_DETECTION', 'Disable')         # snappier local calls
t = setkv(t, 'ENABLE_INBOUND', 'true')                      # answer inbound calls
t = setkv(t, 'ALLOW_ARBITRARY_OUTBOUND', 'true')            # solo dev: allow /callme to any number
p.write_text(t)
PY
  ok "generated BRIDGE_API_KEY and set local-dev defaults"
fi

# ---------- done ----------
step "Setup complete — now add YOUR external keys to .env"
cat <<EOF

Open ${c_b}.env${c_reset} and fill these in (everything else is ready):

  ${c_b}Required${c_reset}
    GRADIUM_API_KEY         Gradium key — STT/TTS/voice cloning      (gsk_…)
    AGENT_VOICE_ID          a voice UID in your Gradium account
    LLM_BASE_URL + LLM_MODEL    OpenAI-compatible endpoint  (or set OPENAI_API_KEY)
    TELEGRAM_BOT_TOKEN      from @BotFather
    ALLOWED_TELEGRAM_IDS    your Telegram user id (from @userinfobot)
    TWILIO_ACCOUNT_SID      AC…           ─┐
    TWILIO_AUTH_TOKEN                      ├─ required for phone calls
    TWILIO_PHONE_NUMBER     +1…          ─┘

  ${c_b}Optional${c_reset}
    LINKUP_API_KEY          live web search on calls
    GMAIL_ADDRESS + GMAIL_APP_PASSWORD     "summarize my emails"

Then start everything with one command:

    ${c_b}scripts/run_local.sh${c_reset}

It starts the tunnel, writes PUBLIC_* into .env, points your Twilio number's
voice webhook at the tunnel, then launches the bridge + Telegram bot.

(Manual alternative — three terminals, from the repo root:
   ${CF} tunnel --url http://localhost:8082
   PYTHONPATH=src ${VENV}/bin/uvicorn gradphone.bridge:app --host 0.0.0.0 --port 8082
   PYTHONPATH=src ${VENV}/bin/python -m gradphone.bot )
EOF
