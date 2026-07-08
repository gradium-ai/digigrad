# gradphone — single-container deploy (one Render Web Service).
#
# Bundles BOTH processes that make up one person's clone:
#   • the bridge  — Twilio webhooks + the Media-Stream WebSocket (binds $PORT)
#   • the bot     — Telegram onboarding / voice-note + text chat
# They share one SQLite DB + recordings on a mounted disk (see render.yaml:
# HOME=/data). ffmpeg is required by the voice-note transcode path and is NOT
# present in Render's native Python env, which is why this is a Docker deploy.
FROM python:3.12-slim

# ffmpeg: OGG/Opus <-> PCM/WAV for the Telegram voice-note clone + chat.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching. Editable install so the running code
# reads templates/ + static/ straight from the copied source (no wheel-data
# packaging to worry about). constraints.txt pins every dependency to the
# exact versions the test suite ran against — deploys don't float.
COPY pyproject.toml constraints.txt ./
COPY src ./src
RUN pip install --no-cache-dir -c constraints.txt -e .

COPY start.sh ./
RUN chmod +x ./start.sh

CMD ["./start.sh"]
