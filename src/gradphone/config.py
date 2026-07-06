from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _req(name: str) -> str:
    """Read a required env var, failing with a clear, actionable message.

    Raises a readable RuntimeError naming the variable instead of a cryptic
    ``KeyError`` from deep in a call stack.
    """
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            "Set it in your .env (see .env.example) and restart."
        )
    return val


@dataclass(frozen=True)
class Config:
    """Twilio credentials, resolved lazily.

    Only the outbound-dial path needs these, so they are validated at first
    use (``get_cfg()``) rather than at import: the bridge boots fine for a
    Telegram-only setup, and the test suite imports modules on a fresh clone
    without a .env. Gradium/LLM keys are read from the environment directly
    by the modules that use them.
    """

    twilio_account_sid: str = field(default_factory=lambda: _req("TWILIO_ACCOUNT_SID"))
    twilio_auth_token: str = field(default_factory=lambda: os.environ.get("TWILIO_AUTH_TOKEN", ""))
    twilio_phone_number: str = field(default_factory=lambda: _req("TWILIO_PHONE_NUMBER"))


@lru_cache(maxsize=1)
def get_cfg() -> Config:
    """Validate and return the Twilio config; raises RuntimeError with the
    missing variable's name if unset. Cached after the first success."""
    return Config()
