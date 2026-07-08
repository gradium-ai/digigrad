"""Test env isolation: a fresh clone (and CI) must collect and pass with no
.env and no real credentials.

- Dummy env vars satisfy fail-closed checks (session signing key length).
- GRADPHONE_DB points at a per-session temp file so tests never touch a
  developer's real ~/.gradphone database.
- Vars that switch optional features on (Places, conference mode) are set to
  "" so tool lists match the defaults the smoke tests pin. Empty (not popped):
  the app modules call load_dotenv() at import, which fills MISSING vars from
  a developer's local .env but never overrides existing ones — so "" reliably
  wins both locally and in CI.
"""

import os
import tempfile

_TEST_DB = os.path.join(tempfile.mkdtemp(prefix="gradphone-tests-"), "test.db")

os.environ.setdefault("BRIDGE_API_KEY", "test-suite-key-0123456789abcdef")
os.environ["GRADPHONE_DB"] = _TEST_DB
for _flag in (
    "GOOGLE_PLACES_API_KEY",
    "GRADIUM_CONFERENCE_MODE",
    "ALLOW_ARBITRARY_OUTBOUND",
    "OUTBOUND_ALLOWLIST",
    "ALLOW_INSECURE_LOCAL",
):
    os.environ[_flag] = ""
