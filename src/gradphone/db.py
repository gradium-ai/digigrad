"""Database engine + schema for gradphone — dual-backend via SQLAlchemy.

SQLite for local dev/tests (the default, zero setup), Postgres when
``DATABASE_TYPE=postgresql`` or ``DATABASE_URL`` is set — matching gradium-serve's
``api`` service (psycopg3 + SQLAlchemy + Alembic, ``DATABASE_*`` env vars).

The SQL in tenants.py / memory.py is written portably (``ON CONFLICT``,
``LOWER() LIKE``, named ``:params``) so the *same* statements run on both
dialects through ``text()`` — no per-dialect branching. The schema lives here
as SQLAlchemy ``MetaData`` (the single source of truth for ``create_all`` AND
Alembic autogenerate).

Connection model: engines are cached per resolved URL (so tests that point
``GRADPHONE_DB`` at a fresh temp file each get an isolated engine). SQLite uses
NullPool so the engine can be reused across ``asyncio.run`` loops (the tests do
this); Postgres uses a real pool since the hosted app runs in one event loop.
"""

from __future__ import annotations

import os
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

metadata = sa.MetaData()

# Timestamps are stored as ISO-8601 TEXT and booleans as INTEGER, exactly as the
# original SQLite schema did — keeps the data format identical across backends
# and avoids any migration of existing rows.
tenants = sa.Table(
    "tenants",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("telegram_id", sa.BigInteger, unique=True),
    sa.Column("name", sa.Text),
    sa.Column("created_at", sa.Text, nullable=False),
    sa.Column("is_active", sa.Integer, server_default=sa.text("1")),
    sa.Column("custom_calls_per_day", sa.Integer),
    sa.Column("voice_id", sa.Text),
    sa.Column("voice_name", sa.Text),
    sa.Column("phone", sa.Text),
)

calls = sa.Table(
    "calls",
    metadata,
    sa.Column("room", sa.Text, primary_key=True),
    sa.Column("tenant_id", sa.Integer),
    sa.Column("twilio_call_sid", sa.Text),
    sa.Column("destination", sa.Text),
    sa.Column("task", sa.Text),
    sa.Column("language", sa.Text),
    sa.Column("business_name", sa.Text),
    sa.Column("started_at", sa.Text, nullable=False),
    sa.Column("ended_at", sa.Text),
    sa.Column("duration_seconds", sa.Float),
    sa.Column("status", sa.Text),
    sa.Column("answer", sa.Text),
    sa.Column("confidence", sa.Text),
    sa.Column("twilio_call_status", sa.Text),
    sa.Column("answered_by", sa.Text),
)

memories = sa.Table(
    "memories",
    metadata,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("tenant_id", sa.Integer, nullable=False),
    sa.Column("fact", sa.Text, nullable=False),
    sa.Column("source", sa.Text),
    sa.Column("room", sa.Text),
    sa.Column("created_at", sa.Text, nullable=False),
)

sa.Index("idx_calls_tenant_started", calls.c.tenant_id, calls.c.started_at.desc())
sa.Index("idx_memories_tenant", memories.c.tenant_id, memories.c.created_at.desc())


def _database_url() -> str:
    """Resolve the SQLAlchemy URL from env, fresh each call (so tests can
    repoint it). Precedence: DATABASE_URL > DATABASE_TYPE=postgresql > SQLite."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return url
    dbtype = os.environ.get("DATABASE_TYPE", "sqlite").strip().lower()
    if dbtype in ("postgres", "postgresql"):
        user = os.environ.get("DATABASE_USER", "")
        password = os.environ.get("DATABASE_PASSWORD", "")
        host = os.environ.get("DATABASE_HOST", "localhost")
        port = os.environ.get("DATABASE_PORT", "5432")
        name = os.environ.get("DATABASE_NAME", "gradphone")
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"
    # `or` (not get's default) so an empty GRADPHONE_DB= in .env falls back too.
    path = Path(os.environ.get("GRADPHONE_DB") or "~/.gradphone/gradphone.db").expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{path}"


def is_sqlite() -> bool:
    return _database_url().startswith("sqlite")


def sync_url() -> str:
    """Synchronous-driver URL for Alembic (which runs migrations sync).
    psycopg works in sync mode under the same ``postgresql+psycopg`` name; only
    SQLite needs its async driver suffix stripped."""
    return _database_url().replace("+aiosqlite", "")


def url_for_log() -> str:
    """Connection URL with any password redacted, for startup logging."""
    url = _database_url()
    try:
        return sa.engine.make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001
        return url


_engines: dict[str, AsyncEngine] = {}


def get_engine() -> AsyncEngine:
    url = _database_url()
    engine = _engines.get(url)
    if engine is None:
        if url.startswith("sqlite"):
            # NullPool: no connection is held across calls, so the engine is
            # safe to reuse from different event loops (the test suite runs
            # each coroutine under its own asyncio.run).
            engine = create_async_engine(
                url, poolclass=NullPool, connect_args={"timeout": 30}
            )

            # Two processes (bot + bridge) write this file concurrently. The
            # sqlite defaults (rollback journal, no busy wait) make that raise
            # "database is locked" whenever writes overlap; WAL lets readers
            # and one writer coexist, and busy_timeout makes a second writer
            # wait instead of failing.
            @sa.event.listens_for(engine.sync_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()
        else:
            engine = create_async_engine(
                url,
                pool_size=int(os.environ.get("DB_POOL_SIZE", "10")),
                max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "20")),
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        _engines[url] = engine
    return engine


async def fetch_all(sql: str, **params) -> list[dict]:
    async with get_engine().connect() as conn:
        result = await conn.execute(sa.text(sql), params)
        return [dict(row._mapping) for row in result]


async def fetch_one(sql: str, **params) -> dict | None:
    async with get_engine().connect() as conn:
        result = await conn.execute(sa.text(sql), params)
        row = result.first()
        return dict(row._mapping) if row is not None else None


async def execute(sql: str, **params) -> None:
    async with get_engine().begin() as conn:
        await conn.execute(sa.text(sql), params)


async def init_db() -> None:
    """Idempotent schema bootstrap. On SQLite (local/dev) create the tables
    directly. On Postgres, Alembic owns the schema (run by the deployment
    init-container); set ``DB_CREATE_ALL=1`` to force create_all there too."""
    force = os.environ.get("DB_CREATE_ALL", "").strip().lower() in ("1", "true", "yes")
    if is_sqlite() or force:
        async with get_engine().begin() as conn:
            await conn.run_sync(metadata.create_all)
