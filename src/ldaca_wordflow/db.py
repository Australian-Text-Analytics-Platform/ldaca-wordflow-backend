"""SQLite persistence for users and session data.

Used by:
- Auth service and admin routes for direct async access to auth/session records.

Flow:
- Resolve a SQLite connection target from settings.
- Open an `aiosqlite` connection per operation.
- Create the `users` and `user_sessions` tables on startup.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

import aiosqlite

from .settings import settings

logger = logging.getLogger(__name__)

_DATABASE_URL_OVERRIDE: str | None = None

_CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    hashed_password TEXT NOT NULL,
    name TEXT NOT NULL,
    picture TEXT,
    google_id TEXT UNIQUE,
    user_folder_path TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    is_superuser INTEGER NOT NULL DEFAULT 0,
    is_verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_login TEXT
);
"""

_CREATE_USER_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);
"""

_CREATE_USER_SESSIONS_USER_ID_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);"
)
_CREATE_USER_SESSIONS_EXPIRES_AT_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);"
)


def set_database_url_override(database_url: str | None) -> None:
    """Set an explicit database URL for tests or caller-managed runtime.

    Used by:
    - `backend/tests/conftest.py`, where a temporary sqlite file is used to
      isolate test writes from real workspace data.

    Why:
    - Keeps production setting resolution untouched while giving tests a stable
      override path.
    """

    global _DATABASE_URL_OVERRIDE
    _DATABASE_URL_OVERRIDE = database_url


def _resolve_database_url() -> str:
    """Return the active database URL.

    Used by:
    - every helper in this module because callers should not duplicate override
      logic.

    Why:
    - Preserves production settings and test/runtime override behavior in one
      place.
    """

    if _DATABASE_URL_OVERRIDE is not None:
        return _DATABASE_URL_OVERRIDE
    return settings.get_database_url()


def _resolve_database_target(database_url: str) -> tuple[str, bool]:
    """Convert a SQLAlchemy-style URL into `aiosqlite.connect` arguments.

    Returns:
        tuple[path_or_uri, use_uri]

    Used by:
    - `get_connection()` to centralize target parsing for memory and file DBs.

    Why:
    - Keeps connection setup concentrated, including memory URI handling for
      deterministic tests.
    """

    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if database_url.startswith(prefix):
            target = unquote(database_url.removeprefix(prefix))
            if target == ":memory:":
                return target, False
            if target.startswith(":memory:"):
                return f"file:{target}", True
            return target, False

    if database_url.startswith("file:"):
        # file: URIs are supported as-is by sqlite and aiosqlite.
        return unquote(database_url), True

    raise ValueError(f"Unsupported database URL scheme: {database_url}")


@asynccontextmanager
async def get_connection() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Yield an `aiosqlite` connection configured for the project DB.

    Used by:
    - auth/admin routes and test helpers for transaction-scoped DB reads and writes.

    Why:
    - Keeps SQLite setup (foreign-key enforcement + row factory) in one place.
    """

    database_url = _resolve_database_url()
    database_path, use_uri = _resolve_database_target(database_url)

    if not use_uri and database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    connection = await aiosqlite.connect(database_path, uri=use_uri)
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
    finally:
        await connection.close()


async def create_db_and_tables() -> None:
    """Create auth/session tables if they are not already present.

    Used by:
    - `init_db()` and test startup fixtures before token/user code runs.

    Why:
    - Keeps the application resilient on first launch and in fresh test profiles.
    """

    async with get_connection() as connection:
        for statement in (
            _CREATE_USERS_TABLE_SQL,
            _CREATE_USER_SESSIONS_TABLE_SQL,
            _CREATE_USER_SESSIONS_USER_ID_INDEX_SQL,
            _CREATE_USER_SESSIONS_EXPIRES_AT_INDEX_SQL,
        ):
            await connection.execute(statement)
        await connection.commit()


async def init_db() -> None:
    """Prepare persistent storage for startup and runtime operations."""

    data_root = settings.get_data_root()
    data_root.mkdir(parents=True, exist_ok=True)
    await create_db_and_tables()
    logger.info("Database initialized at: %s", settings.get_database_url())
