"""Auth business logic: user provisioning, session management, token validation.

Used by:
- API auth routes, admin routes, and core auth dependencies because they need a
  single source of truth for user/session persistence operations.

Flow: open a short-lived async session, execute the relevant query/mutation,
    and return normalized dicts or None to the caller.
"""

import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from collections.abc import Mapping

from .. import db as _db
from ..settings import settings

logger = logging.getLogger(__name__)

def _parse_datetime(value: str | datetime | None) -> datetime | None:
    """Parse optional ISO timestamps stored by SQLite.

    Used by:
    - auth row serialization so DB payloads match route expectations.

    Why:
    - SQLite stores timestamps as ISO strings; API contracts use ``datetime``
      objects in-memory.
    """

    if isinstance(value, datetime) or value is None:
        return value
    return datetime.fromisoformat(value)


def _coerce_bool(value: Any) -> bool:
    """Normalize SQLite integer flags into booleans."""

    if isinstance(value, bool):
        return value
    return bool(value)


def _utc_now_naive() -> datetime:
    """Return the current UTC time as a timezone-naive datetime.

    Used by:
    - All auth service functions that compare or assign timestamps against
      the database (which stores naive UTC), and by admin routes that
      perform expiry checks.
    """
    return datetime.now(UTC).replace(tzinfo=None)


def _generate_session_tokens() -> tuple[str, str]:
    """Produce a cryptographically random access+refresh token pair.

    Called by:
    - ``create_user_session`` so token generation logic is reusable and
      testable in isolation.
    """
    return secrets.token_urlsafe(32), secrets.token_urlsafe(32)


def _user_to_dict(user: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize a ``User`` row into the dict shape expected by API callers.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.
    """
    payload: dict[str, Any] = {
        "id": str(user["id"]),
        "email": user["email"],
        "name": user["name"],
        "picture": user["picture"],
        "google_id": user["google_id"],
        "user_folder_path": user["user_folder_path"],
        "created_at": _parse_datetime(user["created_at"]),
        "last_login": _parse_datetime(user["last_login"]),
        "is_active": _coerce_bool(user["is_active"]),
        "is_superuser": _coerce_bool(user["is_superuser"]),
        "is_verified": _coerce_bool(user["is_verified"]),
    }
    return payload


async def get_or_create_user(
    email: str, name: str, picture: str, google_id: str
) -> dict[str, Any]:
    """Fetch existing user by email or create/update OAuth user record.

    Used by:
    - ``api.auth.google_auth`` because they need a backend boundary that validates inputs
      before delegating to workspace or worker state.

    Why:
    - Maintains idempotent user provisioning from Google identity payloads.
    """
    now = _utc_now_naive()
    async with _db.get_connection() as conn:
        row = await (
            await conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,),
            )
        ).fetchone()

        if row:
            user_id = str(row["id"])
            await conn.execute(
                """
                UPDATE users
                SET name = ?,
                    picture = ?,
                    google_id = ?,
                    last_login = ?
                WHERE id = ?
                """,
                (name, picture, google_id, now.isoformat(), user_id),
            )
            await conn.commit()
            row = (
                await (
                    await conn.execute(
                        "SELECT * FROM users WHERE id = ?",
                        (user_id,),
                    )
                ).fetchone()
            )
            return _user_to_dict(cast(Mapping[str, Any], row))

        user_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO users (
                id, email, hashed_password, name, picture, google_id,
                user_folder_path, is_active, is_superuser, is_verified,
                created_at, last_login
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                email,
                "oauth_user",
                name,
                picture,
                google_id,
                None,
                1,
                0,
                1,
                now.isoformat(),
                now.isoformat(),
            ),
        )
        await conn.commit()

        row = (
            await (
                await conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            ).fetchone()
        )

        return _user_to_dict(cast(Mapping[str, Any], row))


async def create_user_session(user_id: str) -> dict[str, Any]:
    """Create/replace active session token pair for a user.

    Used by:
    - ``api.auth.google_auth`` because they need a backend boundary that validates inputs
      before delegating to workspace or worker state.

    Why:
    - Enforces single active session row per user in current design.
    """
    now = _utc_now_naive()
    expires_at = now + timedelta(hours=settings.token_expire_hours)
    async with _db.get_connection() as conn:
        access_token, refresh_token = _generate_session_tokens()
        await conn.execute(
            "DELETE FROM user_sessions WHERE user_id = ?",
            (user_id,),
        )
        await conn.execute(
            """
            INSERT INTO user_sessions (user_id, access_token, refresh_token, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, access_token, refresh_token, expires_at.isoformat(), now.isoformat()),
        )
        await conn.commit()

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": settings.token_expire_hours * 3600,
            "expires_at": expires_at,
        }


async def validate_access_token(access_token: str) -> dict[str, Any] | None:
    """Validate access token and return user/session payload when active.

    Used by:
    - auth dependency validation paths because callers need the shared authentication and
      session persistence rule in one place instead of duplicating it.

    Why:
    - Centralizes token expiry and join logic for user identity resolution.
    """
    now = _utc_now_naive().isoformat()
    async with _db.get_connection() as conn:
        row = await (
            await conn.execute(
                """
                SELECT
                    users.id,
                    users.email,
                    users.name,
                    users.picture,
                    users.google_id,
                    users.user_folder_path,
                    users.created_at,
                    users.last_login,
                    users.is_active,
                    users.is_superuser,
                    users.is_verified,
                    user_sessions.access_token,
                    user_sessions.expires_at
                FROM users
                JOIN user_sessions ON users.id = user_sessions.user_id
                WHERE user_sessions.access_token = ?
                AND user_sessions.expires_at > ?
                """,
                (access_token, now),
            )
        ).fetchone()

        if row:
            payload = _user_to_dict(cast(Mapping[str, Any], row))
            payload["access_token"] = row["access_token"]
            payload["expires_at"] = _parse_datetime(row["expires_at"])
            return payload
        return None


async def get_user_by_email(email: str) -> dict[str, Any] | None:
    """Return user payload by email when present.

    Used by:
    - auth/user administration lookup paths because callers need the shared authentication
      and session persistence rule in one place instead of duplicating it.

    Why:
    - Provides a consistent dict payload shape for caller code.
    """
    async with _db.get_connection() as conn:
        row = await (
            await conn.execute("SELECT * FROM users WHERE email = ?", (email,))
        ).fetchone()
        if row:
            return _user_to_dict(cast(Mapping[str, Any], row))
        return None


async def cleanup_expired_sessions():
    """Delete expired session rows from storage.

    Used by:
    - app startup/shutdown maintenance and logout flows because callers need the shared
      authentication and session persistence rule in one place instead of duplicating it.

    Why:
    - Prevents stale sessions from accumulating indefinitely.
    """
    now = _utc_now_naive().isoformat()
    async with _db.get_connection() as conn:
        await conn.execute(
            "DELETE FROM user_sessions WHERE expires_at <= ?",
            (now,),
        )
        await conn.commit()


async def update_user_folder_path(user_id: str, folder_path: str) -> None:
    """Persist user folder location after storage provisioning.

    Used by:
    - ``api.auth.google_auth`` because they need a backend boundary that validates inputs
      before delegating to workspace or worker state.

    Why:
    - Keeps DB user metadata aligned with filesystem initialization.
    """
    async with _db.get_connection() as conn:
        cursor = await conn.execute("UPDATE users SET user_folder_path = ? WHERE id = ?", (folder_path, user_id))
        if cursor.rowcount:
            await conn.commit()
            logger.info("Updated user %s folder path to: %s", user_id, folder_path)
        else:
            logger.warning("User %s not found for folder path update", user_id)
