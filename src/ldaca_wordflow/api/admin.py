"""Admin endpoints

Used by:
- FastAPI router registration, frontend API clients, and backend tests because they need this unit's "Admin endpoints" behavior.

Flow:
- FastAPI mounts these endpoints under the admin API prefix.
- Route handlers authorize the current user before reading database state.
- Responses expose operational user/session state or an admin-only HTTP error.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .config import AdminConfigResponse, AdminConfigUpdate, build_admin_config_response
from .. import db as _db
from ..core.auth import get_current_user
from ..core.auth_service import _utc_now_naive, cleanup_expired_sessions
from ..core.exceptions import AccessDeniedError
from ..settings import get_settings, reload_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse nullable ISO-8601 timestamp values from SQLite."""

    if value is None:
        return None
    return datetime.fromisoformat(value)


def _require_admin(current_user: dict) -> None:
    """Authorize admin routes.

    - Single-user mode: always allowed.
    - Multi-user mode: requires current user email to be in `ADMIN_EMAILS`.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Authorize admin routes" behavior.
    """
    current_settings = get_settings()
    if not current_settings.multi_user:
        return

    current_email = str(current_user.get("email") or "").strip().lower()
    admin_allowlist = current_settings.get_admin_emails()

    if current_email and current_email in admin_allowlist:
        return

    raise AccessDeniedError("Admin access required")


class AdminUserResponse(BaseModel):
    """Admin-visible user summary with active session count.

    Used by:
    - ``list_users`` because the admin user table returns DB-backed user rows
      plus a computed active-session count.
    """

    id: str
    email: str
    name: str | None
    created_at: datetime | None
    last_login: datetime | None
    active_sessions: int


class AdminUsersResponse(BaseModel):
    """Response schema for the admin user list route."""

    users: list[AdminUserResponse]
    total: int
    requested_by: str


class AdminCleanupResponse(BaseModel):
    """Response schema for the admin session-cleanup route."""

    message: str
    performed_by: str


@router.get("/users", response_model=AdminUsersResponse)
async def list_users(current_user: dict = Depends(get_current_user)):
    """List users with active-session counts.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - admin dashboard/user management views because they need this unit's "List users with active-session counts" behavior.

    Why:
    - Provides operational visibility into user and session activity.

    Refactor note:
    - Add `require_admin` dependency before wider deployment to avoid role drift.
    """
    _require_admin(current_user)
    logger.info("Admin user list requested by %s", current_user["email"])

    async with _db.get_connection() as conn:
        users = (
            await (
                await conn.execute(
                """
                SELECT id, email, name, created_at, last_login
                FROM users
                ORDER BY created_at ASC
                """
                )
            ).fetchall()
        )

        user_list = []
        for user in users:
            # Count active sessions for each user
            count_result = await conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM user_sessions
                WHERE user_id = ? AND expires_at > ?
                """,
                (user["id"], _utc_now_naive().isoformat()),
            )
            count_row = await count_result.fetchone()
            active_sessions = int(count_row["count"]) if count_row else 0

            user_list.append(
                {
                    "id": str(user["id"]),
                    "email": user["email"],
                    "name": user["name"],
                    "created_at": _parse_datetime(user["created_at"]),
                    "last_login": _parse_datetime(user["last_login"]),
                    "active_sessions": active_sessions,
                }
            )

        return {
            "users": user_list,
            "total": len(user_list),
            "requested_by": current_user["email"],
        }


@router.get("/cleanup", response_model=AdminCleanupResponse)
async def admin_cleanup(current_user: dict = Depends(get_current_user)):
    """Trigger cleanup of expired session rows.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - admin maintenance actions because they need this unit's "Trigger cleanup of expired session rows" behavior.

    Why:
    - Allows manual session-store maintenance in addition to automatic cleanup.

    Refactor note:
    - Add `require_admin` dependency before wider deployment.
    """
    _require_admin(current_user)
    logger.info("Session cleanup triggered by %s", current_user["email"])
    await cleanup_expired_sessions()
    return {
        "message": "Expired sessions cleaned up successfully",
        "performed_by": current_user["email"],
    }


@router.patch("/config", response_model=AdminConfigResponse)
async def update_admin_config(
    config: AdminConfigUpdate, current_user: dict = Depends(get_current_user)
):
    """Update process-local runtime configuration.

    Used by:
    - frontend working-directory settings because changing the data root affects
      file/workspace state and must stay behind the admin boundary.

    Flow:
    - Authenticate and authorize the caller through the shared admin allowlist.
    - Store the requested data root in the process environment.
    - Reload the settings singleton so existing import sites observe the new
      process-local value, then return the effective admin config snapshot.
    """
    _require_admin(current_user)

    new_path = Path(config.data_root)
    logger.info("Admin config update by %s: data_root=%s", current_user["email"], new_path)
    os.environ["DATA_ROOT"] = str(new_path)
    reload_settings()
    return build_admin_config_response()
