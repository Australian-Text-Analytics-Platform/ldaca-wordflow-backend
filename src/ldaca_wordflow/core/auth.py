"""Authentication utilities and dependencies

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: open the configured database/session boundary, normalize user or token records,
    and return dependency-ready authentication state.
"""

import logging

from fastapi import Header

from ..core.auth_service import validate_access_token
from ..core.exceptions import UnauthenticatedError
from ..settings import settings

logger = logging.getLogger(__name__)


async def get_current_user(authorization: str | None = Header(None)):
    """Dependency to get current authenticated user.

    Single-user mode: returns configured local user.
    Multi-user mode: validates bearer token.

    Used by:
    - protected API route dependencies across backend routers because they need a backend
      boundary that validates inputs before delegating to workspace or worker state.
    Why:
    - Centralizes auth-mode branching so route handlers stay auth-agnostic.

    Flow: open the configured database/session boundary, normalize user or token records,
        and return dependency-ready authentication state.
    """
    if not settings.multi_user:
        # Single-user mode - always return root user
        logger.debug("Single-user mode: returning root user")
        return {
            "id": settings.single_user_id,
            "email": settings.single_user_email,
            "name": settings.single_user_name,
            "picture": None,
            "is_active": True,
            "is_verified": True,
            "created_at": None,
            "last_login": None,
        }

    # Multi-user mode - require authentication
    if not authorization:
        raise UnauthenticatedError("Authorization header required")

    # Accept raw token or "Bearer <token>"
    token = authorization[7:] if authorization.startswith("Bearer ") else authorization
    user = await validate_access_token(token)
    if not user:
        raise UnauthenticatedError("Invalid or expired token")
    return user


async def get_current_user_from_token(token: str) -> dict:
    """Validate bearer token and return user payload.

    Used by:
    - auth/session bootstrap routes because they need a backend boundary that validates
      inputs before delegating to workspace or worker state.
    Why:
    - Provides direct token validation when header dependency injection is not used.

    Flow: open the configured database/session boundary, normalize user or token records,
        and return dependency-ready authentication state.
    """
    user = await validate_access_token(token)
    if not user:
        raise UnauthenticatedError("Invalid or expired token")
    return user


def get_available_auth_methods() -> list:
    """Return enabled authentication providers for the current config.

    Used by:
    - auth capability/status endpoints because they need a backend boundary that validates
      inputs before delegating to workspace or worker state.
    Why:
    - Lets frontend discover login options dynamically.

    Flow: open the configured database/session boundary, normalize user or token records,
        and return dependency-ready authentication state.
    """
    methods = []

    if settings.multi_user and settings.google_client_id:
        methods.append({"name": "google", "display_name": "Google", "enabled": True})

    if settings.multi_user and settings.cilogon_client_id:
        methods.append({"name": "cilogon", "display_name": "CILogon", "enabled": True})

    return methods
