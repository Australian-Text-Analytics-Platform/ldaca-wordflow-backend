"""Unified authentication endpoints following Single Source of Truth principle

Used by:
- FastAPI router registration, frontend API clients, and backend tests because they need this unit's "Unified authentication endpoints following Single Source of Truth principle" behavior.

Flow:
- FastAPI mounts these endpoints under the auth API prefix.
- Route handlers resolve single-user or multi-user credentials through core auth helpers.
- Responses return the canonical identity/session payload the frontend uses at startup.
"""

import logging
import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Form, Header, Query, Request
from google.auth.transport import requests as grequests
from google.oauth2 import id_token
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from ..core.auth import (
    get_available_auth_methods,
    get_current_user,
    get_current_user_from_token,
)
from ..core.auth_service import cleanup_expired_sessions, create_user_session, get_or_create_user
from ..core.cilogon_auth import complete_cilogon_callback, get_cilogon_config
from ..models.auth import AuthInfoResponse, GoogleIn, GoogleOut, User, UserResponse
from ..models.files import MessageResponse
from ..settings import settings
from ..core.exceptions import (
    AccessDeniedError,
    AppError,
    InternalServiceError,
    InvalidInputError,
)
from ..core.utils import setup_user_folders

router = APIRouter(prefix="/auth", tags=["authentication"])
logger = logging.getLogger(__name__)


class AuthStatusUser(BaseModel):
    """Minimal authenticated-user payload for the auth status route.

    Used by:
    - ``auth_status`` because lightweight probes need identity confirmation
      without the full auth bootstrap response.
    """

    id: str
    email: str
    name: str


class AuthStatusResponse(BaseModel):
    """Response schema for the lightweight auth status route.

    Used by:
    - ``auth_status`` and generated clients because the route previously
      returned an untyped dict despite having a stable JSON shape.
    """

    authenticated: bool
    user: AuthStatusUser
    data_folder: str | None = None


class AuthHealthEndpoints(BaseModel):
    """Auth endpoint index included in ``AuthHealthResponse``."""

    auth_info: str
    google_auth: str
    user_details: str
    logout: str


class AuthHealthResponse(BaseModel):
    """Response schema for authentication subsystem health metadata.

    Used by:
    - ``auth_health`` and generated clients because unauthenticated health
      checks should still expose a typed response contract.
    """

    status: str
    mode: str
    google_configured: bool
    endpoints: AuthHealthEndpoints


@router.get("/", response_model=AuthInfoResponse)
async def get_auth_info(authorization: str | None = Header(None)):
    """
    Main auth endpoint - tells frontend everything it needs to know.

    Returns:
    - In single-user mode: authenticated=True with root user info
    - In multi-user mode with valid token: authenticated=True with user info
    - In multi-user mode without token: authenticated=False with available auth methods

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend app bootstrap auth probe because they need this unit's "Main auth endpoint - tells frontend everything it needs to know" behavior.

    Why:
    - Provides one canonical auth capability + identity payload per startup.
    """
    if not settings.multi_user:
        # Single-user mode - return root user directly
        logger.debug("Single-user mode: returning root user info")

        # Ensure root user folders and sample data are set up
        user_folders = setup_user_folders(settings.single_user_id)
        logger.debug(f"Root user folders ensured at: {user_folders['user_folder']}")

        return AuthInfoResponse(
            authenticated=True,
            user=User(
                id=settings.single_user_id,
                name=settings.single_user_name,
                email=settings.single_user_email,
                picture=None,
            ),
            available_auth_methods=[],
            requires_authentication=False,
            data_folder=str(settings.get_data_root()),
        )

    # Multi-user mode - check for existing authentication
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization.split(" ")[1]
            user = await get_current_user_from_token(token)
            logger.debug(f"Multi-user mode: authenticated user {user['email']}")

            return AuthInfoResponse(
                authenticated=True,
                user=User(
                    id=user["id"],
                    name=user["name"],
                    email=user["email"],
                    picture=user["picture"],
                ),
                available_auth_methods=get_available_auth_methods(),
                requires_authentication=True,
                data_folder=str(settings.get_data_root()),
            )
        except AppError:
            # Invalid token - fall through to unauthenticated response
            pass

    # Multi-user mode - not authenticated, return available auth methods
    logger.debug("Multi-user mode: not authenticated, returning auth methods")
    return AuthInfoResponse(
        authenticated=False,
        user=None,
        available_auth_methods=get_available_auth_methods(),
        requires_authentication=True,
        data_folder=str(settings.get_data_root()),
    )


@router.post("/google", response_model=GoogleOut)
async def google_auth(payload: GoogleIn):
    """Authenticate user via Google OAuth and create app session tokens.

    Used by the frontend Google sign-in flow (JSON body variant). Delegates
    the verification + provisioning logic to :func:`_verify_and_create_session`
    and shapes the response into ``GoogleOut``.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /google route because they need this unit's "Authenticate user via Google OAuth and create app session tokens" behavior.
    """
    result = await _verify_and_create_session(payload.id_token)
    user = result["user"]
    session = result["session"]

    return GoogleOut(
        access_token=session["access_token"],
        refresh_token=session["refresh_token"],
        expires_in=session["expires_in"],
        scope="openid email profile",
        token_type="Bearer",
        user=User(
            id=user["id"],
            email=user["email"],
            name=user["name"],
            picture=user["picture"],
        ),
    )


async def _verify_and_create_session(credential: str) -> dict:
    """Verify a Google ID token and provision a local user session.

    Shared by both the JSON API (``google_auth``) and the redirect callback
    (``google_auth_callback``).

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Verify a Google ID token and provision a local user session" behavior.
    """
    if not settings.multi_user:
        raise InvalidInputError("Google authentication not available in single-user mode",)
    if not settings.google_client_id:
        raise InternalServiceError("Google authentication not configured")
    try:
        info = id_token.verify_oauth2_token(
            credential, grequests.Request(), audience=settings.google_client_id
        )
        logger.info(f"Google auth successful for: {info.get('email')}")
    except ValueError as e:
        logger.error(f"Google token verification failed: {e}")
        raise InvalidInputError(f"Invalid ID token: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during Google token verification: {e}")
        raise InternalServiceError(f"Authentication error: {str(e)}")
    if not info.get("email_verified"):
        raise InvalidInputError("Email not verified")
    user = await get_or_create_user(
        email=info.get("email"),
        name=info.get("name"),
        picture=info.get("picture"),
        google_id=info.get("sub"),
    )
    user_folders = setup_user_folders(user["id"])
    from ..core.auth_service import update_user_folder_path

    await update_user_folder_path(user["id"], str(user_folders["user_folder"]))

    session = await create_user_session(user["id"])
    return {"user": user, "session": session}


@router.post(
    "/google/callback",
    response_class=RedirectResponse,
    status_code=303,
    responses={303: {"description": "Redirect back to the SPA with an auth token."}},
)
async def google_auth_callback(
    request: Request,
    credential: str = Form(...),
    g_csrf_token: str = Form(None),
):
    """Handle the Google Identity Services redirect callback.

    Google POSTs the ID-token ``credential`` and a CSRF token here after the
    user authenticates on Google's consent page (``ux_mode: 'redirect'``).
    We verify the token, create/find the local user, issue a session, and
    redirect back to the SPA with the access token in the URL fragment.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /google/callback route because they need this unit's "Handle the Google Identity Services redirect callback" behavior.
    """
    if g_csrf_token:
        cookie_token = request.cookies.get("g_csrf_token")
        if cookie_token != g_csrf_token:
            raise AccessDeniedError("CSRF token mismatch")
    result = await _verify_and_create_session(credential)
    token = result["session"]["access_token"]

    redirect_url = f"/?{urlencode({'auth_token': token})}"
    return RedirectResponse(url=redirect_url, status_code=303)


def _cilogon_redirect_uri(request: Request) -> str:
    """Return the registered callback URL, falling back to auto-detection.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Return the registered callback URL, falling back to auto-detection" behavior.
    """
    if settings.cilogon_redirect_uri:
        return settings.cilogon_redirect_uri
    # Auto-detect: scheme + host + /api/auth/cilogon/callback
    # Works for local dev; production deployments should set CILOGON_REDIRECT_URI.
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/auth/cilogon/callback"


# ---------------------------------------------------------------------------
# CILogon OIDC endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/cilogon/login",
    response_class=RedirectResponse,
    status_code=302,
    responses={302: {"description": "Redirect to the CILogon authorization URL."}},
)
async def cilogon_login(request: Request):
    """Redirect the browser to the CILogon authorization endpoint.

    Generates a random ``state`` for CSRF protection (stored as a short-lived
    cookie), builds the OIDC authorization URL from the discovery document,
    and redirects the user to CILogon to authenticate.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /cilogon/login route because they need this unit's "Redirect the browser to the CILogon authorization endpoint" behavior.
    """
    if not settings.multi_user:
        raise InvalidInputError("CILogon not available in single-user mode")
    if not settings.cilogon_client_id:
        raise InternalServiceError("CILogon not configured (missing CILOGON_CLIENT_ID)")

    config = await get_cilogon_config()
    state = secrets.token_urlsafe(32)
    redirect_uri = _cilogon_redirect_uri(request)

    params = urlencode(
        {
            "response_type": "code",
            "client_id": settings.cilogon_client_id,
            "redirect_uri": redirect_uri,
            "scope": "openid email profile org.cilogon.userinfo",
            "state": state,
        }
    )
    auth_url = f"{config['authorization_endpoint']}?{params}"

    response = RedirectResponse(url=auth_url, status_code=302)
    response.set_cookie(
        "cilogon_state",
        state,
        max_age=600,  # 10 minutes — enough for the user to complete login
        httponly=True,
        samesite="lax",
    )
    return response


@router.get(
    "/cilogon/callback",
    response_class=RedirectResponse,
    status_code=303,
    responses={303: {"description": "Redirect back to the SPA with an auth token."}},
)
async def cilogon_callback(
    request: Request,
    code: str | None = Query(None),
    state: str | None = Query(None),
    error: str | None = Query(None),
    error_description: str | None = Query(None),
):
    """Handle the CILogon authorization code callback.

    CILogon redirects here after the user authenticates. We verify the CSRF
    ``state``, exchange the authorization code for tokens, fetch the user's
    profile from the userinfo endpoint, provision a local session, and
    redirect back to the SPA with the access token in the URL query string.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /cilogon/callback route because they need this unit's "Handle the CILogon authorization code callback" behavior.
    """
    if error:
        detail = error_description or error
        logger.error(
            "CILogon callback error — error=%r error_description=%r all_params=%s",
            error,
            error_description,
            dict(request.query_params),
        )
        raise InvalidInputError(f"CILogon authentication failed: {detail}")

    if not code or not state:
        raise InvalidInputError("Missing code or state parameter")

    # CSRF check
    stored_state = request.cookies.get("cilogon_state")
    if not stored_state or stored_state != state:
        raise AccessDeniedError("State mismatch — possible CSRF")

    if not settings.multi_user:
        raise InvalidInputError("CILogon not available in single-user mode")
    if not settings.cilogon_client_id:
        raise InternalServiceError("CILogon not configured")

    config = await get_cilogon_config()
    redirect_uri = _cilogon_redirect_uri(request)
    token = await complete_cilogon_callback(
        code=code,
        redirect_uri=redirect_uri,
        config=config,
    )

    redirect_url = f"/?{urlencode({'auth_token': token})}"
    response = RedirectResponse(url=redirect_url, status_code=303)
    response.delete_cookie("cilogon_state")
    return response


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Return normalized current user profile fields.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend profile/session widgets because they need this unit's "Return normalized current user profile fields" behavior.

    Why:
    - Provides stable user response shape independent of DB row types.
    """
    # Convert datetime objects to ISO format strings
    created_at_str = (
        current_user["created_at"].isoformat() if current_user["created_at"] else ""
    )
    last_login_str = (
        current_user["last_login"].isoformat() if current_user["last_login"] else ""
    )

    return {
        "id": current_user["id"],  # Already a string from validate_access_token
        "email": current_user["email"],
        "name": current_user["name"],
        "picture": current_user["picture"],
        "is_active": current_user["is_active"],
        "is_verified": current_user["is_verified"],
        "created_at": created_at_str,
        "last_login": last_login_str,
    }


@router.post("/logout", response_model=MessageResponse)
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout current user session (multi-user) or no-op (single-user).

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend sign-out action because they need this unit's "Logout current user session (multi-user) or no-op (single-user)" behavior.

    Why:
    - Keeps logout behavior mode-aware while preserving shared endpoint contract.
    """
    if not settings.multi_user:
        return {"message": "Logout not applicable in single-user mode"}

    await cleanup_expired_sessions()
    logger.info(f"User {current_user['email']} logged out successfully")
    return {"message": f"User {current_user['email']} logged out successfully"}


@router.get(
    "/status",
    response_model=AuthStatusResponse,
    response_model_exclude_none=True,
)
async def auth_status(current_user: dict = Depends(get_current_user)):
    """Return minimal authenticated status payload.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - lightweight frontend auth status checks because they need this unit's "Return minimal authenticated status payload" behavior.

    Why:
    - Allows cheap auth verification without full `get_auth_info` metadata.
    """
    response: dict[str, object] = {
        "authenticated": True,
        "user": {
            "id": current_user["id"],
            "email": current_user["email"],
            "name": current_user["name"],
        },
    }

    # Add data folder path in single-user mode only
    if not settings.multi_user:
        response["data_folder"] = str(settings.get_data_root())

    return response


@router.get("/health", response_model=AuthHealthResponse)
async def auth_health():
    """Return authentication subsystem readiness metadata.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - health/status probes and diagnostics pages because they need this unit's "Return authentication subsystem readiness metadata" behavior.

    Why:
    - Exposes auth mode and endpoint availability without authentication.
    """
    return {
        "status": "healthy",
        "mode": "single-user" if not settings.multi_user else "multi-user",
        "google_configured": bool(settings.google_client_id),
        "endpoints": {
            "auth_info": "/auth/",
            "google_auth": "/auth/google",
            "user_details": "/auth/me",
            "logout": "/auth/logout",
        },
    }
