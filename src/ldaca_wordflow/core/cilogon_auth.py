"""CILogon OIDC exchange and local session provisioning.

Used by:
- ``api.auth`` CILogon login/callback routes because the route layer should
  handle request cookies and redirects while this module handles provider I/O
  and local user/session state.

Flow:
- Cache the OIDC discovery document for the process lifetime.
- Exchange an authorization code for a provider access token.
- Fetch and validate the CILogon userinfo payload.
- Provision/update the local user, user folder path, and application session.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..settings import settings
from .auth_service import create_user_session, get_or_create_user, update_user_folder_path
from .exceptions import BadGatewayError, InvalidInputError
from .utils import setup_user_folders

logger = logging.getLogger(__name__)

_cilogon_config_cache: dict[str, Any] | None = None


async def get_cilogon_config() -> dict[str, Any]:
    """Fetch and cache the CILogon discovery document.

    Used by:
    - ``api.auth.cilogon_login`` to build the authorization URL.
    - ``api.auth.cilogon_callback`` to exchange the callback code.

    Why:
    - Discovery is provider metadata that changes rarely, so caching avoids a
      network round trip on every login/callback request.
    """
    global _cilogon_config_cache
    if _cilogon_config_cache is not None:
        return _cilogon_config_cache
    async with httpx.AsyncClient() as client:
        resp = await client.get(settings.cilogon_discovery_url, timeout=10)
        resp.raise_for_status()
        _cilogon_config_cache = resp.json()
    return _cilogon_config_cache


async def _fetch_cilogon_userinfo(
    *,
    code: str,
    redirect_uri: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Exchange one callback code and fetch its CILogon userinfo payload.

    Used by:
    - ``complete_cilogon_callback`` before local user provisioning.

    Flow: post the authorization-code grant to the provider token endpoint,
        require an access token, then call the userinfo endpoint with that
        bearer token.
    """
    async with httpx.AsyncClient() as client:
        try:
            token_resp = await client.post(
                config["token_endpoint"],
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": settings.cilogon_client_id,
                    "client_secret": settings.cilogon_client_secret,
                },
                timeout=15,
            )
            token_resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("CILogon token exchange failed: %s", exc.response.text)
            raise BadGatewayError("Token exchange with CILogon failed") from exc

        tokens = token_resp.json()
        access_token = tokens.get("access_token")
        if not access_token:
            raise BadGatewayError("No access_token in CILogon token response")

        try:
            userinfo_resp = await client.get(
                config["userinfo_endpoint"],
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10,
            )
            userinfo_resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("CILogon userinfo fetch failed: %s", exc.response.text)
            raise BadGatewayError("Fetching user info from CILogon failed") from exc

    payload = userinfo_resp.json()
    if not isinstance(payload, dict):
        raise BadGatewayError("Invalid CILogon userinfo response")
    return payload


def _cilogon_display_name(userinfo: dict[str, Any], email: str) -> str:
    """Resolve a display name from CILogon userinfo.

    Used by:
    - ``complete_cilogon_callback`` before creating/updating the local user.
    """
    return (
        userinfo.get("name")
        or (
            f"{userinfo.get('given_name', '')} {userinfo.get('family_name', '')}".strip()
        )
        or email
        or "Unknown"
    )


async def complete_cilogon_callback(
    *,
    code: str,
    redirect_uri: str,
    config: dict[str, Any],
) -> str:
    """Provision a local session for one validated CILogon callback code.

    Used by:
    - ``api.auth.cilogon_callback`` after query parameters, state cookie, and
      deployment settings have been validated.

    Flow:
    - Fetch userinfo from CILogon.
    - Require verified email metadata.
    - Create or update the local user and data folder path.
    - Create the application session and return its access token.
    """
    userinfo = await _fetch_cilogon_userinfo(
        code=code,
        redirect_uri=redirect_uri,
        config=config,
    )
    logger.info("CILogon auth successful for: %s", userinfo.get("email"))

    if not userinfo.get("email_verified", True):
        raise InvalidInputError("Email not verified by CILogon")

    email = userinfo.get("email")
    if not isinstance(email, str) or not email:
        raise BadGatewayError("CILogon user info missing email")
    subject = userinfo.get("sub")
    if not isinstance(subject, str) or not subject:
        raise BadGatewayError("CILogon user info missing subject")
    picture = userinfo.get("picture")

    user = await get_or_create_user(
        email=email,
        name=_cilogon_display_name(userinfo, email),
        picture=picture if isinstance(picture, str) else "",
        google_id=subject,
    )
    user_folders = setup_user_folders(user["id"])
    await update_user_folder_path(user["id"], str(user_folders["user_folder"]))

    session = await create_user_session(user["id"])
    return session["access_token"]
