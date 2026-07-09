"""Read-only runtime bootstrap configuration routes.

Used by:
- FastAPI router registration, frontend API clients, and backend tests because
  they need public bootstrap metadata without exposing mutable settings.

Flow:
- FastAPI mounts this router under the API prefix.
- Route handlers read current settings through the shared settings accessor.
- Responses return generated runtime-config models for auth and OAuth bootstrap.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..settings import get_settings

router = APIRouter(tags=["configuration"])


class RuntimeConfigResponse(BaseModel):
    """Public response schema for frontend runtime bootstrap.

    Used by:
    - `get_runtime_config`, frontend auth bootstrap, and generated API clients
      because callers need auth-mode and OAuth-provider metadata before login.
    """

    multi_user_mode: bool
    google_client_id: str = ""


class AdminConfigResponse(RuntimeConfigResponse):
    """Admin-only runtime configuration response.

    Used by:
    - admin config mutation routes and generated API clients because changing
      process-local configuration should return the newly effective storage root.
    """

    data_root: str


class AdminConfigUpdate(BaseModel):
    """Request schema for admin-only process-local runtime config updates.

    Used by:
    - admin config mutation routes because the frontend settings form only needs
      to change the data root.
    """

    data_root: str = Field(min_length=1)


def build_runtime_config_response() -> RuntimeConfigResponse:
    """Build the public runtime-config payload from current settings.

    Why:
    - Keeps the read-only route and admin response construction aligned without
      exposing admin-only fields on the public endpoint.

    Called by:
    - `get_runtime_config` and admin config helpers because both need one source
      for auth-mode and OAuth-provider metadata.
    """
    current_settings = get_settings()
    return RuntimeConfigResponse(
        multi_user_mode=current_settings.multi_user,
        google_client_id=current_settings.google_client_id or "",
    )


def build_admin_config_response() -> AdminConfigResponse:
    """Build the admin runtime-config payload from current settings.

    Called by:
    - admin config mutation routes because callers need confirmation of the
      process-local data root after settings reload.
    """
    current_settings = get_settings()
    return AdminConfigResponse(
        data_root=str(current_settings.get_data_root()),
        multi_user_mode=current_settings.multi_user,
        google_client_id=current_settings.google_client_id or "",
    )


@router.get("/runtime-config", response_model=RuntimeConfigResponse)
async def get_runtime_config():
    """Return public frontend bootstrap configuration.

    Used by:
    - frontend auth bootstrap and generated API clients because the app needs
      auth mode and Google client metadata before user authentication.

    Flow:
    - Read the current process settings.
    - Return only public runtime fields; mutable and filesystem settings stay
      behind admin routes.
    """
    return build_runtime_config_response()
