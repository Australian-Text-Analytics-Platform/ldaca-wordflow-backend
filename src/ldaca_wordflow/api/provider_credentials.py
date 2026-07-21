"""Write-only provider credential configuration routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response, Security, status

from ..models.provider_credentials import (
    ProviderCredentialPatch,
    ProviderCredentialSummary,
)
from ..runtime import Runtime, get_runtime
from ..services.sessions import SessionPrincipal
from .responses import api_errors
from .security import get_current_session

router = APIRouter(
    prefix="/provider-credentials",
    tags=["provider-credentials"],
    responses=api_errors(401),
)


@router.get(
    "",
    response_model=ProviderCredentialSummary,
    responses=api_errors(500),
)
async def get_provider_credentials(
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    runtime: Runtime = Depends(get_runtime),
) -> ProviderCredentialSummary:
    return await runtime.provider_credential_store.summary()


@router.patch(
    "",
    response_model=ProviderCredentialSummary,
    responses=api_errors(400, 403, 409, 422, 500),
)
async def update_provider_credentials(
    patch: ProviderCredentialPatch,
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    runtime: Runtime = Depends(get_runtime),
) -> ProviderCredentialSummary:
    return await runtime.provider_credential_store.update(patch)


@router.delete(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=api_errors(400, 403, 500),
)
async def clear_provider_credentials(
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    runtime: Runtime = Depends(get_runtime),
) -> Response:
    await runtime.provider_credential_store.clear()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
