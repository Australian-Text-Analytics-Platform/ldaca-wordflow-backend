"""Provider-level annotation discovery routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Security

from ..models.annotations import (
    AnnotationModelsResource,
    AnnotationProvider,
)
from ..runtime import Runtime, get_runtime
from ..services.sessions import SessionPrincipal
from .responses import api_errors
from .security import get_current_session

router = APIRouter(
    prefix="/annotation-providers",
    tags=["annotations"],
    responses=api_errors(401),
)


@router.get(
    "/{provider}/models",
    response_model=AnnotationModelsResource,
    responses=api_errors(400, 403, 422, 502),
)
async def list_annotation_models(
    provider: AnnotationProvider,
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    runtime: Runtime = Depends(get_runtime),
) -> AnnotationModelsResource:
    """Discover models using the authenticated user's stored credential."""

    return await runtime.annotation_service.models(principal.user.id, provider)


__all__ = ["router"]
