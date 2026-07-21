"""Stateless annotation-preview HTTP boundary."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Security

from ...models.annotations import (
    AnnotationPreviewRequest,
    AnnotationPreviewResource,
)
from ...runtime import Runtime, get_runtime
from ...services.sessions import SessionPrincipal
from ..security import get_current_session
from ..responses import api_errors

router = APIRouter(
    prefix="/workspaces/{workspace_id}/nodes/{node_id}",
    tags=["annotations"],
    responses=api_errors(401),
)


@router.post(
    "/annotation-previews",
    response_model=AnnotationPreviewResource,
    responses=api_errors(400, 403, 404, 409, 422, 502),
)
async def preview_annotation(
    workspace_id: uuid.UUID,
    node_id: str,
    request: AnnotationPreviewRequest,
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    runtime: Runtime = Depends(get_runtime),
) -> AnnotationPreviewResource:
    """Classify one page without creating backend preview-session state."""

    return await runtime.annotation_service.preview(
        user_id=principal.user.id,
        workspace_id=str(workspace_id),
        node_id=node_id,
        request=request,
    )


__all__ = ["router"]
