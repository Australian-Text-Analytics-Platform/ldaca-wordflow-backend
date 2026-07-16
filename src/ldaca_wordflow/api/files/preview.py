"""Thin HTTP adapters for safe, bounded file reads."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Security
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask

from ...models.files import FilePreviewRequest, FilePreviewResource
from ...runtime import Runtime, get_runtime
from ...services.sessions import SessionPrincipal
from ...services.user_files import UserFileStore
from ..responses import api_errors
from ..security import get_current_session
from .dependencies import get_user_file_store

router = APIRouter()
TEXT_RESPONSE_SCHEMA = {"schema": {"type": "string"}}
BINARY_RESPONSE_SCHEMA = {"schema": {"type": "string", "format": "binary"}}


@router.post(
    "/preview",
    response_model=FilePreviewResource,
    responses=api_errors(400, 403, 404, 413, 422),
)
async def preview_file(
    request: FilePreviewRequest,
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    runtime: Runtime = Depends(get_runtime),
) -> FilePreviewResource:
    """Return one format-aware page after safe path resolution."""

    return await runtime.file_read_service.preview(principal.user.id, request)


@router.get(
    "/raw",
    response_class=Response,
    responses={
        **api_errors(400, 404, 413, 422),
        200: {
            "description": "Raw UTF-8 file content.",
            "content": {
                "text/plain": TEXT_RESPONSE_SCHEMA,
                "text/markdown": TEXT_RESPONSE_SCHEMA,
            },
        },
    },
)
async def get_raw_file(
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    path: str = Query(..., description="Path relative to the user's data directory"),
    runtime: Runtime = Depends(get_runtime),
) -> Response:
    """Return validated UTF-8 content through the bounded read service."""

    content, media_type = await runtime.file_read_service.read_text(
        principal.user.id,
        path,
    )
    return Response(content=content, media_type=media_type)


@router.get(
    "/content",
    response_class=FileResponse,
    responses={
        **api_errors(400, 404, 413, 422, 507),
        200: {
            "description": "Binary file download.",
            "content": {"application/octet-stream": BINARY_RESPONSE_SCHEMA},
        },
    },
)
async def download_file(
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    path: str = Query(..., description="Path relative to the user's data directory"),
    file_store: UserFileStore = Depends(get_user_file_store),
) -> FileResponse:
    """Return a range-capable response after safe regular-file resolution."""

    snapshot = await file_store.response_snapshot(principal.user.id, path)
    return FileResponse(
        snapshot.path,
        media_type="application/octet-stream",
        filename=Path(path).name,
        background=BackgroundTask(snapshot.cleanup),
    )
