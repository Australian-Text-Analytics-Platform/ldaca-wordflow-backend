"""Canonical workspace node resource and row-query routes."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Request,
    Response,
    Security,
    status,
)

from ...models.node_resources import (
    NodeCreateRequest,
    NodeDerivationRequest,
    NodeRowsResponse,
    NodeUpdateRequest,
)
from ...models.workspace import WorkspaceNodeInfo
from ...runtime import Runtime, get_runtime
from ...services.sessions import SessionPrincipal
from ..security import get_current_session
from ..responses import api_errors, route_path, workspace_etag

router = APIRouter(
    prefix="/workspaces/{workspace_id}/nodes",
    tags=["nodes"],
    responses=api_errors(401),
)


@router.get(
    "",
    response_model=list[WorkspaceNodeInfo],
    responses=api_errors(403, 404, 409, 422),
)
async def list_nodes(
    workspace_id: uuid.UUID,
    response: Response,
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    runtime: Runtime = Depends(get_runtime),
) -> list[WorkspaceNodeInfo]:
    """Return every Data Block in the persisted graph order."""

    nodes, revision = await runtime.node_service.list_nodes(
        principal.user.id,
        str(workspace_id),
    )
    response.headers["ETag"] = workspace_etag(revision)
    return nodes


@router.post(
    "",
    response_model=WorkspaceNodeInfo,
    status_code=status.HTTP_201_CREATED,
    responses=api_errors(400, 403, 404, 409, 413, 422, 507),
)
async def create_node(
    workspace_id: uuid.UUID,
    request: NodeCreateRequest,
    http_request: Request,
    response: Response,
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    runtime: Runtime = Depends(get_runtime),
) -> WorkspaceNodeInfo:
    """Create one file-backed source or immutable derived Data Block."""

    node, revision = await runtime.node_service.create(
        principal.user.id,
        str(workspace_id),
        request,
    )
    response.headers["Location"] = route_path(
        http_request,
        "get_node",
        workspace_id=workspace_id,
        node_id=node.id,
    )
    response.headers["ETag"] = workspace_etag(revision)
    return node


@router.post(
    "/previews",
    response_model=NodeRowsResponse,
    responses=api_errors(400, 403, 404, 413, 422),
)
async def preview_node_creation(
    workspace_id: uuid.UUID,
    request: NodeDerivationRequest,
    response: Response,
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    runtime: Runtime = Depends(get_runtime),
) -> NodeRowsResponse:
    """Preview a derived-node plan without changing workspace state."""

    rows, revision = await runtime.node_service.preview(
        principal.user.id,
        str(workspace_id),
        request,
        page=page,
        page_size=page_size,
    )
    response.headers["ETag"] = workspace_etag(revision)
    return rows


@router.get(
    "/{node_id}",
    response_model=WorkspaceNodeInfo,
    responses=api_errors(404, 422),
)
async def get_node(
    workspace_id: uuid.UUID,
    node_id: uuid.UUID,
    response: Response,
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    runtime: Runtime = Depends(get_runtime),
) -> WorkspaceNodeInfo:
    """Return one node's schema and topology metadata."""

    node, revision = await runtime.node_service.get(
        principal.user.id,
        str(workspace_id),
        str(node_id),
    )
    response.headers["ETag"] = workspace_etag(revision)
    return node


@router.patch(
    "/{node_id}",
    response_model=WorkspaceNodeInfo,
    responses=api_errors(400, 403, 404, 409, 422, 507),
)
async def update_node(
    workspace_id: uuid.UUID,
    node_id: uuid.UUID,
    request: NodeUpdateRequest,
    response: Response,
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    runtime: Runtime = Depends(get_runtime),
) -> WorkspaceNodeInfo:
    """Update one node's public metadata and return the committed resource."""

    node, revision = await runtime.node_service.update(
        principal.user.id,
        str(workspace_id),
        str(node_id),
        request,
    )
    response.headers["ETag"] = workspace_etag(revision)
    return node


@router.delete(
    "/{node_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=api_errors(400, 403, 404, 409, 422, 507),
)
async def delete_node(
    workspace_id: uuid.UUID,
    node_id: uuid.UUID,
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    runtime: Runtime = Depends(get_runtime),
) -> Response:
    """Delete one node under the workspace gate and return an empty body."""

    revision = await runtime.node_service.delete(
        principal.user.id,
        str(workspace_id),
        str(node_id),
    )
    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={"ETag": workspace_etag(revision)},
    )


@router.get(
    "/{node_id}/rows",
    response_model=NodeRowsResponse,
    responses=api_errors(400, 404, 422),
)
async def get_node_rows(
    workspace_id: uuid.UUID,
    node_id: uuid.UUID,
    response: Response,
    principal: Annotated[SessionPrincipal, Security(get_current_session)],
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    sort_by: str | None = Query(None),
    descending: bool = Query(False),
    runtime: Runtime = Depends(get_runtime),
) -> NodeRowsResponse:
    """Materialize one bounded, one-based page from a lazy node plan."""

    rows, revision = await runtime.node_service.rows(
        principal.user.id,
        str(workspace_id),
        str(node_id),
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        descending=descending,
    )
    response.headers["ETag"] = workspace_etag(revision)
    return rows
