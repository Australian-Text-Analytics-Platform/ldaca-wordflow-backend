"""Join node endpoints: join two nodes, preview results.

Used by:
- Frontend and API clients through the FastAPI join routes.

Flow:
- Resolve workspace join endpoints, perform Polars joins,
- Return preview rows or persist a new joined child node.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query

from ...core.auth import get_current_user
from ...models import FilterPreviewResponse, WorkspaceNodeInfo
from .schema_filter import frontend_node_info
from ...core.exceptions import InternalServiceError
from .node_join import joined_lazyframe_for_nodes, preview_joined_nodes
from .utils import (
    _create_and_persist_child_node,
    require_workspace,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
    tags=["nodes"],
)


@router.post("/nodes/join/preview", response_model=FilterPreviewResponse)
async def join_nodes_preview(
    workspace_id: uuid.UUID,
    left_node_id: str,
    right_node_id: str,
    left_on: str | None = None,
    right_on: str | None = None,
    how: str = "inner",
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=200),
    current_user: dict = Depends(get_current_user),
):
    """Preview a join between two workspace nodes."""
    user_id = current_user["id"]
    workspace = require_workspace(user_id, str(workspace_id))
    left_node = workspace.nodes[left_node_id]
    right_node = workspace.nodes[right_node_id]
    try:
        return preview_joined_nodes(
            left_node=left_node,
            right_node=right_node,
            left_on=left_on,
            right_on=right_on,
            how=how,
            page=page,
            page_size=page_size,
        )
    except KeyError:
        raise
    except Exception as exc:
        raise InternalServiceError(str(exc)) from exc
@router.post("/nodes/join", response_model=WorkspaceNodeInfo)
async def join_nodes(
    workspace_id: uuid.UUID,
    left_node_id: str,
    right_node_id: str,
    left_on: str,
    right_on: str,
    how: str = "inner",
    new_node_name: str | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Join two workspace nodes and persist the result as a new child node."""
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    workspace = require_workspace(user_id, workspace_id_str)
    left_node = workspace.nodes[left_node_id]
    right_node = workspace.nodes[right_node_id]
    try:
        joined_data = joined_lazyframe_for_nodes(
            left_node,
            right_node,
            left_on=left_on,
            right_on=right_on,
            how=how,
        )
        node_name = new_node_name or f"{left_node.name}_join_{right_node.name}"
        new_node = _create_and_persist_child_node(
            workspace=workspace,
            data=joined_data,
            name=node_name,
            operation=f"join({left_node.name}, {right_node.name})",
            parents=[left_node, right_node],
            user_id=user_id,
            workspace_id=workspace_id_str,
        )
        return frontend_node_info(new_node)
    except KeyError:
        raise
    except Exception as exc:
        raise InternalServiceError(str(exc)) from exc
