"""Refactored workspace API endpoints - thin HTTP layer over DocWorkspace.

These endpoints are now simple HTTP wrappers around DocWorkspace methods.
All business logic is handled by the DocWorkspace library itself.

Used by:
- FastAPI workspace routers, frontend workspace features, and backend tests because they need this unit's "Refactored workspace API endpoints - thin HTTP layer over DocWorkspace" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers resolve and persist the workspace selected by the path
  `workspace_id`.
- Export and mutation endpoints materialize only at artifact or response boundaries.
- Responses return node metadata, files, exports, or HTTP errors for invalid workspace state.
"""

import importlib
import importlib.util
import logging
import os
import uuid
from typing import cast

import polars as pl
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ...core.auth import get_current_user
from ...core.node_casting import cast_lazyframe_column

# Note: DocWorkspace API helpers are not used directly in this HTTP layer
from ...models import (
    CastNodeRequest,
    CastNodeResponse,
    RenameColumnRequest,
    WorkspaceNodeInfo,
)
from .schema_filter import frontend_node_info
from .node_creation import create_workspace_node_from_file
from .node_export import export_workspace_nodes_response
from .utils import require_workspace, stage_dataframe_as_lazy, update_workspace
from ...core.exceptions import (
    InternalServiceError,
    InvalidInputError,
    NotFoundError,
    WorkspaceNotFoundError,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
    tags=["workspace"],
)

logger = logging.getLogger(__name__)
BINARY_RESPONSE_SCHEMA = {"schema": {"type": "string", "format": "binary"}}
EXPORT_NODES_RESPONSES = {
    200: {
        "description": (
            "Node export download. Single-node exports use the requested format; "
            "multi-node exports are returned as a ZIP archive."
        ),
        "content": {
            "text/csv": BINARY_RESPONSE_SCHEMA,
            "application/json": BINARY_RESPONSE_SCHEMA,
            "application/octet-stream": BINARY_RESPONSE_SCHEMA,
            "application/vnd.apache.arrow.file": BINARY_RESPONSE_SCHEMA,
            "application/x-ndjson": BINARY_RESPONSE_SCHEMA,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": BINARY_RESPONSE_SCHEMA,
            "application/zip": BINARY_RESPONSE_SCHEMA,
        },
    }
}


class WorkspaceNodeCreateRequest(BaseModel):
    """Request body for creating a workspace node from a user data file.

    Used by:
    - `add_node_to_workspace` because node creation changes workspace state and
      should carry creation inputs in a JSON body rather than query parameters.
    """

    filename: str = Field(min_length=1)
    sheet_name: str | None = Field(
        default=None,
        description="Optional Excel sheet name to load when the source file is a workbook.",
    )
    mode: str = Field(
        default="LazyFrame",
        description=(
            "How to treat the file: currently only 'LazyFrame' is supported; "
            "files are staged as parquet and reloaded lazily."
        ),
    )

@router.delete(
    "/nodes/{node_id}/columns/{column_name}", response_model=WorkspaceNodeInfo
)
async def delete_node_column(
    workspace_id: uuid.UUID,
    node_id: str,
    column_name: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a column from a node by delegating to DocWorkspace Node.drop.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI DELETE /nodes/{node_id}/columns/{column_name} route because they need this unit's "Delete a column from a node by delegating to DocWorkspace Node.drop" behavior.
    """

    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    node = ws.nodes[node_id]

    node.data = node.data.drop(column_name)
    if node.document == column_name:
        node.document = None
    update_workspace(user_id, workspace_id_str, ws, best_effort=True)
    return frontend_node_info(node)


@router.put("/nodes/{node_id}/columns/{column_name}", response_model=WorkspaceNodeInfo)
async def rename_node_column(
    workspace_id: uuid.UUID,
    node_id: str,
    column_name: str,
    payload: RenameColumnRequest,
    current_user: dict = Depends(get_current_user),
):
    """Rename a column by delegating to DocWorkspace Node.rename.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI PUT /nodes/{node_id}/columns/{column_name} route because they need this unit's "Rename a column by delegating to DocWorkspace Node.rename" behavior.
    """

    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    new_name = payload.new_name

    ws = require_workspace(user_id, workspace_id_str)
    node = ws.nodes[node_id]

    trimmed_name = new_name.strip()

    node.rename({column_name: trimmed_name})
    update_workspace(user_id, workspace_id_str, ws, best_effort=True)
    return frontend_node_info(node)


@router.post("/nodes/{node_id}/undo", response_model=WorkspaceNodeInfo)
async def undo_node_operation(
    workspace_id: uuid.UUID,
    node_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Undo the latest in-memory execution plan change for a node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /nodes/{node_id}/undo route because they need this unit's "Undo the latest in-memory execution plan change for a node" behavior.
    """

    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    node = ws.nodes[node_id]

    try:
        node.undo()
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    update_workspace(user_id, workspace_id_str, ws, best_effort=True)
    return frontend_node_info(node)


@router.post("/nodes/{node_id}/redo", response_model=WorkspaceNodeInfo)
async def redo_node_operation(
    workspace_id: uuid.UUID,
    node_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Redo the latest undone in-memory execution plan change for a node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /nodes/{node_id}/redo route because they need this unit's "Redo the latest undone in-memory execution plan change for a node" behavior.
    """

    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    node = ws.nodes[node_id]

    try:
        node.redo()
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    update_workspace(user_id, workspace_id_str, ws, best_effort=True)
    return frontend_node_info(node)


# -----------------------------------------------------------------------------
# Configure Numba threading layer with automatic TBB detection and fallback
# -----------------------------------------------------------------------------
def _configure_numba_threading():
    """Configure process-wide Numba threading defaults with safe fallbacks.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - module import side effect in `base.py` because they need this unit's "Configure process-wide Numba threading defaults with safe fallbacks" behavior.

    Why:
    - Reduces runtime instability from incompatible threading backends.

    Refactor note:
    - Duplicates intent of `api.workspaces.utils.configure_numba_threading`; these
        should converge to one shared helper.
    """
    try:
        # Presence in THREADING_LAYER_PRIORITY is only a preference list, not
        # evidence that the TBB runtime is installed/usable.
        numba_available = bool(importlib.util.find_spec("numba"))
        tbb_available = False
        tbb_import_error: Exception | None = None

        if numba_available:
            try:
                if importlib.util.find_spec("tbb"):
                    importlib.import_module("tbb")
                    tbb_available = True
                elif importlib.util.find_spec("tbb4py"):
                    importlib.import_module("tbb4py")
                    tbb_available = True
            except Exception as exc:  # pragma: no cover - environment dependent
                tbb_import_error = exc
                tbb_available = False

        if numba_available and tbb_available:
            # Use TBB if available (thread-safe for concurrent access)
            os.environ.setdefault("NUMBA_THREADING_LAYER_PRIORITY", "tbb workqueue omp")
            os.environ.setdefault("NUMBA_THREADING_LAYER", "tbb")
            # Don't set NUMBA_NUM_THREADS when using TBB - let TBB manage threading
            # Also prevent conflicts by not overriding if already set
            if "NUMBA_NUM_THREADS" not in os.environ:
                # TBB will manage its own threads
                pass
            logger.info(
                "Numba: Using TBB threading layer (thread-safe, TBB-managed threads)"
            )
        else:
            # Fall back to workqueue with single thread for safety
            os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
            os.environ.setdefault("NUMBA_THREADING_LAYER_PRIORITY", "workqueue omp tbb")
            # Only set num threads if not already set to avoid conflicts
            if "NUMBA_NUM_THREADS" not in os.environ:
                os.environ["NUMBA_NUM_THREADS"] = "1"
            if not numba_available:
                logger.info(
                    "Numba: numba not detected; using workqueue defaults for safety"
                )
            elif tbb_import_error is not None:
                logger.info(
                    "Numba: TBB detected but not importable (%s); using workqueue fallback",
                    tbb_import_error,
                )
            else:
                logger.info(
                    "Numba: TBB not installed; using workqueue threading layer (single-threaded fallback)"
                )

    except Exception as e:
        # Final fallback - basic workqueue setup
        os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
        os.environ.setdefault("NUMBA_THREADING_LAYER_PRIORITY", "workqueue omp tbb")
        os.environ.setdefault("NUMBA_NUM_THREADS", "1")
        logger.warning("Numba: Threading configuration warning: %s", e)


# Apply the configuration
_configure_numba_threading()


@router.post("/nodes", response_model=WorkspaceNodeInfo)
async def add_node_to_workspace(
    workspace_id: uuid.UUID,
    request: WorkspaceNodeCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Add a data file as a new node to workspace.

    Files are eagerly loaded into a Polars DataFrame, persisted as parquet under the workspace's `data/` folder,
    and then reloaded as a LazyFrame. This separates bulk data from `metadata.json` while keeping
    lazy processing semantics.

    Flow:
    - Resolve authentication and request body from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /nodes route because they need this unit's "Add a data file as a new node to workspace" behavior.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    return create_workspace_node_from_file(
        user_id=user_id,
        workspace_id=workspace_id_str,
        filename=request.filename,
        sheet_name=request.sheet_name,
        mode=request.mode,
    )


@router.post("/nodes/{node_id}/cast", response_model=CastNodeResponse)
async def cast_node(
    workspace_id: uuid.UUID,
    node_id: str,
    cast_data: CastNodeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Cast a single node column in place.

    Used by:
    - frontend preprocessing/data-table cast actions because the HTTP route
      owns workspace resolution and persistence while ``core.node_casting`` owns
      the Polars expression workflow.

    Flow:
    - Resolve the workspace and target node from the path ids.
    - Delegate cast validation and lazy-plan construction to
      ``cast_lazyframe_column``.
    - Persist the mutated workspace and return the existing cast response shape.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    node = ws.nodes[node_id]
    result = cast_lazyframe_column(
        node.data,
        column_name=cast_data.column,
        target_type=cast_data.target_type,
        datetime_format=cast_data.format,
        strict=cast_data.strict,
    )

    node.data = result.lazyframe
    update_workspace(user_id, workspace_id_str, ws)
    return {
        "state": "successful",
        "node_id": node_id,
        "cast_info": {
            "column": cast_data.column,
            "original_type": result.original_type,
            "new_type": result.new_type,
            "target_type": result.target_type,
            "format_used": result.format_used,
            "strict_used": result.strict_used,
        },
        "message": (
            f"Successfully cast column '{cast_data.column}' "
            f"from {result.original_type} to {result.new_type}"
            + (" (UTC timezone applied)" if result.strict_used is not None else "")
        ),
    }


@router.get(
    "/export",
    response_class=FileResponse,
    responses=EXPORT_NODES_RESPONSES,
)
async def export_nodes(
    workspace_id: uuid.UUID,
    node_ids: str,  # comma separated list
    format: str = "csv",
    current_user: dict = Depends(get_current_user),
):
    """Export one or more workspace nodes as downloadable file(s).

    If multiple node_ids are provided, a ZIP archive is returned.
    Supported formats: csv, json, parquet, ipc, ndjson.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /export route because they need this unit's "Export one or more workspace nodes as downloadable file(s)" behavior.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    return export_workspace_nodes_response(
        workspace=ws,
        workspace_id=workspace_id_str,
        node_ids=node_ids,
        export_format=format,
    )
