"""Workspace lifecycle endpoints for workspace create/load/save/import flows.

Used by:
- FastAPI workspace routers, frontend workspace features, and backend tests because they need this unit's "Workspace lifecycle endpoints for workspace create/load/save/import flows" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers validate workspace IDs, names, uploads, archive members, and current-user state.
- Helpers delegate workspace creation, loading, persistence, and task starts to the manager layer.
- Responses return workspace graphs, summaries, streamed archives, or lifecycle task handles.
"""

import json
import logging
import re
import uuid
from pathlib import Path

from docworkspace.workspace.core import Workspace
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse

from ...core.auth import get_current_user
from ...core.exceptions import (
    AccessDeniedError,
    InvalidInputError,
    ResourceConflictError,
    ResourceGoneError,
    TaskNotFoundError,
    WorkspaceNotFoundError,
)
from ...core.utils import validate_workspace_name
from ...core.workspace_archive_import import import_workspace_zip
from ...core.workspace import workspace_manager
from ...models import (
    WorkspaceActionResponse,
    WorkspaceCreateRequest,
    WorkspaceGraphResponse,
    WorkspaceInfo,
    WorkspaceNodeReorderRequest,
    WorkspaceSummary,
    WorkspaceTaskStartResponse,
    WorkspaceUpdateRequest,
    WorkspaceUploadResponse,
)
from .utils import (
    require_current_workspace,
    require_workspace,
    update_workspace,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces", tags=["lifecycle"])
WORKSPACE_ARTIFACT_RESPONSES = {
    200: {
        "description": "Workspace ZIP artifact download.",
        "content": {
            "application/zip": {"schema": {"type": "string", "format": "binary"}}
        },
    }
}


def _safe_download_name(name: str) -> str:
    """Create safe download name values for workspace lifecycle routes.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Create safe download name values for workspace lifecycle routes" behavior.
    """

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "workspace"


def _workspace_name_from_dir(workspace_dir: Path, fallback: str) -> str:
    """Read a persisted workspace display name without loading the workspace.

    Used by:
    - ``start_workspace_download`` because inactive workspace downloads should
      package the requested workspace directory without switching the user's
      active workspace just to name the background task.

    Flow: inspect ``metadata.json`` when it exists, return a non-empty
        ``workspace_metadata.name`` value, and fall back to the path workspace id
        when metadata is missing or malformed.
    """
    metadata_file = workspace_dir / "metadata.json"
    try:
        with metadata_file.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return fallback

    workspace_name = metadata.get("workspace_metadata", {}).get("name")
    if isinstance(workspace_name, str) and workspace_name.strip():
        return workspace_name
    return fallback


def workspace_graph_payload(workspace: Workspace) -> dict[str, object]:
    """Build the lightweight graph/topology response for the current workspace.

    Used by:
    - ``get_workspace_graph`` and ``reorder_workspace_nodes`` because graph
      refreshes should return only topology plus display/action fields. Full
      node metadata, including schema and columns, is served by the collection
      node-info route.

    Flow: walk nodes in workspace order, copy graph-facing node attributes
        without calling ``Node.info()``, then derive edges from child links.
    """

    def linked_node_id(linked_node: object) -> str:
        """Return the id for DocWorkspace links stored as nodes or ids."""

        if isinstance(linked_node, str):
            return linked_node
        return str(getattr(linked_node, "id"))

    nodes_payload: list[dict[str, object | None]] = []
    edges_payload: list[dict[str, str]] = []

    for node in workspace.nodes.values():
        node_id = node.id
        nodes_payload.append(
            {
                "id": node_id,
                "name": node.name,
                "operation": node.operation,
                "parent_ids": [linked_node_id(parent) for parent in node.parents],
                "child_ids": [linked_node_id(child) for child in node.children],
                "document": node.document,
                "color": node.color,
                "can_undo": node.can_undo,
                "can_redo": node.can_redo,
            }
        )
        for child in node.children:
            edges_payload.append({"source": node_id, "target": linked_node_id(child)})

    return {"nodes": nodes_payload, "edges": edges_payload}


@router.get("/", response_model=list[WorkspaceSummary])
async def list_workspaces(current_user: dict = Depends(get_current_user)):
    """List all persisted workspaces visible to the current user.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend workspace switcher/landing views because they need this unit's "List all persisted workspaces visible to the current user" behavior.

    Why:
    - Provides fast summary metadata without loading full workspace graphs.
    """
    user_id = current_user["id"]
    summaries = workspace_manager.list_user_workspaces_summaries(user_id)
    return summaries


@router.post("/", response_model=WorkspaceInfo)
async def create_workspace(
    request: WorkspaceCreateRequest, current_user: dict = Depends(get_current_user)
):
    """Create a workspace and return normalized workspace metadata.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend new-workspace dialog because they need this unit's "Create a workspace and return normalized workspace metadata" behavior.

    Why:
    - Centralizes workspace-name validation and initialization metadata.
    """
    user_id = current_user["id"]
    is_valid, reason = validate_workspace_name(request.name)
    if not is_valid:
        raise InvalidInputError(f"Invalid workspace name: {reason}")
    workspace = Workspace(name=request.name)
    workspace_id = workspace.id
    workspace.description = request.description or ""

    update_workspace(user_id, workspace_id, workspace)
    workspace_manager.set_current_workspace(user_id, workspace_id)

    workspace_info = workspace.info_json()
    workspace_info["id"] = workspace_id
    return WorkspaceInfo(**workspace_info)


@router.post("/{workspace_id:uuid}/unload", response_model=WorkspaceActionResponse)
async def unload_workspace(
    workspace_id: uuid.UUID,
    save: bool = True,
    current_user: dict = Depends(get_current_user),
):
    """Unload the explicit workspace id in the path.

    Used by:
    - workspace manager clients that want to unload the active workspace while
      keeping the target visible in the URL.

    Flow: resolve the authenticated user, pass the path workspace id to the
        manager unload operation, and return the standard action response or
        a not-found error when that workspace is not currently loaded.
    """

    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    existed = workspace_manager.unload_workspace(user_id, workspace_id_str, save=save)
    if not existed:
        raise WorkspaceNotFoundError("Workspace not found")
    return {
        "state": "successful",
        "message": f"Workspace {workspace_id_str} unloaded",
        "id": workspace_id_str,
    }


@router.post("/{workspace_id:uuid}/download", response_model=WorkspaceTaskStartResponse)
async def start_workspace_download(
    workspace_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
):
    """Start a background task to package the explicit workspace as a ZIP.

    Used by:
    - frontend Download button in Workspace Manager because each row can start
      a download for its own workspace, independent of active selection.

    Why:
    - Moves potentially slow ZIP compression into the Task Center so users can
      track progress and the UI stays responsive.

    Flow: resolve the target workspace directory from the path id, persist the
        latest in-memory state only when that workspace is already active,
        submit the packaging task with explicit workspace metadata, and return
        the task id to the caller.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)

    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id_str)
    if workspace_dir is None or not workspace_dir.exists():
        raise WorkspaceNotFoundError("Workspace not found")

    current_workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if current_workspace_id == workspace_id_str:
        ws = require_current_workspace(user_id)
        update_workspace(user_id, workspace_id_str, ws)
        ws_name = ws.name
    else:
        ws_name = _workspace_name_from_dir(workspace_dir, workspace_id_str)

    tm = workspace_manager.get_task_manager(user_id)
    task_info = await tm.submit_task(
        user_id=user_id,
        workspace_id=workspace_id_str,
        task_type="workspace_download",
        task_args={
            "target_workspace_dir": str(workspace_dir),
        },
        task_name=f"Download: {ws_name}",
    )

    return {
        "state": "running",
        "message": "Workspace download started",
        "metadata": {
            "task_id": task_info.id,
        },
    }


@router.get(
    "/{workspace_id:uuid}/download/tasks/{task_id}/artifact",
    response_class=StreamingResponse,
    responses=WORKSPACE_ARTIFACT_RESPONSES,
)
async def download_workspace_artifact(
    workspace_id: uuid.UUID,
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Stream a completed workspace ZIP artifact and delete it after download.

    Used by:
    - frontend auto-download on task completion because the artifact task must
      still be checked against the workspace row that started the download.

    Why:
    - One-time artifact policy: the ZIP is deleted after the first successful
      download to avoid unbounded disk usage.

    Flow: resolve the path workspace id, fetch the task, reject task/workspace
        mismatches, stream a successful ZIP artifact, and remove it afterwards.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)

    tm = workspace_manager.get_task_manager(user_id)
    task_info = await tm.get_task(task_id)
    if task_info is None:
        raise TaskNotFoundError("Task not found")
    # Verify the task belongs to this workspace
    if task_info.workspace_id != workspace_id_str:
        raise AccessDeniedError("Task does not belong to this workspace")
    if task_info.task_type != "workspace_download":
        raise InvalidInputError("Task is not a workspace download")
    from ...core.worker_task_manager import TaskStatus

    if task_info.status != TaskStatus.SUCCESSFUL:
        raise ResourceConflictError(
            f"Task is not completed (state: {task_info.status.value})",
        )
    result = task_info.result
    if not isinstance(result, dict) or not result.get("artifact_path"):
        raise ResourceGoneError("Artifact metadata missing")
    artifact_path = Path(result["artifact_path"])
    if not artifact_path.exists():
        raise ResourceGoneError("Artifact already downloaded or deleted")
    filename = result.get("filename", f"{workspace_id_str}.zip")

    def _stream_and_delete():
        """Yield ZIP content then delete the artifact file.

        Steps:
        - Normalize caller input into the representation this module expects.
        - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
        - Return the compact value the caller uses for artifacts, validation, or response shaping.

        Called by:
        - The `download_workspace_artifact` local workflow in this module because they need this unit's "Yield ZIP content then delete the artifact file" behavior.
        """
        try:
            with open(artifact_path, "rb") as fh:
                while True:
                    chunk = fh.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk
        finally:
            try:
                artifact_path.unlink(missing_ok=True)
            except OSError:
                pass

    return StreamingResponse(
        _stream_and_delete(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/upload", response_model=WorkspaceUploadResponse)
async def upload_workspace_zip(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a workspace ZIP archive and import it into user workspace storage.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /upload route because they need this unit's "Upload a workspace ZIP archive and import it into user workspace storage" behavior.
    """
    user_id = current_user["id"]

    summary = import_workspace_zip(
        user_id=user_id,
        filename=file.filename or "workspace.zip",
        file_bytes=await file.read(),
    )
    return {"state": "successful", "workspace": summary}


@router.get("/{workspace_id:uuid}", response_model=WorkspaceInfo)
async def get_workspace_by_id(
    workspace_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
):
    """Return workspace metadata for the explicit workspace id in the path.

    Used by:
    - frontend workspace-detail reads because callers already know the selected
      workspace id and should not depend on hidden current-workspace state.

    Flow: resolve the authenticated user, load the requested workspace id, and
        serialize the standard ``WorkspaceInfo`` payload for that explicit
        workspace.
    """
    user_id = current_user["id"]
    workspace = require_workspace(user_id, str(workspace_id))
    return workspace.info_json()


@router.patch("/{workspace_id:uuid}", response_model=WorkspaceInfo)
async def update_workspace_by_id(
    workspace_id: uuid.UUID,
    request: WorkspaceUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Update metadata for the explicit workspace id in the path.

    Used by:
    - frontend workspace manager and header rename flows because workspace
      mutations should name their target resource directly.

    Flow: load the requested workspace, validate any submitted name, apply only
        supplied metadata fields, persist through the shared workspace saver,
        and return refreshed workspace metadata.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    workspace = require_workspace(user_id, workspace_id_str)

    if request.name is not None:
        is_valid, reason = validate_workspace_name(request.name)
        if not is_valid:
            raise InvalidInputError(f"Invalid workspace name: {reason}")
        workspace.name = request.name

    if request.description is not None:
        workspace.description = request.description.strip()

    update_workspace(user_id, workspace_id_str, workspace)
    return workspace.info_json()


@router.delete("/{workspace_id:uuid}", response_model=WorkspaceActionResponse)
async def delete_workspace_by_id(
    workspace_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
):
    """Delete the explicit workspace id in the path.

    Used by:
    - frontend workspace manager delete actions and test cleanup because deletion
      should not depend on the backend-selected current workspace.

    Flow: let the UUID path converter reject static route names, delegate
        deletion to the manager, and return the
        same action response shape as the legacy query-parameter route.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    success = workspace_manager.delete_workspace(user_id, workspace_id_str)
    if not success:
        raise WorkspaceNotFoundError("Workspace not found")
    return {
        "state": "successful",
        "message": f"Workspace {workspace_id_str} deleted successfully",
        "id": workspace_id_str,
    }


@router.post("/{workspace_id:uuid}/save", response_model=WorkspaceActionResponse)
async def save_workspace_by_id(
    workspace_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
):
    """Persist the explicit workspace id in the path.

    Used by:
    - frontend workspace manager save actions because the target workspace is
      already known by selection state.

    Flow: load the requested workspace id, persist it through the shared saver,
        and return the standard workspace action response.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    workspace = require_workspace(user_id, workspace_id_str)
    update_workspace(user_id, workspace_id_str, workspace)
    return {"state": "successful", "message": "Workspace saved", "id": workspace_id_str}


@router.get("/{workspace_id:uuid}/graph", response_model=WorkspaceGraphResponse)
async def get_workspace_graph_by_id(
    workspace_id: uuid.UUID,
    current_user: dict = Depends(get_current_user),
):
    """Return graph topology for the explicit workspace id in the path.

    Used by:
    - frontend graph and sidebar queries because cache keys are already scoped by
      workspace id and should fetch the same id over the API boundary.

    Flow: load the requested workspace and serialize its lightweight graph
        topology without duplicating full node metadata.
    """
    user_id = current_user["id"]
    workspace = require_workspace(user_id, str(workspace_id))
    return workspace_graph_payload(workspace)


@router.put("/{workspace_id:uuid}/nodes/order", response_model=WorkspaceGraphResponse)
async def reorder_workspace_nodes_by_id(
    workspace_id: uuid.UUID,
    request: WorkspaceNodeReorderRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist node order for the explicit workspace id in the path.

    Used by:
    - frontend list-view drag-to-reorder because graph order mutations are
      workspace-scoped and should not rely on hidden current-workspace state.

    Flow: load the requested workspace, apply the submitted order, persist, and
        return the rebuilt graph in the new order.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    workspace = require_workspace(user_id, workspace_id_str)
    workspace.reorder_nodes(request.ordered_ids)
    update_workspace(user_id, workspace_id_str, workspace)
    return workspace_graph_payload(workspace)
