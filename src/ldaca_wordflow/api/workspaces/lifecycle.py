"""Workspace lifecycle endpoints for workspace create/load/save/import flows.

Used by:
- FastAPI workspace routers, frontend workspace features, and backend tests because they need this unit's "Workspace lifecycle endpoints for workspace create/load/save/import flows" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers validate workspace IDs, names, uploads, archive members, and current-user state.
- Helpers delegate workspace creation, loading, persistence, and task starts to the manager layer.
- Responses return workspace graphs, summaries, streamed archives, or lifecycle task handles.
"""

import io
import json
import logging
import re
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath

from docworkspace.workspace.core import Workspace
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
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
from ...core.workspace import workspace_manager
from ...models import (
    CurrentWorkspaceResponse,
    SetCurrentWorkspaceResponse,
    WorkspaceActionResponse,
    WorkspaceCreateRequest,
    WorkspaceGraphResponse,
    WorkspaceInfo,
    WorkspaceNodeReorderRequest,
    WorkspaceNodesResponse,
    WorkspaceSummary,
    WorkspaceTaskStartResponse,
    WorkspaceUploadResponse,
)
from .schema_filter import frontend_node_info
from .utils import (
    require_current_workspace,
    require_current_workspace_id,
    update_workspace,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces", tags=["lifecycle"])


def _safe_download_name(name: str) -> str:
    """Create safe download name values for workspace lifecycle routes.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Create safe download name values for workspace lifecycle routes" behavior.
    """

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "workspace"


def _safe_member_path(name: str) -> PurePosixPath:
    """Create safe member path values for workspace lifecycle routes.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Create safe member path values for workspace lifecycle routes" behavior.
    """

    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise InvalidInputError("Invalid zip entry path")
    if any(part in {"", "."} for part in path.parts):
        raise InvalidInputError("Invalid zip entry path")
    return path


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


@router.get("/current", response_model=CurrentWorkspaceResponse)
async def get_current_workspace(current_user: dict = Depends(get_current_user)):
    """Return get current workspace API requests for workspace lifecycle routes.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /current route because they need this unit's "Return get current workspace API requests for workspace lifecycle routes" behavior.
    """

    user_id = current_user["id"]
    current_workspace_id = workspace_manager.get_current_workspace_id(user_id)
    return {"id": current_workspace_id}


@router.post("/current", response_model=SetCurrentWorkspaceResponse)
async def set_current_workspace(
    workspace_id: str | None = None, current_user: dict = Depends(get_current_user)
):
    """Set or clear the current in-memory workspace for the user.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend workspace selection flow because they need this unit's "Set or clear the current in-memory workspace for the user" behavior.

    Why:
    - Ensures subsequent node/analysis operations target the intended workspace.
    """
    user_id = current_user["id"]
    previous_workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if previous_workspace_id is not None and previous_workspace_id != workspace_id:
        await workspace_manager.clear_workspace_tasks(user_id, previous_workspace_id)
    success = workspace_manager.set_current_workspace(user_id, workspace_id)
    if not success and workspace_id is not None:
        raise WorkspaceNotFoundError("Workspace not found")
    return {"state": "successful", "id": workspace_id}


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


@router.delete("/delete", response_model=WorkspaceActionResponse)
async def delete_workspace(
    workspace_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete delete workspace API requests for workspace lifecycle routes.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI DELETE /delete route because they need this unit's "Delete delete workspace API requests for workspace lifecycle routes" behavior.
    """

    user_id = current_user["id"]
    if not workspace_id.strip():
        raise InvalidInputError("workspace_id is required")
    success = workspace_manager.delete_workspace(user_id, workspace_id)
    if not success:
        raise WorkspaceNotFoundError("Workspace not found")
    return {
        "state": "successful",
        "message": f"Workspace {workspace_id} deleted successfully",
        "id": workspace_id,
    }


@router.post("/unload", response_model=WorkspaceActionResponse)
async def unload_workspace(
    save: bool = True,
    current_user: dict = Depends(get_current_user),
):
    """Handle unload workspace API requests for workspace lifecycle routes.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /unload route because they need this unit's "Handle unload workspace API requests for workspace lifecycle routes" behavior.
    """

    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if workspace_id is not None:
        await workspace_manager.clear_workspace_tasks(user_id, workspace_id)
    existed = workspace_manager.unload_workspace(user_id, workspace_id, save=save)
    if not existed:
        raise WorkspaceNotFoundError("Workspace not found")
    return {
        "state": "successful",
        "message": f"Workspace {workspace_id} unloaded",
        "id": workspace_id,
    }


@router.put("/name", response_model=WorkspaceInfo)
async def rename_workspace(
    new_name: str,
    current_user: dict = Depends(get_current_user),
):
    """Update rename workspace API requests for workspace lifecycle routes.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI PUT /name route because they need this unit's "Update rename workspace API requests for workspace lifecycle routes" behavior.
    """

    user_id = current_user["id"]
    workspace_id = require_current_workspace_id(user_id)
    workspace = require_current_workspace(user_id)
    is_valid, reason = validate_workspace_name(new_name)
    if not is_valid:
        raise InvalidInputError(f"Invalid workspace name: {reason}")
    workspace.name = new_name
    update_workspace(user_id, workspace_id, workspace)
    return workspace.info_json()


@router.put("/description", response_model=WorkspaceInfo)
async def update_workspace_description(
    description: str = "",
    current_user: dict = Depends(get_current_user),
):
    """Update update workspace description API requests for workspace lifecycle routes.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI PUT /description route because they need this unit's "Update update workspace description API requests for workspace lifecycle routes" behavior.
    """

    user_id = current_user["id"]
    workspace_id = require_current_workspace_id(user_id)
    workspace = require_current_workspace(user_id)
    workspace.description = description.strip()
    update_workspace(user_id, workspace_id, workspace)
    return workspace.info_json()


@router.post("/save", response_model=WorkspaceActionResponse)
async def save_workspace(
    current_user: dict = Depends(get_current_user),
):
    """Handle save workspace API requests for workspace lifecycle routes.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /save route because they need this unit's "Handle save workspace API requests for workspace lifecycle routes" behavior.
    """

    user_id = current_user["id"]
    workspace_id = require_current_workspace_id(user_id)
    ws = require_current_workspace(user_id)
    update_workspace(user_id, workspace_id, ws)
    return {"state": "successful", "message": "Workspace saved"}


@router.post("/download", response_model=WorkspaceTaskStartResponse)
async def start_workspace_download(
    current_user: dict = Depends(get_current_user),
):
    """Start a background task to package the workspace as a ZIP archive.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend Download button in Workspace Manager because they need this unit's "Start a background task to package the workspace as a ZIP archive" behavior.

    Why:
    - Moves potentially slow ZIP compression into the Task Center so users can
      track progress and the UI stays responsive.
    """
    user_id = current_user["id"]
    workspace_id = require_current_workspace_id(user_id)
    ws = require_current_workspace(user_id)

    # Persist latest state if this is the current in-memory workspace
    if workspace_manager.get_current_workspace_id(user_id) == workspace_id:
        update_workspace(user_id, workspace_id, ws)

    # Verify workspace directory exists before submitting
    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
    if workspace_dir is None or not workspace_dir.exists():
        raise WorkspaceNotFoundError("Workspace not found")
    # Resolve a human-readable name for the task centre label
    ws_name = ws.name if ws else workspace_id

    tm = workspace_manager.get_task_manager(user_id)
    task_info = await tm.submit_task(
        user_id=user_id,
        workspace_id=workspace_id,
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


@router.get("/download/tasks/{task_id}/artifact")
async def download_workspace_artifact(
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Stream a completed workspace ZIP artifact and delete it after download.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend auto-download on task completion because they need this unit's "Stream a completed workspace ZIP artifact and delete it after download" behavior.

    Why:
    - One-time artifact policy: the ZIP is deleted after the first successful
      download to avoid unbounded disk usage.
    """
    user_id = current_user["id"]
    workspace_id = require_current_workspace_id(user_id)

    tm = workspace_manager.get_task_manager(user_id)
    task_info = await tm.get_task(task_id)
    if task_info is None:
        raise TaskNotFoundError("Task not found")
    # Verify the task belongs to this workspace
    if task_info.workspace_id != workspace_id:
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
    filename = result.get("filename", f"{workspace_id}.zip")

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

    filename = file.filename or "workspace.zip"
    if not filename.lower().endswith(".zip"):
        raise InvalidInputError("Only .zip files are supported")
    file_bytes = await file.read()
    if not file_bytes:
        raise InvalidInputError("Uploaded file is empty")
    existing_ids = {
        item.get("id")
        for item in workspace_manager.list_user_workspaces_summaries(user_id)
        if item.get("id")
    }

    try:
        with tempfile.TemporaryDirectory(prefix="workspace_zip_") as temp_dir:
            extraction_dir = tempfile.mkdtemp(prefix="extracted_", dir=temp_dir)

            with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
                members = [m for m in zf.infolist() if not m.is_dir()]
                if not members:
                    raise InvalidInputError("ZIP archive is empty")
                safe_paths = [_safe_member_path(m.filename) for m in members]
                metadata_candidates = [
                    p
                    for p in safe_paths
                    if p.name == "metadata.json" and "__MACOSX" not in p.parts
                ]
                if not metadata_candidates:
                    raise InvalidInputError(
                        "ZIP must contain workspace metadata.json",
                    )
                metadata_path_in_zip = min(
                    metadata_candidates, key=lambda p: len(p.parts)
                )
                root_prefix = metadata_path_in_zip.parts[:-1]

                for member, safe_path in zip(members, safe_paths):
                    if "__MACOSX" in safe_path.parts:
                        continue
                    if (
                        root_prefix
                        and safe_path.parts[: len(root_prefix)] != root_prefix
                    ):
                        continue

                    relative_parts = (
                        safe_path.parts[len(root_prefix) :]
                        if root_prefix
                        else safe_path.parts
                    )
                    if not relative_parts:
                        continue

                    relative_path = PurePosixPath(*relative_parts)
                    if relative_path.name in {".DS_Store"}:
                        continue

                    destination = Path(extraction_dir) / Path(*relative_path.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member, "r") as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)

            extracted_root = Path(extraction_dir)
            metadata_file = extracted_root / "metadata.json"
            if not metadata_file.exists():
                raise InvalidInputError(
                    "ZIP missing required metadata.json at workspace root",
                )
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            workspace_metadata = metadata.setdefault("workspace_metadata", {})
            incoming_id = workspace_metadata.get("id")
            incoming_name = workspace_metadata.get("name")

            if (
                isinstance(incoming_id, str)
                and incoming_id
                and incoming_id not in existing_ids
            ):
                workspace_id = incoming_id
            else:
                workspace_id = str(uuid.uuid4())

            workspace_name = (
                incoming_name
                if isinstance(incoming_name, str) and incoming_name.strip()
                else filename.rsplit(".zip", 1)[0]
            )

            workspace_metadata["id"] = workspace_id
            workspace_metadata["name"] = workspace_name
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(metadata, f)

            target_dir = workspace_manager._resolve_workspace_dir(
                user_id=user_id,
                workspace_id=workspace_id,
                workspace_name=workspace_name,
            )
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            shutil.copytree(extracted_root, target_dir)

            workspace_manager._refresh_user_workspace_paths(user_id)

        summary = next(
            (
                item
                for item in workspace_manager.list_user_workspaces_summaries(user_id)
                if item.get("id") == workspace_id
            ),
            {
                "id": workspace_id,
                "name": workspace_name,
            },
        )
        return {"state": "successful", "workspace": summary}
    except zipfile.BadZipFile as exc:
        raise InvalidInputError(f"Invalid ZIP file: {exc}")


@router.get("/info", response_model=WorkspaceInfo)
async def get_workspace_info(
    current_user: dict = Depends(get_current_user),
):
    """Return get workspace info API requests for workspace lifecycle routes.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /info route because they need this unit's "Return get workspace info API requests for workspace lifecycle routes" behavior.
    """

    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    return workspace.info_json()


@router.get("/graph", response_model=WorkspaceGraphResponse)
async def get_workspace_graph(
    current_user: dict = Depends(get_current_user),
):
    """Return workspace graph payload.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend graph canvas initialization and refresh because they need this unit's "Return workspace graph payload" behavior.
    """
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    graph = workspace.graph_json()
    graph["nodes"] = [
        frontend_node_info(workspace.nodes[entry["id"]])
        for entry in graph.get("nodes", [])
        if entry.get("id") in workspace.nodes
    ]
    return graph


@router.put("/nodes/order", response_model=WorkspaceGraphResponse)
async def reorder_workspace_nodes(
    request: WorkspaceNodeReorderRequest,
    current_user: dict = Depends(get_current_user),
):
    """Persist a new node order for the current workspace and return the graph.

    Flow:
    - Resolve the authenticated user's active workspace id and workspace object.
    - Apply ``ordered_ids`` through ``Workspace.reorder_nodes`` (unknown ids are
      ignored and any omitted node keeps its tail position, so a stale client
      payload can never drop nodes).
    - Persist the workspace and return the rebuilt graph in the new order.

    Used by:
    - Frontend workspace list-view drag-to-reorder gesture because dropping a row
      commits the full node sequence as the durable source of truth.
    """
    user_id = current_user["id"]
    workspace_id = require_current_workspace_id(user_id)
    workspace = require_current_workspace(user_id)
    workspace.reorder_nodes(request.ordered_ids)
    update_workspace(user_id, workspace_id, workspace)
    graph = workspace.graph_json()
    graph["nodes"] = [
        frontend_node_info(workspace.nodes[entry["id"]])
        for entry in graph.get("nodes", [])
        if entry.get("id") in workspace.nodes
    ]
    return graph


@router.get("/nodes", response_model=WorkspaceNodesResponse)
async def get_workspace_nodes(
    current_user: dict = Depends(get_current_user),
):
    """Return get workspace nodes API requests for workspace lifecycle routes.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /nodes route because they need this unit's "Return get workspace nodes API requests for workspace lifecycle routes" behavior.
    """

    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    nodes = [frontend_node_info(node) for node in workspace.nodes.values()]
    return {"nodes": nodes}
