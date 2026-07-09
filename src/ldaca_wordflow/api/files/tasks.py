"""File-import task management endpoints.

Used by:
- FastAPI router aggregation in ``__init__.py``.

Flow:
- ``import_ldaca_dataset`` submits a background worker task.
- ``list_files_tasks`` and ``clear_files_tasks`` list and purge task records
  through the workspace manager's task manager.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header

from ...core.auth import get_current_user
from ...core.workspace import workspace_manager
from ...models.files import (
    FilesImportTaskStartResponse,
    FilesTaskActionResponse,
    FilesTasksListResponse,
    LDaCAImportRequest,
)
from .ldaca import LDACA_API_TOKEN_HEADER, _normalise_ldaca_api_token

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/import-ldaca", response_model=FilesImportTaskStartResponse)
async def import_ldaca_dataset(
    request: LDaCAImportRequest,
    current_user: dict = Depends(get_current_user),
    ldaca_api_token: Annotated[str | None, Header(alias=LDACA_API_TOKEN_HEADER)] = None,
):
    """Submit background task to import LDaCA dataset from URL.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend dataset import action.

    Why:
    - Runs network/download/import pipeline outside request-response lifecycle.
    """
    user_id = current_user["id"]
    # File imports write into the user's file tree, not a workspace graph.
    workspace_id = "global"
    tm = workspace_manager.get_task_manager(user_id)
    task_info = await tm.submit_task(
        user_id=user_id,
        workspace_id=workspace_id,
        task_type="ldaca_import",
        task_args={
            "url": request.url,
            "filename": request.filename,
            "api_token": _normalise_ldaca_api_token(ldaca_api_token),
        },
    )

    return {
        "state": "running",
        "message": "LDaCA import started",
        "metadata": {
            "task_id": task_info.id,
        },
    }


@router.get("/tasks", response_model=FilesTasksListResponse)
async def list_files_tasks(current_user: dict = Depends(get_current_user)):
    """List file-import worker tasks exposed via the files API.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend import-task status polling.

    Why:
    - Exposes file-import task status via explicit task types.
    """
    user_id = current_user["id"]
    tm = workspace_manager.get_task_manager(user_id)
    all_tasks = await tm.list(user_id=user_id)
    data = [
        task
        for task in all_tasks
        if isinstance(task, dict) and task.get("task_type") == "ldaca_import"
    ]
    return {
        "state": "successful",
        "data": data,
        "message": "Tasks retrieved successfully.",
    }


@router.post("/tasks/clear", response_model=FilesTaskActionResponse)
async def clear_files_tasks(
    task_type: str | None = None,
    task_id: str | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Clear persisted file-import task records.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the
      owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend task-list cleanup actions.

    Why:
    - Removes completed/failed import task clutter while keeping artifacts.
    """
    user_id = current_user["id"]
    tm = workspace_manager.get_task_manager(user_id)
    if task_id:
        task = await tm.get_task(task_id)
        cleared = bool(task and task.task_type == "ldaca_import")
        if cleared:
            cleared = await tm.clear_task(task_id)
        return {
            "state": "successful",
            "data": {"cleared_count": 1 if cleared else 0},
            "message": "Task cleared successfully.",
        }
    effective_task_type = task_type or "ldaca_import"
    count = await tm.clear_tasks(task_type=effective_task_type, user_id=user_id)
    return {
        "state": "successful",
        "data": {"cleared_count": count},
        "message": "All tasks cleared successfully.",
    }
