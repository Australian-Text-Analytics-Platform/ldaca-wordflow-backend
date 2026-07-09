"""Authenticated user-scoped API endpoints.

Used by:
- FastAPI router registration and generated frontend clients because UI session
  state such as the selected workspace belongs to the authenticated user.

Flow:
- Resolve the current user through the shared auth dependency.
- Delegate workspace selection persistence to the workspace manager.
- Return typed response bodies consumed by generated clients.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..core.auth import get_current_user
from ..core.exceptions import WorkspaceNotFoundError
from ..core.workspace import workspace_manager
from ..models import (
    CurrentWorkspaceResponse,
    CurrentWorkspaceUpdateRequest,
    SetCurrentWorkspaceResponse,
)

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me/current-workspace", response_model=CurrentWorkspaceResponse)
async def get_my_current_workspace(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's selected workspace id.

    Used by:
    - frontend startup hydration because the browser needs to restore the
      server-side workspace selection after authentication.

    Flow: resolve the user id and read the workspace manager's selected
        workspace pointer without loading workspace data.
    """
    user_id = current_user["id"]
    return {"id": workspace_manager.get_selected_workspace_id(user_id)}


@router.put("/me/current-workspace", response_model=SetCurrentWorkspaceResponse)
async def set_my_current_workspace(
    request: CurrentWorkspaceUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Set or clear the authenticated user's selected workspace id.

    Used by:
    - frontend workspace switcher and data-folder reset flows because current
      workspace is a user selection preference, not an implicit API target.

    Flow: submit the requested id or ``None`` to the workspace manager, surface
        missing workspaces as 404, and return the selected id in the response.
    """
    user_id = current_user["id"]
    workspace_id = request.workspace_id
    success = workspace_manager.set_current_workspace(user_id, workspace_id)
    if not success and workspace_id is not None:
        raise WorkspaceNotFoundError("Workspace not found")
    return {"state": "successful", "id": workspace_id}
