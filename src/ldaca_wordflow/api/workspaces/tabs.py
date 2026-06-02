"""Workspace analysis-tab sidecar endpoints.

Persists the analysis tab system's structure into
``<workspace_dir>/tabs.json`` — a Chrome-style tab model layered on top of the
analysis task system. Each analysis type (e.g. ``concordance``) owns a *tab
group*: an ordered list of tabs plus the active tab id. Tabs are deliberately
*thin*: a tab carries only its own id, an optional ``task_id`` (the analysis
result it currently shows), and a display ``title``. Node selection and all
analysis parameters live in the referenced ``AnalysisTask.request`` — never on
the tab — so the task remains the single source of truth.

Endpoints:

    GET  /workspaces/{workspace_id}/tabs
        Returns the parsed tab state, or an empty ``{"groups": {}}`` default
        when the file doesn't exist yet. 404 on unknown workspace.

    PUT  /workspaces/{workspace_id}/tabs
        Replaces the file contents with the request body.

Why PUT (not PATCH): the frontend tab store maintains the canonical tab
structure in memory and writes the whole thing back, mirroring the existing
``ui_state.json`` sidecar. Full replacement keeps both sides simple and avoids
recursive merge logic in the backend.

Used by:
- FastAPI workspace routers (registered in `api/workspaces/__init__.py`),
  frontend analysis-tab features, and backend tests.

Flow:
- Route handlers resolve the user's workspace directory before touching the
  sidecar file.
- GET returns the parsed tab state or the empty default; PUT replaces the
  sidecar with the typed state.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ...core.auth import get_current_user
from ...core.exceptions import InternalServiceError, WorkspaceNotFoundError
from ...core.workspace import workspace_manager

router = APIRouter(prefix="/workspaces", tags=["workspace_tabs"])
logger = logging.getLogger(__name__)

_TABS_FILENAME = "tabs.json"


class AnalysisTab(BaseModel):
    """A single thin analysis tab.

    Carries only identity and a pointer to the analysis result it shows. All
    parameters/node selection live on the referenced ``AnalysisTask.request``.

    Used by:
    - `AnalysisTabGroup` and the GET/PUT tab routes because the frontend tab
      store round-trips this exact shape.
    """

    tab_id: str
    task_id: str | None = None
    title: str = "Untitled"


class AnalysisTabGroup(BaseModel):
    """Ordered tab group for one analysis type.

    Tab order is the array order of ``tabs``; ``active_tab_id`` selects the
    visible tab. Used by `WorkspaceTabsState` to namespace tabs per analysis
    type (concordance, token_frequencies, ...).
    """

    tabs: list[AnalysisTab] = Field(default_factory=list)
    active_tab_id: str | None = None


class WorkspaceTabsState(BaseModel):
    """Full per-workspace analysis-tab state.

    API schema round-tripped by the GET/PUT ``/{workspace_id}/tabs`` routes and
    the frontend tab store. ``groups`` is keyed by analysis type.

    Used by:
    - backend API routes, generated frontend client, and backend tests.
    """

    groups: dict[str, AnalysisTabGroup] = Field(default_factory=dict)


def _tabs_path_for(user_id: str, workspace_id: str) -> Path:
    """Resolve the ``tabs.json`` sidecar path for a user's workspace.

    Called by the GET/PUT tab route handlers. Raises `WorkspaceNotFoundError`
    (HTTP 404) when the workspace directory cannot be resolved.
    """
    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
    if workspace_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    return Path(workspace_dir) / _TABS_FILENAME


@router.get("/{workspace_id}/tabs")
async def get_workspace_tabs(
    workspace_id: str,
    current_user: dict = Depends(get_current_user),
) -> WorkspaceTabsState:
    """Return the persisted analysis-tab state for a workspace.

    Used by the frontend tab store on view entry to restore tabs (and their
    task ids) after a reload. Returns the empty default when no sidecar exists.
    """
    user_id = current_user["id"]
    path = _tabs_path_for(user_id, workspace_id)
    if not path.exists():
        return WorkspaceTabsState()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to read tabs.json for workspace %s: %s — returning default state",
            workspace_id,
            exc,
        )
        return WorkspaceTabsState()
    if not isinstance(data, dict):
        logger.warning(
            "tabs.json for workspace %s was not a JSON object — returning default state",
            workspace_id,
        )
        return WorkspaceTabsState()
    return WorkspaceTabsState.model_validate(data)


@router.put("/{workspace_id}/tabs")
async def put_workspace_tabs(
    workspace_id: str,
    payload: WorkspaceTabsState,
    current_user: dict = Depends(get_current_user),
) -> WorkspaceTabsState:
    """Replace the persisted analysis-tab state for a workspace.

    Used by the frontend tab store whenever tabs are created, closed, renamed,
    reordered, activated, or wired to a new task id. Full-replacement semantics
    mirror the ``ui_state.json`` sidecar.
    """
    user_id = current_user["id"]
    path = _tabs_path_for(user_id, workspace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload.model_dump(), f, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.error(
            "Failed to write tabs.json for workspace %s: %s",
            workspace_id,
            exc,
        )
        raise InternalServiceError("Failed to persist tab state") from exc
    return payload
