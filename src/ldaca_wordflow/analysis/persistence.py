"""Disk persistence for analysis task records.

The analysis tab system stores task ids on persisted tabs (``tabs.json``).
For those task ids to remain resolvable after a workspace unload/reload or a
server restart, the underlying in-memory ``AnalysisTask`` records must be
written to disk on unload and restored on load. Concordance (the pilot)
rebuilds its result table from the persisted ``request`` plus the on-disk node
parquet, so persisting the request is sufficient to fully reconstruct results.

Used by:
- `core.workspace.WorkspaceManager.unload_workspace` to snapshot tasks before
  the in-memory store is cleared.
- `core.workspace.WorkspaceManager.set_current_workspace` to rehydrate tasks
  right after a workspace is loaded.

Flow: locate the per-workspace sidecar file, serialize/deserialize via the
    per-user `TaskManager`, and tolerate missing/corrupt files quietly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .manager import get_task_manager

logger = logging.getLogger(__name__)

_ANALYSIS_TASKS_FILENAME = "analysis_tasks.json"


def _tasks_path(workspace_dir: Path) -> Path:
    """Return the sidecar path for persisted analysis tasks within a workspace dir."""
    return workspace_dir / _ANALYSIS_TASKS_FILENAME


def save_workspace_analysis_tasks(
    user_id: str, workspace_id: str, workspace_dir: Path
) -> None:
    """Write the workspace-scoped analysis task snapshot to disk.

    Called by `WorkspaceManager.unload_workspace` before the per-user task store
    is cleared, so that task ids referenced by persisted tabs survive the unload.

    Flow: serialize the workspace slice of the task store and write it as JSON;
        if there are no tasks, remove any stale sidecar to avoid restoring
        deleted state.
    """
    try:
        manager = get_task_manager(user_id)
        payload: dict[str, Any] = manager.serialize_workspace(workspace_id)
        path = _tasks_path(workspace_dir)
        if not payload.get("tasks"):
            path.unlink(missing_ok=True)
            return
        path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - persistence is best-effort
        logger.warning(
            "Failed to persist analysis tasks for workspace %s: %s", workspace_id, exc
        )


def load_workspace_analysis_tasks(
    user_id: str, workspace_id: str, workspace_dir: Path
) -> None:
    """Restore the workspace-scoped analysis task snapshot from disk.

    Called by `WorkspaceManager.set_current_workspace` right after a workspace is
    loaded, making task ids stored on persisted tabs resolvable again.

    Flow: read the sidecar (no-op when absent), then replay records and
        current-task pointers into the per-user task store.
    """
    path = _tasks_path(workspace_dir)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "Failed to read persisted analysis tasks for workspace %s: %s",
            workspace_id,
            exc,
        )
        return
    if not isinstance(data, dict):
        return
    get_task_manager(user_id).restore_workspace(data)
