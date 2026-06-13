"""Simplified Workspace Manager (single in-memory workspace per user).

Design Goals:
* Each user can have many persisted workspaces on disk.
* At most ONE workspace object is resident in memory per user at any time.
* Switching workspaces always saves & unloads the previous one before loading the next.
* Business logic remains in docworkspace.Workspace / Node; this is only orchestration.
* Backward compatibility deliberately dropped.

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
    cleanup, and return stable workspace metadata to callers.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from docworkspace.workspace.io import read_workspace_metadata, rebase_workspace_sources

from docworkspace import Workspace
from ldaca_wordflow.models import WorkspaceSummary

from .utils import (
    allocate_workspace_folder,
    ensure_display_folder_name,
    get_user_workspace_folder,
)

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Single-workspace-per-user in-memory manager.

    Used by:
    - backend tests, core workspace and worker services because tests need the same
      observable contract that production routes and workers rely on.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    def __init__(self) -> None:
        """Initialize WorkspaceManager state used by workspace persistence and selection.

        Called by:
        - `WorkspaceManager` construction in backend services and tests because tests need the
          same observable contract that production routes and workers rely on.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """

        self._current: dict[str, dict[str, Any]] = {}
        # Per-user task managers (single channel per user, not serialized)
        self._task_managers: dict[str, Any] = {}
        # Track on-disk workspace folder paths per user/workspace
        self._paths: dict[tuple[str, str], Path] = {}

    # ---------------- Core helpers ----------------
    def _path_key(self, user_id: str, workspace_id: str) -> tuple[str, str]:
        """Support workspace persistence and selection with a path key helper.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """

        return (user_id, workspace_id)

    def _get_cached_path(self, user_id: str, workspace_id: str) -> Path | None:
        """Return cached path data used by workspace persistence and selection.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """

        return self._paths.get(self._path_key(user_id, workspace_id))

    def _set_cached_path(self, user_id: str, workspace_id: str, path: Path) -> None:
        """Store cached path data used by workspace persistence and selection.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """

        self._paths[self._path_key(user_id, workspace_id)] = path

    def _clear_user_cached_paths(self, user_id: str) -> None:
        """Remove all cached workspace-folder mappings for a user.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        keys = [key for key in self._paths.keys() if key[0] == user_id]
        for key in keys:
            self._paths.pop(key, None)

    def _refresh_user_workspace_paths(self, user_id: str) -> None:
        """Actively rescan user workspace folders and rebuild id->path cache.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        self._clear_user_cached_paths(user_id)
        user_folder = get_user_workspace_folder(user_id)
        if not user_folder.exists():
            return

        for workspace_dir in user_folder.iterdir():
            if not workspace_dir.is_dir():
                continue

            metadata_path = workspace_dir / "metadata.json"
            if not metadata_path.exists() or not metadata_path.is_file():
                continue

            try:
                with metadata_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception:
                continue

            wid = raw.get("workspace_metadata", {}).get("id")
            if wid:
                self._set_cached_path(user_id, wid, workspace_dir)

    def _get_indexed_path(self, user_id: str, workspace_id: str) -> Path | None:
        """Get workspace folder from cache only (no active directory scans).

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        cached = self._get_cached_path(user_id, workspace_id)
        if cached and cached.exists():
            return cached
        return None

    def _attach_workspace_dir(self, workspace: Workspace, path: Path) -> None:
        """Support workspace persistence and selection with an attach workspace dir helper.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """

        try:
            setattr(workspace, "ws_root_dir", path)
        except Exception as exc:
            logger.debug(
                "Failed to attach workspace_dir metadata to workspace object: %s", exc
            )

    def _workspace_artifacts_dir_from_workspace_dir(self, workspace_dir: Path) -> Path:
        """Return workspace-scoped analysis artifact directory.

        Artifact files are transient analysis outputs and are intentionally kept
        outside workspace payload files while still colocated with workspace data.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        return workspace_dir / "data" / "artifacts"

    def _resolve_workspace_dir(
        self, user_id: str, workspace_id: str, workspace_name: str
    ) -> Path:
        """Resolve or allocate on-disk folder for a workspace id/name.

        Used by:
        - workspace persistence operations because workspace flows need user-scoped paths,
          nodes, artifacts, and task state to stay synchronized.
        Why:
        - Keeps workspace folder naming consistent and discoverable on disk.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        cached = self._get_indexed_path(user_id, workspace_id)
        if cached and cached.exists():
            updated = ensure_display_folder_name(cached, workspace_name)
            self._set_cached_path(user_id, workspace_id, updated)
            return updated

        # Allocate a new folder when none exists
        allocated = allocate_workspace_folder(user_id, workspace_name)
        self._set_cached_path(user_id, workspace_id, allocated)
        return allocated

    # ---------------- Public API ----------------
    def get_current_workspace_id(self, user_id: str) -> str | None:
        """Return current workspace id data used by workspace persistence and selection.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """

        entry = self._current.get(user_id)
        if not entry:
            return None
        return entry.get("wid")

    def get_current_workspace(self, user_id: str) -> Any | None:
        """Return current workspace data used by workspace persistence and selection.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """

        entry = self._current.get(user_id)
        if not entry:
            return None
        return entry.get("workspace")

    def set_current_workspace(self, user_id: str, workspace_id: str | None) -> bool:
        """Store current workspace data used by workspace persistence and selection.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """

        if workspace_id is None:
            self.unload_workspace(user_id, save=True)
            return True
        cid = self.get_current_workspace_id(user_id)
        cws = self.get_current_workspace(user_id)
        if cid == workspace_id and cws is not None:
            return True
        if cid is not None and cws is not None:
            # Strict switch behavior: always unload current before loading next.
            self.unload_workspace(user_id, save=True)
        target_dir = self._get_indexed_path(user_id, workspace_id)
        if target_dir is None:
            logger.warning(
                "Workspace folder not found for workspace %s under user %s",
                workspace_id,
                user_id,
            )
            return False
        try:
            # 1. Read metadata to get workspace name (no node deserialization).
            meta = read_workspace_metadata(target_dir)
            ws_name = meta.get("workspace_metadata", {}).get("name", "")

            # 2. Finalize the on-disk folder name so the path is stable.
            updated_dir = (
                ensure_display_folder_name(target_dir, ws_name)
                if ws_name
                else target_dir
            )

            # 3. Rebase plbin source paths to the finalized folder.
            rebase_workspace_sources(updated_dir)

            # 4. Full load (deserialize nodes — paths are now correct).
            new_ws = Workspace.load(updated_dir)
            self._attach_workspace_dir(new_ws, updated_dir)
            self._set_cached_path(user_id, workspace_id, updated_dir)
        except Exception as e:  # pragma: no cover
            logger.error(
                "Failed to deserialize workspace %s from %s: %s",
                workspace_id,
                target_dir,
                e,
            )
            return False
        if not new_ws:
            return False
        current_path = self._get_cached_path(user_id, workspace_id)
        self._current[user_id] = {
            "wid": workspace_id,
            "workspace": new_ws,
            "path": current_path,
        }
        self.ensure_workspace_artifacts_dir(user_id, workspace_id)
        # Rehydrate persisted analysis task records so task ids referenced by
        # persisted tabs (tabs.json) resolve again after this load.
        restore_dir = self.get_workspace_dir(user_id, workspace_id)
        if restore_dir is not None:
            from ..analysis.persistence import load_workspace_analysis_tasks

            load_workspace_analysis_tasks(user_id, workspace_id, restore_dir)
        return True

    def list_user_workspaces_summaries(self, user_id: str) -> list[dict[str, Any]]:
        """Support workspace persistence and selection with a list user workspaces summaries helper.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """

        summaries: list[dict[str, Any]] = []
        # Active refresh point: called when Data Loader opens and when user presses refresh.
        self._refresh_user_workspace_paths(user_id)

        user_workspace_items = [
            (wid, path)
            for (uid, wid), path in self._paths.items()
            if uid == user_id and path.exists()
        ]

        def _workspace_size_bytes(workspace_dir: Path) -> int:
            """Support workspace persistence and selection with a workspace size bytes helper.

            Called by:
            - The `list_user_workspaces_summaries` local workflow in this module because workspace
              flows need user-scoped paths, nodes, artifacts, and task state to stay synchronized.

            Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
                cleanup, and return stable workspace metadata to callers.
            """

            total = 0
            for file_path in workspace_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                try:
                    total += file_path.stat().st_size
                except Exception:
                    continue
            return total

        for wid, workspace_dir in user_workspace_items:
            workspace_size_byte = _workspace_size_bytes(workspace_dir)
            folder_name = workspace_dir.name

            try:
                ws = Workspace.load(workspace_dir)
                summary_payload = WorkspaceSummary(**ws.info_json()).model_dump()
                summary_payload["workspace_size_Byte"] = workspace_size_byte
                summary_payload["workspace_size_byte"] = workspace_size_byte
                summary_payload["folder_name"] = folder_name
                summaries.append(summary_payload)
            except Exception:
                try:
                    metadata = read_workspace_metadata(workspace_dir)[
                        "workspace_metadata"
                    ]
                    summary_payload = WorkspaceSummary(**metadata).model_dump()
                    summary_payload["workspace_size_Byte"] = workspace_size_byte
                    summary_payload["folder_name"] = folder_name
                    summaries.append(summary_payload)
                except Exception:
                    continue
        return summaries

    def delete_workspace(self, user_id: str, workspace_id: str) -> bool:
        """Delete workspace resources used by workspace persistence and selection.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """

        cid = self.get_current_workspace_id(user_id)
        cws = self.get_current_workspace(user_id)
        if cid is not None and cws is not None and cid == workspace_id:
            try:
                cws.modified_at = datetime.now().isoformat()
                target_dir = self._resolve_workspace_dir(
                    user_id=user_id,
                    workspace_id=cid,
                    workspace_name=cws.name,
                )
                self._attach_workspace_dir(cws, target_dir)
                cws.save(target_dir)
                self._set_cached_path(user_id, cid, target_dir)
            except Exception as exc:
                logger.debug(
                    "Best-effort save before delete failed for workspace %s: %s",
                    workspace_id,
                    exc,
                )
            self._current.pop(user_id, None)
        target_dir = self._get_indexed_path(user_id, workspace_id)
        if target_dir and target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
            self._paths.pop(self._path_key(user_id, workspace_id), None)
            return True
        return False

    def get_task_manager(self, user_id: str):
        """Return or create worker-task manager bound to user.

        Used by:
        - task endpoints and analysis routes submitting background work because they need a
          backend boundary that validates inputs before delegating to workspace or worker state.
                Why:
                - Uses one unified task channel per user while retaining workspace
                    filtering at API/query level via task metadata.

        Refactor note:
        - Lazy import avoids cycles but obscures typing; introducing a protocol or
            factory module could reduce import indirection.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        from ldaca_wordflow.core.worker_task_manager import WorkerTaskManager

        tm = self._task_managers.get(user_id)
        if tm is None:
            tm = WorkerTaskManager()
            self._task_managers[user_id] = tm
        return tm

    def get_workspace_dir(self, user_id: str, workspace_id: str) -> Path | None:
        """Return workspace dir data used by workspace persistence and selection.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """

        cached = self._get_indexed_path(user_id, workspace_id)
        if cached is None:
            self._refresh_user_workspace_paths(user_id)
            cached = self._get_indexed_path(user_id, workspace_id)
        if cached and cached.exists():
            cid = self.get_current_workspace_id(user_id)
            cws = self.get_current_workspace(user_id)
            if cid == workspace_id and cws is not None:
                self._attach_workspace_dir(cws, cached)
            return cached
        return None

    def get_workspace_artifacts_dir(
        self, user_id: str, workspace_id: str
    ) -> Path | None:
        """Get workspace analysis artifact directory path (without creating it).

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        workspace_dir = self.get_workspace_dir(user_id, workspace_id)
        if workspace_dir is None:
            return None
        return self._workspace_artifacts_dir_from_workspace_dir(workspace_dir)

    def ensure_workspace_artifacts_dir(
        self, user_id: str, workspace_id: str
    ) -> Path | None:
        """Create workspace analysis artifact directory if missing.

        Called on workspace load/switch to guarantee a dedicated transient
        artifact location exists for background analysis tasks.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        artifact_dir = self.get_workspace_artifacts_dir(user_id, workspace_id)
        if artifact_dir is None:
            return None
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def clear_workspace_artifacts_dir(self, user_id: str, workspace_id: str) -> bool:
        """Delete workspace analysis artifact directory if it exists.

        Called on workspace unload to remove transient analysis artifacts.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        artifact_dir = self.get_workspace_artifacts_dir(user_id, workspace_id)
        if artifact_dir is None or not artifact_dir.exists():
            return False
        shutil.rmtree(artifact_dir, ignore_errors=True)
        return True

    def unload_workspace(
        self,
        user_id: str,
        workspace_id: str | None = None,
        save: bool = True,
    ) -> bool:
        """Unload current workspace object from memory, optionally persisting first.

        Used by:
        - lifecycle unload/switch operations because workspace flows need user-scoped paths,
          nodes, artifacts, and task state to stay synchronized.
        Why:
        - Enforces one-active-workspace-per-user memory policy.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        cid = self.get_current_workspace_id(user_id)
        cws = self.get_current_workspace(user_id)
        if not cid or not cws:
            return False
        if workspace_id is not None and workspace_id != cid:
            return False
        if save:
            cws.modified_at = datetime.now().isoformat()
            target_dir = self._resolve_workspace_dir(
                user_id=user_id,
                workspace_id=cid,
                workspace_name=cws.name,
            )
            self._attach_workspace_dir(cws, target_dir)
            cws.save(target_dir)
            self._set_cached_path(user_id, cid, target_dir)
        self.clear_workspace_artifacts_dir(user_id, cid)
        # Sweep analysis-cache parquets BEFORE evicting task records, so the
        # workspace dir is still resolvable. The sweep is workspace-scoped and
        # cannot touch another user's data.
        try:
            from .analysis_cache import cleanup_workspace_caches

            cleanup_workspace_caches(user_id, cid)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug("Failed to sweep analysis caches on unload: %s", exc)
        # Snapshot analysis task records to disk BEFORE clearing the in-memory
        # store, so task ids referenced by persisted tabs (tabs.json) stay
        # resolvable after reload. Concordance rebuilds results from the stored
        # request + on-disk node parquet, so the request alone is sufficient.
        workspace_dir = self.get_workspace_dir(user_id, cid)
        if workspace_dir is not None:
            from ..analysis.persistence import save_workspace_analysis_tasks

            save_workspace_analysis_tasks(user_id, cid, workspace_dir)
        self._clear_workspace_tasks(user_id, cid)
        self._current.pop(user_id, None)
        return True

    def _clear_workspace_tasks(self, user_id: str, workspace_id: str) -> None:
        """Drop analysis + worker task records belonging to a workspace.

        Without this, per-user task records (analysis manager tasks and worker
        manager TaskInfo records) leak across workspace
        switches and cause UI state from the previous workspace to hydrate on
        the next one.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        try:
            from ..analysis.manager import get_task_manager as _get_analysis_tm

            _get_analysis_tm(user_id).clear_workspace(workspace_id)
        except Exception as exc:
            logger.debug("Failed to clear analysis tasks on unload: %s", exc)

        worker_tm = self._task_managers.get(user_id)
        if worker_tm is None:
            return
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(
                    worker_tm.clear_tasks(user_id=user_id, workspace_id=workspace_id)
                )
            except Exception as exc:
                logger.debug("Failed to clear worker tasks on unload: %s", exc)
            return
        try:
            loop.create_task(
                worker_tm.clear_tasks(user_id=user_id, workspace_id=workspace_id)
            )
        except Exception as exc:
            logger.debug("Failed to schedule worker task cleanup on unload: %s", exc)

    async def clear_workspace_tasks(self, user_id: str, workspace_id: str) -> None:
        """Await task cleanup for a workspace.

        Called by:
        - `WorkspaceManager` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        try:
            from ..analysis.manager import get_task_manager as _get_analysis_tm

            _get_analysis_tm(user_id).clear_workspace(workspace_id)
        except Exception as exc:
            logger.debug("Failed to clear analysis tasks for workspace: %s", exc)

        worker_tm = self._task_managers.get(user_id)
        if worker_tm is None:
            return
        try:
            await worker_tm.clear_tasks(user_id=user_id, workspace_id=workspace_id)
        except Exception as exc:
            logger.debug("Failed to clear worker tasks for workspace: %s", exc)


workspace_manager = WorkspaceManager()
