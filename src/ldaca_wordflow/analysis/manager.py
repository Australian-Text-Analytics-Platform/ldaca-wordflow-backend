"""Analysis storage manager.

Provides per-user in-memory storage for analysis task records.  Each
``AnalysisTask`` tracks the ``workspace_id`` it was created for, so
tasks can be bulk-cleared when a workspace is unloaded.

Used by:
- Analysis routes, worker result persistence, and backend tests because they need a
  backend boundary that validates inputs before delegating to workspace or worker state.

Flow: normalize request payloads, update per-user task maps, maintain current-tab
    pointers, and walk parent-child links for cleanup.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from .models import AnalysisStatus, AnalysisTask, BaseAnalysisRequest
from .results import GenericAnalysisResult

logger = logging.getLogger(__name__)

# In-memory storage: user_id -> TaskManagerStore
_TASK_MANAGER_STORE: dict[str, TaskManagerStore] = {}


class TaskManagerStore:
    """Per-user in-memory storage for analysis task records.

    Used by:
    - analysis task helpers because analysis flows need per-user task state to survive
      across route calls and worker result persistence.

    Flow: normalize request payloads, update per-user task maps, maintain current-tab
        pointers, and walk parent-child links for cleanup.
    """

    def __init__(self) -> None:
        """Initialize TaskManagerStore state used by analysis task storage.

        Called by:
        - `TaskManagerStore` construction in backend services and tests because tests need the
          same observable contract that production routes and workers rely on.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        self.tasks: dict[str, AnalysisTask] = {}
        self.current_task_ids: dict[str, str] = {}

    def get_task(self, task_id: str) -> AnalysisTask | None:
        """Run the get task background job submitted by API routes.

        Called by:
        - `TaskManagerStore` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        return self.tasks.get(task_id)

    def save_task(self, task: AnalysisTask) -> None:
        """Run the save task background job submitted by API routes.

        Called by:
        - `TaskManagerStore` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        self.tasks[task.task_id] = task

    def get_all_tasks(self) -> list[AnalysisTask]:
        """Return all tasks data used by analysis task storage.

        Called by:
        - `TaskManagerStore` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        return list(self.tasks.values())

    def set_current_task(self, tab: str, task_id: str) -> None:
        """Run the set current task background job submitted by API routes.

        Called by:
        - `TaskManagerStore` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        self.current_task_ids[tab] = task_id

    def get_current_task_ids(self, tab: str) -> list[str]:
        """Return current task ids data used by analysis task storage.

        Called by:
        - `TaskManagerStore` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        task_id = self.current_task_ids.get(tab)
        return [task_id] if task_id else []

    def clear_task(self, task_id: str) -> None:
        """Run the clear task background job submitted by API routes.

        Called by:
        - `TaskManagerStore` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        task = self.tasks.pop(task_id, None)
        if task is not None and task.parent_task_id:
            parent = self.tasks.get(task.parent_task_id)
            if parent is not None and task_id in parent.child_task_ids:
                parent.child_task_ids = [
                    child_id
                    for child_id in parent.child_task_ids
                    if child_id != task_id
                ]
        for tab, current_id in list(self.current_task_ids.items()):
            if current_id == task_id:
                del self.current_task_ids[tab]

    def link_child_task(self, parent_task_id: str, child_task_id: str) -> None:
        """Run the link child task background job submitted by API routes.

        Called by:
        - `TaskManagerStore` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        parent = self.tasks.get(parent_task_id)
        if parent is None:
            return
        child = self.tasks.get(child_task_id)
        if child is not None:
            child.parent_task_id = parent_task_id
        if child_task_id not in parent.child_task_ids:
            parent.child_task_ids.append(child_task_id)

    def get_descendant_task_ids(self, task_id: str) -> list[str]:
        """Return descendant task ids data used by analysis task storage.

        Called by:
        - `TaskManagerStore` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        descendants: list[str] = []
        visited: set[str] = set()

        def visit(parent_id: str) -> None:
            """Visit task relationships while analysis task storage walks descendants.

            Called by:
            - The `get_descendant_task_ids` local workflow in this module because the local analysis
              task state management flow needs this step kept close to the code that consumes it.

            Flow: normalize request payloads, update per-user task maps, maintain current-tab
                pointers, and walk parent-child links for cleanup.
            """

            parent = self.tasks.get(parent_id)
            child_ids = parent.child_task_ids if parent is not None else []
            if not child_ids:
                child_ids = [
                    child_id
                    for child_id, child in self.tasks.items()
                    if child.parent_task_id == parent_id
                ]
            for child_id in child_ids:
                if child_id in visited:
                    continue
                visited.add(child_id)
                visit(child_id)
                descendants.append(child_id)

        visit(task_id)
        return descendants


class TaskManager:
    """Per-user task storage keyed by task_id with per-tab current mapping.

    Used by:
    - analysis task helpers, backend tests, core workspace and worker services because tests
      need the same observable contract that production routes and workers rely on.

    Flow: normalize request payloads, update per-user task maps, maintain current-tab
        pointers, and walk parent-child links for cleanup.
    """

    def __init__(self, user_id: str) -> None:
        """Initialize TaskManager state used by analysis task storage.

        Called by:
        - `TaskManager` construction in backend services and tests because tests need the same
          observable contract that production routes and workers rely on.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        self.user_id = user_id
        if user_id not in _TASK_MANAGER_STORE:
            _TASK_MANAGER_STORE[user_id] = TaskManagerStore()
        self.store = _TASK_MANAGER_STORE[user_id]

    def create_task(
        self,
        request: BaseAnalysisRequest | dict[str, Any],
    ) -> str:
        """Create and store a new pending analysis task.

        The current workspace is resolved automatically from
        ``workspace_manager`` so callers don't need to pass it.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """
        from ..core.workspace import workspace_manager

        workspace_id = workspace_manager.get_current_workspace_id(self.user_id) or ""
        task_id = str(uuid4())
        normalized_request = self._normalize_request(request)
        task = AnalysisTask(
            task_id=task_id,
            user_id=self.user_id,
            workspace_id=workspace_id,
            request=normalized_request,
            status=AnalysisStatus.PENDING,
        )
        self.store.save_task(task)
        logger.info("Created analysis task %s for user %s", task_id, self.user_id)
        return task_id

    def get_task(self, task_id: str) -> AnalysisTask | None:
        """Run the get task background job submitted by API routes.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        return self.store.get_task(task_id)

    def save_task(self, task: AnalysisTask) -> None:
        """Run the save task background job submitted by API routes.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        self.store.save_task(task)

    def set_current_task(self, tab: str, task_id: str) -> None:
        """Pin ``task_id`` as the current task for ``tab``.

        When a different task was previously current for the same tab, the
        displaced task is evicted along with any analysis-cache parquets it
        owned. This keeps the in-memory task store and the on-disk side-effect
        caches bounded to "at most one record per tab" — matching the
        frontend's mental model where rerunning a tool replaces the previous
        result rather than accumulating.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """
        previous_ids = self.store.get_current_task_ids(tab)
        previous_id = previous_ids[0] if previous_ids else None
        if previous_id and previous_id != task_id:
            self.clear_task(previous_id)
        self.store.set_current_task(tab, task_id)

    def get_current_task_ids(self, tab: str) -> list[str]:
        """Return current task ids data used by analysis task storage.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        return self.store.get_current_task_ids(tab)

    def update_task(self, task_id: str, result: Any) -> None:
        """Run the update task background job submitted by API routes.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        task = self.store.get_task(task_id)
        if task is None:
            logger.warning("Attempted to update non-existent task %s", task_id)
            return
        task.result = result
        task.status = AnalysisStatus.COMPLETED
        task.updated_at = datetime.now()
        self.store.save_task(task)
        logger.info("Analysis task %s completed", task_id)

    def clear_task(self, task_id: str) -> None:
        """Run the clear task background job submitted by API routes.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        task = self.store.get_task(task_id)
        if task is not None:
            from ..core.task_artifacts import cleanup_analysis_task_artifacts

            try:
                cleanup_analysis_task_artifacts(self.user_id, task)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to clean artifacts for analysis task %s: %s",
                    task_id,
                    exc,
                )
        self.store.clear_task(task_id)

    def link_child_task(self, parent_task_id: str, child_task_id: str) -> None:
        """Run the link child task background job submitted by API routes.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        self.store.link_child_task(parent_task_id, child_task_id)

    def get_descendant_task_ids(self, task_id: str) -> list[str]:
        """Return descendant task ids data used by analysis task storage.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        return self.store.get_descendant_task_ids(task_id)

    def clear_all(self) -> list[str]:
        """Clear all tasks for this user (used by test fixtures)."""
        to_remove = [task.task_id for task in self.store.get_all_tasks()]
        for task_id in to_remove:
            self.clear_task(task_id)
        return to_remove

    def clear_workspace(self, workspace_id: str) -> list[str]:
        """Clear all tasks belonging to *workspace_id*.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
            cleanup, and return stable workspace metadata to callers.
        """
        to_remove = [
            task.task_id
            for task in self.store.get_all_tasks()
            if task.workspace_id == workspace_id
        ]
        for task_id in to_remove:
            self.clear_task(task_id)
        return to_remove

    def get_all_tasks(self) -> list[AnalysisTask]:
        """Return all tasks data used by analysis task storage.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        return self.store.get_all_tasks()

    def _normalize_request(
        self, request: BaseAnalysisRequest | dict[str, Any]
    ) -> BaseAnalysisRequest:
        """Normalize request values before analysis task storage uses them.

        Called by:
        - `TaskManager` instances owned by backend services, routes, and tests because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: normalize request payloads, update per-user task maps, maintain current-tab
            pointers, and walk parent-child links for cleanup.
        """

        if isinstance(request, BaseAnalysisRequest):
            return request
        return BaseAnalysisRequest.model_validate(request)

    def serialize_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Serialize analysis task records + current-tab pointers for a workspace.

        Why:
        - The analysis tab system keeps task ids on persisted tabs (``tabs.json``).
          For those task ids to stay resolvable after a workspace unload/reload
          (or a server restart), the underlying ``AnalysisTask`` records must be
          written to disk and restored on load. Concordance (the pilot) rebuilds
          its result table from the stored ``request`` + on-disk node parquet, so
          persisting the request is enough; the readiness ``result`` marker is
          round-tripped through ``GenericAnalysisResult`` for generality.

        Called by:
        - `analysis.persistence.save_workspace_analysis_tasks` because workspace
          unload needs a JSON-safe snapshot of the per-user task store scoped to
          one workspace.

        Flow: collect workspace-scoped tasks, JSON-encode request/result/status,
            and emit the current-task-id pointers that drive supersede behavior.
        """
        tasks_payload: list[dict[str, Any]] = []
        workspace_task_ids: set[str] = set()
        for task in self.store.get_all_tasks():
            if task.workspace_id != workspace_id:
                continue
            workspace_task_ids.add(task.task_id)
            result_payload = task.result.to_json() if task.result is not None else None
            tasks_payload.append(
                {
                    "task_id": task.task_id,
                    "user_id": task.user_id,
                    "workspace_id": task.workspace_id,
                    "status": task.status.value,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat(),
                    "request": task.request.model_dump() if task.request else {},
                    "result": result_payload,
                    "error": task.error,
                    "parent_task_id": task.parent_task_id,
                    "child_task_ids": list(task.child_task_ids),
                }
            )
        current_pointers = {
            tab: task_id
            for tab, task_id in self.store.current_task_ids.items()
            if task_id in workspace_task_ids
        }
        return {"tasks": tasks_payload, "current_task_ids": current_pointers}

    def restore_workspace(self, data: dict[str, Any]) -> None:
        """Rehydrate analysis task records + current-tab pointers from disk.

        Counterpart to `serialize_workspace`. Reconstructs each ``AnalysisTask``
        with a generic ``BaseAnalysisRequest`` (``extra=allow`` preserves every
        field) and a ``GenericAnalysisResult`` wrapper, then restores the
        current-task pointers so per-tab supersede continues to work.

        Called by:
        - `analysis.persistence.load_workspace_analysis_tasks` because workspace
          load needs to make persisted tab task ids resolvable again.

        Flow: rebuild typed task records, save them into the per-user store, and
            replay the current-task-id pointers.
        """
        for record in data.get("tasks", []):
            try:
                result_dict = record.get("result")
                task = AnalysisTask(
                    task_id=record["task_id"],
                    user_id=record["user_id"],
                    workspace_id=record["workspace_id"],
                    status=AnalysisStatus(record.get("status", "completed")),
                    created_at=datetime.fromisoformat(record["created_at"]),
                    updated_at=datetime.fromisoformat(record["updated_at"]),
                    request=BaseAnalysisRequest.model_validate(
                        record.get("request") or {}
                    ),
                    result=(
                        GenericAnalysisResult(result_dict)
                        if isinstance(result_dict, dict)
                        else None
                    ),
                    error=record.get("error"),
                    parent_task_id=record.get("parent_task_id"),
                    child_task_ids=list(record.get("child_task_ids", [])),
                )
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed persisted analysis task: %s", exc)
                continue
            self.store.save_task(task)
        for tab, task_id in (data.get("current_task_ids") or {}).items():
            if self.store.get_task(task_id) is not None:
                self.store.current_task_ids[tab] = task_id


def get_task_manager(user_id: str) -> TaskManager:
    """Return the analysis task manager for a user.

    A single user can only have one workspace loaded at a time, so the
    manager is keyed by *user_id* only.  Individual tasks track their
    ``workspace_id`` internally.

    Used by:
    - backend API routes, backend tests, core workspace and worker services because they
      need a backend boundary that validates inputs before delegating to workspace or worker
      state.

    Flow: normalize request payloads, update per-user task maps, maintain current-tab
        pointers, and walk parent-child links for cleanup.
    """
    return TaskManager(user_id)
