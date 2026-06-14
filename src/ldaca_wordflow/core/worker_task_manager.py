"""Per-user worker task orchestration for long-running workspace jobs.

Used by:
- ``workspace_manager.get_task_manager`` to lazily allocate one manager per user because
  task state, progress queues, and subscribers must not leak across user workspaces.
- workspace, file, and analysis API routes that submit/cancel/clear worker jobs because
  those routes need a shared lifecycle owner for processes that outlive the
  request/response cycle.
- ``api.tasks`` to list tasks and stream task lifecycle events to the frontend because
  the Task Center needs one API-visible source of truth for active and historical
  background work.

Flow: register submitted work with metadata, feed progress queues from worker processes,
    emit bounded SSE events, reconcile futures into ``TaskInfo`` records, cancel or
    clear task trees on demand, and clean up artifacts after terminal states.
"""

import asyncio
import builtins
import logging
import multiprocessing as mp
import os
import queue as std_queue
import signal
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .task_artifacts import cleanup_worker_task_artifacts
from .worker import TASK_REGISTRY, get_worker_pool

logger = logging.getLogger(__name__)


def _is_terminal_task_event(event: dict[str, Any]) -> bool:
    """Identify final task-change events so ``emit`` preserves them for SSE clients.

    Called by:
    - ``WorkerTaskManager.emit`` because final task states should displace stale queued
      updates before a bounded SSE queue drops them.

    Flow: accept only ``task_changed`` events, confirm the payload is a task dictionary,
        normalize its state, and return true for the three terminal states the frontend must
        not miss.
    """

    if event.get("type") != "task_changed":
        return False
    task = event.get("task")
    if not isinstance(task, dict):
        return False
    state = str(task.get("state") or "").lower()
    return state in {"successful", "failed", "cancelled"}


ANALYSIS_TASK_TYPES = {
    "topic_modeling",
    "concordance",
    "token_frequencies",
}


class TaskStatus(str, Enum):
    """Canonical worker task states serialized by ``TaskInfo`` and task APIs.

    Used by:
    - ``TaskInfo`` and task response serializers because every worker lifecycle transition
      needs stable string values for API clients, tests, and SSE payloads.

    Flow: expose the only accepted pending, running, successful, failed, and cancelled
        labels so routes and frontend code do not infer state names from worker
        implementation details.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Runtime worker task record consumed by ``WorkerTaskManager`` and task endpoints.

    Used by:
    - ``WorkerTaskManager`` because it needs to keep the worker ``Future`` together with
      API-facing metadata, progress, timestamps, parent-child links, and worker process IDs.
    - ``api.tasks`` and tests because list/clear/cancel endpoints serialize this record to
      explain task state to the frontend and assert lifecycle behavior.

    Flow: store immutable task identity plus mutable runtime fields, let the manager update
        progress and worker PID as messages arrive, and let ``update_status`` reconcile the
        future into terminal API state.
    """

    id: str
    future: Future
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    progress: float = 0.0  # 0..1 for UI progress bars
    progress_message: str | None = None
    task_type: str = ""
    name: str = ""
    user_id: str = ""
    workspace_id: str = ""
    parent_task_id: str | None = None
    worker_pid: int | None = None  # PID of the worker process, set once known

    def update_status(self):
        """Refresh API-visible state from the underlying worker ``Future``.

        Called by:
        - task listing and lifecycle code before returning ``TaskInfo`` to callers because the
          stored record can lag behind the worker ``Future`` between progress messages.

        Flow: mark cancelled futures as cancelled, resolve completed futures into success with
            result/progress or failure with an error message, and mark unfinished futures
            running while recording the first start timestamp.
        """
        if self.future.cancelled():
            self.status = TaskStatus.CANCELLED
            if not self.finished_at:
                self.finished_at = time.time()
        elif self.future.done():
            if not self.finished_at:
                self.finished_at = time.time()
            try:
                self.result = self.future.result()
                self.status = TaskStatus.SUCCESSFUL
                self.progress = 1.0
                self.progress_message = "Completed"
            except Exception as e:
                self.error = str(e)
                self.status = TaskStatus.FAILED
                self.progress = -1.0  # Indicates failure
                self.progress_message = f"Failed: {str(e)}"
        else:
            # Future is still running
            self.status = TaskStatus.RUNNING
            if not self.started_at:
                self.started_at = time.time()


class WorkerTaskManager:
    """Owns worker ``Future`` records for one user's workspace operations.

    Used by:
    - ``WorkspaceManager.get_task_manager`` as the per-user worker manager factory because
      each authenticated user needs isolated task records and event subscribers.
    - ``api.files`` and workspace lifecycle routes for imports/downloads because large file
      operations must keep progressing after the HTTP request starts.
    - analysis routes for topic modeling, concordance, quotation, and token frequency jobs
      because CPU-heavy work runs in the worker pool and reports back through this manager.
    - ``api.tasks`` for list/clear/cancel/SSE task-center operations because the frontend
      Task Center needs one lifecycle controller for every background task it displays.

    Flow: maintain task records under an async lock, allocate multiprocessing progress
        queues, submit named worker functions, monitor progress and future completion, fan
        out task events to bounded subscriber queues, and cancel or clear individual tasks
        and task trees when routes request cleanup.
    """

    def __init__(self):
        """Initialize per-user task, progress, and subscriber stores.

        Called by:
        - ``WorkspaceManager.get_task_manager`` and backend tests because they need a fresh
          manager with isolated in-memory task state for one user/test scenario.

        Flow: create the task/progress dictionaries, async locks, multiprocessing manager-backed
            progress queue registry, and per-user SSE subscriber sets that later methods mutate
            during task lifecycles.
        """

        self._tasks: dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()
        self._progress_store: dict[str, dict[str, Any]] = {}  # task_id -> progress info
        self._mp_manager = mp.Manager()
        self._task_progress_queues: dict[str, Any] = {}

        # Event bus for real-time updates (single channel per user)
        self._subscribers: dict[
            str, set[asyncio.Queue]
        ] = {}  # user_id -> set of queues
        self._subscriber_lock = asyncio.Lock()

    async def subscribe(
        self, user_id: str, workspace_id: str | None = None
    ) -> asyncio.Queue:
        """Create the queue consumed by ``api.tasks.stream_tasks`` for SSE updates.

        Called by:
        - ``api.tasks.stream_tasks`` because each browser connection needs a bounded queue where
          task events can wait until the SSE generator writes them.

        Flow: allocate a bounded queue, register it under the user key while holding the
            subscriber lock, create the subscriber set if needed, and return the queue for the
            streaming route to read.
        """
        queue = asyncio.Queue(maxsize=100)  # Bounded to prevent memory leaks
        key = user_id

        async with self._subscriber_lock:
            if key not in self._subscribers:
                self._subscribers[key] = set()
            self._subscribers[key].add(queue)

        logger.debug(f"Subscribed to events for user {user_id}")
        return queue

    async def unsubscribe(
        self, user_id: str, workspace_id: str | None, queue: asyncio.Queue
    ):
        """Remove an SSE queue when ``api.tasks.stream_tasks`` disconnects.

        Called by:
        - ``api.tasks.stream_tasks`` cleanup because disconnected browser streams should stop
          receiving task events and should not keep queue objects alive.

        Flow: remove the queue from the user's subscriber set under lock, delete the empty set
            when the last stream disconnects, and return the queue for parity with the subscribe
            call path.
        """
        key = user_id

        async with self._subscriber_lock:
            if key in self._subscribers:
                self._subscribers[key].discard(queue)
                if not self._subscribers[key]:  # Clean up empty sets
                    del self._subscribers[key]

        logger.debug(f"Unsubscribed from events for user {user_id}")
        return queue

    async def emit(self, user_id: str, workspace_id: str, event: dict[str, Any]):
        """Broadcast task/workspace events to Task Center stream subscribers.

        Called by:
        - submission, completion, stop, and clear paths in this manager because every task
          lifecycle change must reach open Task Center streams.
        - ``api.tasks.clear_tasks`` when an analysis-only task is removed because route-level
          cleanup still needs to notify frontend state stores.

        Flow: find the user's subscriber queues, classify terminal task events, enqueue the
            event without blocking, replace one stale queued event for terminal states when
            possible, drop only overflowed non-terminal updates, and prune queues that can no
            longer accept events.
        """
        key = user_id

        async with self._subscriber_lock:
            if key not in self._subscribers:
                logger.debug(
                    f"No subscribers for user {user_id} - event {event.get('type')} dropped"
                )
                return

            subscriber_count = len(self._subscribers[key])
            logger.debug(
                f"Emitting {event.get('type')} event to {subscriber_count} subscribers for user {user_id}"
            )

            is_terminal = _is_terminal_task_event(event)

            # Send to all subscribers. For terminal task events, prefer replacing
            # one stale queued event before dropping subscriber.
            active_queues = set()
            for queue in self._subscribers[key]:
                try:
                    queue.put_nowait(event)
                    active_queues.add(queue)
                except asyncio.QueueFull:
                    if is_terminal:
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        try:
                            queue.put_nowait(event)
                            active_queues.add(queue)
                        except asyncio.QueueFull:
                            logger.warning(
                                f"Event queue full for user {user_id}, dropping terminal event"
                            )
                    else:
                        logger.warning(
                            f"Event queue full for user {user_id}, dropping event"
                        )
                        # Keep subscriber active; drop only this stale/non-terminal event.
                        active_queues.add(queue)

            self._subscribers[key] = active_queues
            if not self._subscribers[key]:
                del self._subscribers[key]

    def _serialize_task(self, task_info: TaskInfo) -> dict[str, Any]:
        """Shape ``TaskInfo`` records for ``api.tasks`` responses and SSE payloads.

        Called by:
        - `WorkerTaskManager` instances owned by backend services, routes, and tests because
          they need a backend boundary that validates inputs before delegating to workspace or
          worker state.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        return {
            "task_id": task_info.id,
            "task_type": task_info.task_type,
            "name": task_info.name,
            "user_id": task_info.user_id,
            "workspace_id": task_info.workspace_id,
            # Public API field renamed from 'status' -> 'state'
            "state": task_info.status.value,
            "created_at": task_info.created_at,
            "started_at": task_info.started_at,
            "finished_at": task_info.finished_at,
            "progress": task_info.progress,
            "progress_message": task_info.progress_message,
            "parent_task_id": task_info.parent_task_id,
        }

    def _task_tree_ids_locked(self, task_id: str) -> builtins.list[str]:
        """Return child worker IDs for clear paths while the task lock is held.

        Called by:
        - `WorkerTaskManager` instances owned by backend services, routes, and tests because
          they need a backend boundary that validates inputs before delegating to workspace or
          worker state.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        ordered: builtins.list[str] = []
        visited: set[str] = set()

        def visit(parent_id: str) -> None:
            """Walk nested materialize/detach children before their parent is cleared.

            Called by:
            - The `_task_tree_ids_locked` local workflow in this module because background jobs need
              one lifecycle owner for submission, progress, cancellation, and artifact cleanup.

            Flow: resolve task records under locks, reconcile progress or future state, emit bounded
                subscriber events, and clean up queues or artifacts when a task reaches a terminal
                state.
            """

            for child_id, child in list(self._tasks.items()):
                if child.parent_task_id != parent_id or child_id in visited:
                    continue
                visited.add(child_id)
                visit(child_id)
                ordered.append(child_id)

        visit(task_id)
        if task_id in self._tasks and task_id not in visited:
            ordered.append(task_id)
        return ordered

    async def get_descendant_task_ids(self, task_id: str) -> builtins.list[str]:
        """List child worker IDs used by task-clear endpoints before deletion.

        Called by:
        - `WorkerTaskManager` instances owned by backend services, routes, and tests because
          they need a backend boundary that validates inputs before delegating to workspace or
          worker state.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        async with self._lock:
            return [
                tid for tid in self._task_tree_ids_locked(task_id) if tid != task_id
            ]

    def _cleanup_progress_queue(self, task_id: str) -> None:
        """Close the multiprocessing progress queue owned by a submitted task.

        Called by:
        - `WorkerTaskManager` instances owned by backend services, routes, and tests because
          they need a backend boundary that validates inputs before delegating to workspace or
          worker state.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """

        progress_queue = self._task_progress_queues.pop(task_id, None)
        if progress_queue is None:
            return
        try:
            close = getattr(progress_queue, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            logger.debug("Failed to close progress queue for task %s: %s", task_id, exc)

    async def _consume_worker_progress(
        self,
        task_info: TaskInfo,
        user_id: str,
        workspace_id: str,
        progress_queue: Any,
    ) -> None:
        """Mirror worker progress messages into task state and SSE events.

        Used by:
        - ``submit_task`` background monitor for every worker job because background jobs need
          one lifecycle owner for submission, progress, cancellation, and artifact cleanup.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """

        try:
            while True:
                if task_info.future.done():
                    break

                try:
                    payload = await asyncio.to_thread(progress_queue.get, True, 0.5)
                except std_queue.Empty:
                    continue
                except Exception as exc:
                    logger.debug(
                        "Progress consumer stopped for task %s: %s",
                        task_info.id,
                        exc,
                    )
                    break

                if not isinstance(payload, dict):
                    continue

                # PID notification sent by _pid_reporting_wrapper at process start
                if payload.get("type") == "pid":
                    pid = payload.get("pid")
                    if isinstance(pid, int):
                        task_info.worker_pid = pid
                    continue

                raw_progress = payload.get("progress")
                message = payload.get("message")

                try:
                    progress_value = float(raw_progress)
                except TypeError, ValueError:
                    continue

                message_value = str(message) if message is not None else ""
                now = time.time()

                self._progress_store[task_info.id] = {
                    "progress": progress_value,
                    "message": message_value,
                    "updated_at": payload.get("timestamp", now),
                    "source": "real",
                }

                task_info.progress = progress_value
                task_info.progress_message = message_value

                await self.emit(
                    user_id,
                    workspace_id,
                    {
                        "type": "task_changed",
                        "task": self._serialize_task(task_info),
                        "timestamp": now,
                    },
                )

            while True:
                try:
                    payload = await asyncio.to_thread(progress_queue.get_nowait)
                except std_queue.Empty:
                    break
                except Exception:
                    break

                if not isinstance(payload, dict):
                    continue

                raw_progress = payload.get("progress")
                message = payload.get("message")
                try:
                    progress_value = float(raw_progress)
                except TypeError, ValueError:
                    continue

                message_value = str(message) if message is not None else ""
                self._progress_store[task_info.id] = {
                    "progress": progress_value,
                    "message": message_value,
                    "updated_at": payload.get("timestamp", time.time()),
                    "source": "real",
                }
                task_info.progress = progress_value
                task_info.progress_message = message_value
        finally:
            self._cleanup_progress_queue(task_info.id)

    def _reconcile_task_progress(self, task_info: TaskInfo) -> None:
        """Refresh task status and progress fields.

        For terminal states, `TaskInfo.update_status()` already writes the
        correct final progress values on the task; we mirror them into the
        progress store so follow-up SSE clients see a consistent view.
        For running tasks we read back the latest worker-reported progress.

        Called by:
        - `WorkerTaskManager` instances owned by backend services, routes, and tests because
          they need a backend boundary that validates inputs before delegating to workspace or
          worker state.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        task_info.update_status()
        task_id = task_info.id

        if task_info.status == TaskStatus.SUCCESSFUL:
            self._progress_store[task_id] = {
                "progress": 1.0,
                "message": "Completed successfully",
                "updated_at": time.time(),
            }
        elif task_info.status == TaskStatus.FAILED:
            self._progress_store[task_id] = {
                "progress": -1.0,
                "message": f"Failed: {task_info.error or 'Unknown error'}",
                "updated_at": time.time(),
            }
        elif task_info.status == TaskStatus.CANCELLED:
            self._progress_store[task_id] = {
                "progress": -1.0,
                "message": "Cancelled",
                "updated_at": time.time(),
            }
        elif task_id in self._progress_store:
            progress_info = self._progress_store[task_id]
            task_info.progress = progress_info["progress"]
            task_info.progress_message = progress_info["message"]

    async def _monitor_task_completion(
        self, task_info: TaskInfo, user_id: str, workspace_id: str
    ):
        """Monitor worker completion and persist/emit final task state.

        Used by:
        - `submit_task` background completion monitor because background jobs need one lifecycle
          owner for submission, progress, cancellation, and artifact cleanup.
        Why:
        - Centralizes completion side effects (analysis persistence, workspace
          updates, and event emission) in one lifecycle path.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        result_persisted = False

        try:
            # Wait for the task to complete
            result = await asyncio.wrap_future(task_info.future)

            # Update task status
            task_info.update_status()

            # Workers catch exceptions internally and return
            # {"state": "failed", ...} as a normal value, so the future
            # completes without error and update_status() marks SUCCESSFUL.
            # Detect this and override the status.
            if isinstance(result, dict) and result.get("state") == "failed":
                task_info.status = TaskStatus.FAILED
                task_info.error = result.get("error") or result.get(
                    "message", "Worker reported failure"
                )
                task_info.progress = -1.0
                task_info.progress_message = f"Failed: {task_info.error}"

            if task_info.status == TaskStatus.SUCCESSFUL:
                task_type = task_info.task_type

                # Handle DETACH tasks (add node to workspace)
                if task_type in [
                    "concordance_detach",
                    "concordance_dispersion_detach",
                    "quotation_detach",
                ]:
                    try:
                        from docworkspace.node.io import from_dict as node_from_dict

                        from .workspace import workspace_manager

                        # Worker contract: {"state": "successful", "result": {...}}
                        if not isinstance(result, dict):
                            raise ValueError(
                                "Detach task result must be a dictionary payload"
                            )

                        data = result.get("result")
                        if not isinstance(data, dict):
                            raise ValueError(
                                "Detach task result missing structured result payload"
                            )

                        node_payload = data.get("node_payload")
                        if not isinstance(node_payload, dict):
                            raise ValueError("Task result missing node_payload")

                        if (
                            workspace_manager.get_current_workspace_id(user_id)
                            != workspace_id
                        ):
                            if not workspace_manager.set_current_workspace(
                                user_id, workspace_id
                            ):
                                raise RuntimeError("Workspace not found")
                        workspace = workspace_manager.get_current_workspace(user_id)
                        if workspace is None:
                            raise RuntimeError("Workspace not found")

                        target_dir = workspace_manager._resolve_workspace_dir(
                            user_id=user_id,
                            workspace_id=workspace_id,
                            workspace_name=workspace.name,
                        )
                        workspace_manager._attach_workspace_dir(workspace, target_dir)

                        new_node = node_from_dict(node_payload, base_dir=target_dir)
                        workspace.add_node(new_node)
                        # Smart insertion: keep async-detached result nodes directly
                        # below their mother node in the list view, matching the
                        # synchronous creation path.
                        workspace.place_node_after_parent(new_node)
                        workspace.modified_at = datetime.now().isoformat()
                        workspace.save(target_dir)
                        workspace_manager._set_cached_path(
                            user_id, workspace_id, target_dir
                        )

                        result_persisted = True
                        await self.emit(
                            user_id,
                            workspace_id,
                            {
                                "type": "workspace_updated",
                                "task_type": task_type,
                                "task_id": task_info.id,
                                "new_node_id": new_node.id,
                                "timestamp": time.time(),
                            },
                        )

                        # Dispersion detach has a materialise side-effect: when
                        # the slow path runs (no client-provided
                        # `materialized_path`), the worker also writes the
                        # flat per-hit parquet so subsequent bin-filtered
                        # detaches reuse it. Surface this to the frontend
                        # via the same `analysis_materialized` event the
                        # standalone materialise task emits — the dispersion
                        # view then auto-switches "page above" to "whole
                        # data block" without forcing the user to press
                        # "Process All" again.
                        if task_type == "concordance_dispersion_detach":
                            materialized_path = data.get("materialized_path")
                            parent_task_id = data.get("parent_task_id")
                            parent_node_id_for_mat = data.get("parent_node_id")
                            if (
                                materialized_path
                                and parent_task_id
                                and parent_node_id_for_mat
                            ):
                                try:
                                    from ..analysis.manager import (
                                        get_task_manager,
                                    )

                                    task_manager = get_task_manager(user_id)
                                    parent_task = task_manager.get_task(parent_task_id)
                                    if parent_task is not None:
                                        materialize_summary = {
                                            "record_count": data.get("record_count"),
                                            "unique_documents_with_hits": data.get(
                                                "unique_documents_with_hits"
                                            ),
                                            "total_source_documents": data.get(
                                                "total_source_documents"
                                            ),
                                        }
                                        existing = (
                                            getattr(
                                                parent_task.request,
                                                "materialized_paths",
                                                None,
                                            )
                                            or {}
                                        )
                                        updated = dict(existing)
                                        updated[str(parent_node_id_for_mat)] = str(
                                            materialized_path
                                        )
                                        parent_task.request.materialized_paths = updated

                                        existing_summaries = (
                                            getattr(
                                                parent_task.request,
                                                "materialize_summaries",
                                                None,
                                            )
                                            or {}
                                        )
                                        updated_summaries = dict(existing_summaries)
                                        updated_summaries[
                                            str(parent_node_id_for_mat)
                                        ] = materialize_summary
                                        parent_task.request.materialize_summaries = (
                                            updated_summaries
                                        )

                                        parent_task.updated_at = datetime.now()
                                        task_manager.save_task(parent_task)

                                        await self.emit(
                                            user_id,
                                            workspace_id,
                                            {
                                                "type": "analysis_materialized",
                                                "task_type": task_type,
                                                "task_id": task_info.id,
                                                "parent_task_id": parent_task_id,
                                                "parent_node_id": parent_node_id_for_mat,
                                                "materialized_path": materialized_path,
                                                "timestamp": time.time(),
                                            },
                                        )
                                except Exception as side_err:
                                    # Don't fail the whole detach if the
                                    # materialise side-effect can't be
                                    # recorded — the workspace node is
                                    # already created.
                                    logger.warning(
                                        "Failed to record dispersion-detach materialisation: %s",
                                        side_err,
                                    )

                    except Exception as detach_err:
                        logger.error(
                            f"Failed to finalize detach task {task_info.id}: {detach_err}"
                        )
                        task_info.status = TaskStatus.FAILED
                        task_info.error = str(detach_err)
                        # We must send an update to reflect the failure
                        await self.emit(
                            user_id,
                            workspace_id,
                            {
                                "type": "task_changed",
                                "task": self._serialize_task(task_info),
                                "timestamp": time.time(),
                            },
                        )

                # Handle MATERIALIZE tasks (update parent analysis task request)
                elif task_type in [
                    "concordance_materialize",
                    "quotation_materialize",
                ]:
                    try:
                        from ..analysis.manager import get_task_manager

                        if not isinstance(result, dict):
                            raise ValueError(
                                "Materialize task result must be a dictionary payload"
                            )
                        data = result.get("result")
                        if not isinstance(data, dict):
                            raise ValueError(
                                "Materialize task result missing structured result payload"
                            )
                        parent_task_id = data.get("parent_task_id")
                        parent_node_id = data.get("parent_node_id")
                        materialized_path = data.get("materialized_path")
                        if not (
                            parent_task_id and parent_node_id and materialized_path
                        ):
                            raise ValueError(
                                "Materialize result missing parent_task_id, parent_node_id, or materialized_path"
                            )

                        task_manager = get_task_manager(user_id)
                        parent_task = task_manager.get_task(parent_task_id)
                        if parent_task is None:
                            raise RuntimeError(
                                f"Parent analysis task {parent_task_id} not found"
                            )

                        materialize_summary = {
                            "record_count": data.get("record_count"),
                            "unique_documents_with_hits": data.get(
                                "unique_documents_with_hits"
                            ),
                            "total_source_documents": data.get(
                                "total_source_documents"
                            ),
                        }

                        if task_type == "concordance_materialize":
                            existing = (
                                getattr(parent_task.request, "materialized_paths", None)
                                or {}
                            )
                            updated = dict(existing)
                            updated[str(parent_node_id)] = str(materialized_path)
                            parent_task.request.materialized_paths = updated

                            existing_summaries = (
                                getattr(
                                    parent_task.request,
                                    "materialize_summaries",
                                    None,
                                )
                                or {}
                            )
                            updated_summaries = dict(existing_summaries)
                            updated_summaries[str(parent_node_id)] = materialize_summary
                            parent_task.request.materialize_summaries = (
                                updated_summaries
                            )
                        else:
                            parent_task.request.materialized_path = str(
                                materialized_path
                            )
                            parent_task.request.materialize_summary = (
                                materialize_summary
                            )

                        parent_task.updated_at = datetime.now()
                        task_manager.save_task(parent_task)
                        result_persisted = True

                        await self.emit(
                            user_id,
                            workspace_id,
                            {
                                "type": "analysis_materialized",
                                "task_type": task_type,
                                "task_id": task_info.id,
                                "parent_task_id": parent_task_id,
                                "parent_node_id": parent_node_id,
                                "materialized_path": materialized_path,
                                "timestamp": time.time(),
                            },
                        )
                    except Exception as mat_err:
                        logger.error(
                            f"Failed to finalize materialize task {task_info.id}: {mat_err}"
                        )
                        task_info.status = TaskStatus.FAILED
                        task_info.error = str(mat_err)
                        await self.emit(
                            user_id,
                            workspace_id,
                            {
                                "type": "task_changed",
                                "task": self._serialize_task(task_info),
                                "timestamp": time.time(),
                            },
                        )

                # Handle ANALYSIS tasks (save to TaskManager)
                elif task_type in ANALYSIS_TASK_TYPES:
                    try:
                        # Save the analysis result
                        await self._save_analysis_result(
                            user_id, workspace_id, task_type, task_info, result
                        )
                        result_persisted = True
                    except Exception as save_error:
                        logger.error(
                            f"Failed to save {task_type} result for task {task_info.id}: {save_error}"
                        )

                        # Emit analysis save failure event
                        await self.emit(
                            user_id,
                            workspace_id,
                            {
                                "type": "analysis_save_failed",
                                "task_type": task_type,
                                "task_id": task_info.id,
                                "message": f"Failed to save result: {str(save_error)}",
                                "timestamp": time.time(),
                            },
                        )

            # Always emit task_changed for completion with accurate result_persisted flag
            await self.emit(
                user_id,
                workspace_id,
                {
                    "type": "task_changed",
                    "task": self._serialize_task(task_info),
                    "result_persisted": result_persisted,
                    "timestamp": time.time(),
                },
            )

        except asyncio.CancelledError:
            # The future was cancelled before it started running.
            if task_info.status != TaskStatus.CANCELLED:
                task_info.update_status()
                await self.emit(
                    user_id,
                    workspace_id,
                    {
                        "type": "task_changed",
                        "task": self._serialize_task(task_info),
                        "result_persisted": False,
                        "timestamp": time.time(),
                    },
                )
        except Exception as e:
            # If stop_task already set CANCELLED (process was killed), trust that
            # status and don't override it with a failure state.
            if task_info.status == TaskStatus.CANCELLED:
                logger.info(
                    "Task %s was stopped by user; ignoring worker exception: %s",
                    task_info.id,
                    e,
                )
                return

            logger.error(f"Error monitoring task completion for {task_info.id}: {e}")
            task_info.update_status()  # Update with error

            # Emit failure event
            await self.emit(
                user_id,
                workspace_id,
                {
                    "type": "task_changed",
                    "task": self._serialize_task(task_info),
                    "result_persisted": False,
                    "timestamp": time.time(),
                },
            )
        finally:
            self._cleanup_progress_queue(task_info.id)

    async def _save_analysis_result(
        self,
        user_id: str,
        workspace_id: str,
        task_type: str,
        task_info: TaskInfo,
        result: Any,
    ):
        """Persist worker analysis output into analysis task storage.

        Used by:
        - ``_monitor_task_completion`` for topic modeling, concordance, and token-frequency
          worker completions because background jobs need one lifecycle owner for submission,
          progress, cancellation, and artifact cleanup.
        Why:
        - Keeps worker-result serialization synchronized with TaskManager records.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        try:
            from ..analysis.manager import get_task_manager
            from ..analysis.results import GenericAnalysisResult

            task_manager = get_task_manager(user_id)
            task = task_manager.get_task(task_info.id)
            if task:
                task.complete(GenericAnalysisResult(result))
                task_manager.save_task(task)
                logger.info(
                    f"{task_type} result saved for task {task_info.id} via TaskManager"
                )
            else:
                logger.warning(f"Task {task_info.id} not found in TaskManager")

        except Exception as e:
            logger.error(
                f"Failed to save {task_type} result for task {task_info.id}: {e}"
            )
            raise  # Re-raise to mark task as failed

    async def submit_task(
        self,
        user_id: str,
        workspace_id: str,
        task_type: str,
        task_args: dict[str, Any],
        task_name: str | None = None,
        task_id: str | None = None,
    ) -> TaskInfo:
        """Submit a worker task, register tracking, and start monitors.

        Used by:
        - ``api.files`` for LDaCA downloads/imports because they need a backend boundary that
          validates inputs before delegating to workspace or worker state.
        - workspace lifecycle routes for workspace import jobs because they need a backend
          boundary that validates inputs before delegating to workspace or worker state.
        - analysis routes for analysis, materialize, detach, and tokenization jobs because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.
        Why:
        - Provides a single task lifecycle entry point with event emission.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """

        if task_type not in TASK_REGISTRY:
            raise ValueError(f"Unknown task type: {task_type}")

        task_func = TASK_REGISTRY[task_type]

        # Use pre-supplied ID when the caller needs the task ID to match a
        # filename (e.g. materialize tasks that key the cache parquet by the
        # child worker task ID); otherwise generate a fresh UUID.
        task_id = task_id or str(uuid.uuid4())

        # Submit task to worker pool with process-safe progress queue
        worker_pool = get_worker_pool()
        if not worker_pool.is_running:
            worker_pool.start()

        progress_queue = self._mp_manager.Queue()

        future = worker_pool.submit_task(
            task_func,
            user_id=user_id,
            workspace_id=workspace_id,
            **task_args,
            progress_callback=None,
            progress_queue=progress_queue,
        )

        raw_parent_task_id = task_args.get("parent_task_id")
        parent_task_id = (
            raw_parent_task_id if isinstance(raw_parent_task_id, str) else None
        )

        task_info = TaskInfo(
            id=task_id,
            future=future,
            status=TaskStatus.RUNNING,
            started_at=time.time(),
            task_type=task_type,
            name=task_name or task_type,
            user_id=user_id,
            workspace_id=workspace_id,
            parent_task_id=parent_task_id,
        )

        # Initialize progress tracking
        self._progress_store[task_id] = {
            "progress": 0.0,
            "message": "Task submitted",
            "updated_at": time.time(),
            "source": "real",
        }
        self._task_progress_queues[task_id] = progress_queue

        async with self._lock:
            self._tasks[task_id] = task_info

        # Start monitoring task completion in background
        asyncio.create_task(
            self._monitor_task_completion(task_info, user_id, workspace_id)
        )
        asyncio.create_task(
            self._consume_worker_progress(
                task_info, user_id, workspace_id, progress_queue
            )
        )

        # Emit task_changed event for initial submission
        logger.info(f"Emitting initial task_changed for task {task_info.id}")
        await self.emit(
            user_id,
            workspace_id,
            {
                "type": "task_changed",
                "task": self._serialize_task(task_info),
                "timestamp": time.time(),
            },
        )

        logger.info(
            f"Task {task_info.id} submitted successfully for user {user_id}, workspace {workspace_id}"
        )
        return task_info

    async def stop_task(self, task_id: str) -> bool:
        """Stop a running task, mark it cancelled, and keep the record for user dismissal.

        Used by:
        - ``api.tasks.cancel_task`` when the frontend Task Center cancels a job. Unlike
          ``clear_task``, the task record is *not* removed so the UI can display the cancelled
          state until the user explicitly clears it because they need a stable JSON contract
          shared by route handlers, generated clients, and tests.
        Strategy:
        - For tasks not yet picked up by a worker: ``future.cancel()`` works and
          the future is marked cancelled immediately.
        - For tasks already running: send SIGTERM to the worker PID (obtained via
          the progress-queue PID notification), then call ``future.cancel()`` as a
          belt-and-braces measure.  The worker process will exit, causing the
          future to raise ``BrokenProcessPool``; ``_monitor_task_completion``
          detects the pre-set CANCELLED status and does not override it.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        async with self._lock:
            task_info = self._tasks.get(task_id)
            if not task_info:
                return False

            if task_info.future.done():
                return False

            # Mark cancelled immediately so _monitor_task_completion can detect it
            task_info.status = TaskStatus.CANCELLED
            task_info.progress = -1.0
            task_info.progress_message = "Cancelled by user"
            task_info.finished_at = time.time()
            self._progress_store[task_id] = {
                "progress": -1.0,
                "message": "Cancelled by user",
                "updated_at": time.time(),
            }

            # Kill the worker process if we know its PID
            pid = task_info.worker_pid
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.info(
                        "Sent SIGTERM to worker PID %d for task %s", pid, task_id
                    )
                except (ProcessLookupError, OSError) as exc:
                    logger.debug("Could not send SIGTERM to PID %d: %s", pid, exc)

            # Also attempt future.cancel() for queued-but-not-yet-running tasks
            task_info.future.cancel()

            user_id = task_info.user_id
            workspace_id = task_info.workspace_id
            serialized = self._serialize_task(task_info)

        await self.emit(
            user_id,
            workspace_id,
            {
                "type": "task_changed",
                "task": serialized,
                "timestamp": time.time(),
            },
        )
        return True

    async def list(
        self, *, user_id: str | None = None, workspace_id: str | None = None
    ) -> builtins.list[dict[str, Any]]:
        """List tasks with normalized progress fields for API consumption.

        Used by:
        - ``api.tasks.list_tasks`` for the frontend Task Center list because they need a stable
          JSON contract shared by route handlers, generated clients, and tests.
        Why:
        - Keeps UI state queries independent of raw `Future` internals.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        async with self._lock:
            out: builtins.list[dict[str, Any]] = []
            for task_info in self._tasks.values():
                # Apply filters
                if user_id and task_info.user_id != user_id:
                    continue
                if workspace_id and task_info.workspace_id != workspace_id:
                    continue

                self._reconcile_task_progress(task_info)
                out.append(self._serialize_task(task_info))
            return out

    async def any_running(
        self,
        *,
        task_type: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
    ) -> bool:
        """Gate duplicate analysis submissions for routes that reuse in-flight jobs.

        Used by:
        - topic modeling and token-frequency submit routes before starting a job because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        async with self._lock:
            for task_info in self._tasks.values():
                # Apply filters
                if task_type and task_info.task_type != task_type:
                    continue
                if user_id and task_info.user_id != user_id:
                    continue
                if workspace_id and task_info.workspace_id != workspace_id:
                    continue

                task_info.update_status()
                if task_info.status == TaskStatus.RUNNING:
                    return True
            return False

    async def latest_by_type(
        self,
        task_type: str,
        *,
        user_id: str | None = None,
        workspace_id: str | None = None,
    ) -> TaskInfo | None:
        """Return the newest matching worker task for duplicate-submit responses.

        Used by:
        - topic modeling and token-frequency routes after ``any_running`` is true because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        async with self._lock:
            items = []
            for task_info in self._tasks.values():
                if task_info.task_type != task_type:
                    continue
                if user_id and task_info.user_id != user_id:
                    continue
                if workspace_id and task_info.workspace_id != workspace_id:
                    continue

                task_info.update_status()
                items.append(task_info)

            if not items:
                return None

            items.sort(key=lambda x: x.created_at, reverse=True)
            return items[0]

    async def get_task(self, task_id: str) -> TaskInfo | None:
        """Return one task with current status/progress reconciled.

        Used by:
        - analysis polling endpoints and sync helpers because they need a backend boundary that
          validates inputs before delegating to workspace or worker state.
        Why:
        - Ensures callers receive up-to-date state derived from future + progress
            store data.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        async with self._lock:
            task_info = self._tasks.get(task_id)
            if task_info:
                self._reconcile_task_progress(task_info)

            return task_info

    async def clear_task(self, task_id: str) -> bool:
        """Remove one worker task record and emit ``task_removed``.

        Used by:
        - analysis clear endpoints that already know the exact worker task ID because they need
          a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        async with self._lock:
            task_info = self._tasks.get(task_id)
            if not task_info:
                return False

            self._reconcile_task_progress(task_info)

            # Cancel the future if it's still running
            if not task_info.future.done():
                task_info.future.cancel()

            # Remove from tracking
            del self._tasks[task_id]
            self._progress_store.pop(task_id, None)
            self._cleanup_progress_queue(task_id)

            user_id = task_info.user_id
            workspace_id = task_info.workspace_id

        cleanup_worker_task_artifacts(task_info)
        await self.emit(
            user_id,
            workspace_id,
            {
                "type": "task_removed",
                "task_id": task_id,
                "workspace_id": workspace_id,
                "timestamp": time.time(),
            },
        )
        return True

    async def clear_task_tree(self, task_id: str) -> builtins.list[str]:
        """Remove a worker task plus child materialize/detach tasks.

        Used by:
        - ``api.tasks.clear_tasks`` and analysis cleanup helpers so parent task removal also
          clears worker-side child artifacts because they need a backend boundary that validates
          inputs before delegating to workspace or worker state.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        removed_tasks: builtins.list[tuple[str, str, str]] = []
        removed_task_infos: builtins.list[TaskInfo] = []
        async with self._lock:
            task_ids_to_remove = self._task_tree_ids_locked(task_id)
            for current_task_id in task_ids_to_remove:
                task_info = self._tasks.get(current_task_id)
                if task_info is None:
                    continue
                self._reconcile_task_progress(task_info)
                if not task_info.future.done():
                    task_info.future.cancel()
                removed_task_infos.append(task_info)
                removed_tasks.append(
                    (current_task_id, task_info.user_id, task_info.workspace_id)
                )
                del self._tasks[current_task_id]
                self._progress_store.pop(current_task_id, None)
                self._cleanup_progress_queue(current_task_id)

        for removed_task_info in removed_task_infos:
            cleanup_worker_task_artifacts(removed_task_info)

        timestamp = time.time()
        for removed_task_id, removed_user_id, removed_workspace_id in removed_tasks:
            await self.emit(
                removed_user_id,
                removed_workspace_id,
                {
                    "type": "task_removed",
                    "task_id": removed_task_id,
                    "workspace_id": removed_workspace_id,
                    "timestamp": timestamp,
                },
            )

        return [removed_task_id for removed_task_id, _, _ in removed_tasks]

    async def clear_tasks(
        self,
        task_type: str | None = None,
        *,
        user_id: str | None = None,
        workspace_id: str | None = None,
    ) -> int:
        """Bulk-remove worker tasks for workspace/file cleanup flows.

        Used by:
        - ``api.files`` when clearing download/import task records because they need a backend
          boundary that validates inputs before delegating to workspace or worker state.
        - ``WorkspaceManager`` workspace close/delete paths because background jobs need one
          lifecycle owner for submission, progress, cancellation, and artifact cleanup.

        Flow: resolve task records under locks, reconcile progress or future state, emit bounded
            subscriber events, and clean up queues or artifacts when a task reaches a terminal
            state.
        """
        removed_tasks: list[tuple[str, str, str]] = []
        removed_task_infos: list[TaskInfo] = []
        async with self._lock:
            task_ids_to_remove = []
            for task_id, task_info in self._tasks.items():
                # Apply filters
                if task_type and task_info.task_type != task_type:
                    continue
                if user_id and task_info.user_id != user_id:
                    continue
                if workspace_id and task_info.workspace_id != workspace_id:
                    continue

                self._reconcile_task_progress(task_info)
                if not task_info.future.done():
                    task_info.future.cancel()
                task_ids_to_remove.append(task_id)

            for task_id in task_ids_to_remove:
                task_info = self._tasks[task_id]
                removed_task_infos.append(task_info)
                removed_tasks.append(
                    (task_id, task_info.user_id, task_info.workspace_id)
                )
                del self._tasks[task_id]
                # Clean up progress store
                self._progress_store.pop(task_id, None)
                self._cleanup_progress_queue(task_id)

        for removed_task_info in removed_task_infos:
            cleanup_worker_task_artifacts(removed_task_info)

        for task_id, removed_user_id, removed_workspace_id in removed_tasks:
            await self.emit(
                removed_user_id,
                removed_workspace_id,
                {
                    "type": "task_removed",
                    "task_id": task_id,
                    "workspace_id": removed_workspace_id,
                    "timestamp": time.time(),
                },
            )

        return len(removed_tasks)
