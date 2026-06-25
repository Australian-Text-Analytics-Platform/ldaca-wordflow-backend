"""Worker task-manager lifecycle tests.

These tests isolate multiprocessing manager ownership from real worker-pool work so
Windows teardown regressions can be caught without starting child worker processes.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

import pytest

from ldaca_wordflow.core import worker_task_manager as worker_task_manager_module
from ldaca_wordflow.core.worker_task_manager import WorkerTaskManager
from ldaca_wordflow.core.workspace import WorkspaceManager


class _FakeProgressQueue:
    """Track queue close calls made by ``WorkerTaskManager.shutdown``."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeMpManager:
    """Small multiprocessing manager stand-in for lifecycle assertions."""

    def __init__(self) -> None:
        self.queues: list[_FakeProgressQueue] = []
        self.shutdown_called = False

    def Queue(self) -> _FakeProgressQueue:
        queue = _FakeProgressQueue()
        self.queues.append(queue)
        return queue

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeWorkerPool:
    """Capture worker submission arguments without starting child processes."""

    def __init__(self) -> None:
        self.is_running = True
        self.submitted_kwargs: dict[str, Any] | None = None

    def submit_task(self, _task_func: Any, **kwargs: Any) -> Future:
        self.submitted_kwargs = kwargs
        return Future()


def test_worker_task_manager_does_not_start_mp_manager_on_init(monkeypatch):
    """Constructing a worker task manager should not start multiprocessing state."""
    created: list[_FakeMpManager] = []

    def manager_factory() -> _FakeMpManager:
        manager = _FakeMpManager()
        created.append(manager)
        return manager

    monkeypatch.setattr(worker_task_manager_module.mp, "Manager", manager_factory)

    manager = WorkerTaskManager()

    assert created == []
    assert manager._mp_manager is None


@pytest.mark.asyncio
async def test_submit_task_creates_mp_manager_and_shutdown_releases_it(monkeypatch):
    """Submitting real worker work should lazily own and release the progress manager."""
    fake_mp_manager = _FakeMpManager()
    fake_worker_pool = _FakeWorkerPool()
    scheduled_coroutines: list[Any] = []

    def fake_task() -> None:
        return None

    def fake_create_task(coro: Any) -> object:
        scheduled_coroutines.append(coro)
        coro.close()
        return object()

    monkeypatch.setattr(
        worker_task_manager_module.mp,
        "Manager",
        lambda: fake_mp_manager,
    )
    monkeypatch.setattr(
        worker_task_manager_module,
        "get_worker_pool",
        lambda: fake_worker_pool,
    )
    monkeypatch.setattr(
        worker_task_manager_module.asyncio,
        "create_task",
        fake_create_task,
    )
    monkeypatch.setitem(
        worker_task_manager_module.TASK_REGISTRY,
        "fake_lifecycle_task",
        fake_task,
    )

    manager = WorkerTaskManager()
    task_info = await manager.submit_task(
        user_id="user-1",
        workspace_id="workspace-1",
        task_type="fake_lifecycle_task",
        task_args={"value": 1},
    )

    assert manager._mp_manager is fake_mp_manager
    assert fake_worker_pool.submitted_kwargs is not None
    assert (
        fake_worker_pool.submitted_kwargs["progress_queue"]
        is fake_mp_manager.queues[0]
    )
    assert manager._task_progress_queues[task_info.id] is fake_mp_manager.queues[0]
    assert len(scheduled_coroutines) == 2

    manager.shutdown()

    assert fake_mp_manager.queues[0].closed is True
    assert fake_mp_manager.shutdown_called is True
    assert manager._mp_manager is None
    assert manager._task_progress_queues == {}


def test_workspace_manager_shutdown_releases_cached_task_managers(monkeypatch):
    """Workspace shutdown should close and forget all cached worker managers."""
    fake_mp_manager = _FakeMpManager()
    monkeypatch.setattr(
        worker_task_manager_module.mp,
        "Manager",
        lambda: fake_mp_manager,
    )

    manager = WorkspaceManager()
    task_manager = manager.get_task_manager("user-1")
    queue = task_manager._get_mp_manager().Queue()
    task_manager._task_progress_queues["task-1"] = queue

    manager.shutdown_task_managers()

    assert fake_mp_manager.shutdown_called is True
    assert queue.closed is True
    assert manager._task_managers == {}
