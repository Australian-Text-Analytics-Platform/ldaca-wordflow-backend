"""Integration tests for task-id based analysis endpoints."""

import asyncio
from concurrent.futures import Future

import pytest
from ldaca_wordflow.analysis.manager import get_task_manager
from ldaca_wordflow.analysis.results import GenericAnalysisResult
from ldaca_wordflow.core.worker_task_manager import TaskInfo, WorkerTaskManager
from ldaca_wordflow.core.workspace import workspace_manager


@pytest.mark.asyncio
async def test_task_manager_endpoints_roundtrip(authenticated_client, workspace_id):
    """Task request/result endpoints serve data by explicit task id."""
    user_id = "test"
    manager = get_task_manager(user_id)

    task_id = manager.create_task(
        {
            "node_ids": ["node-1"],
            "node_columns": {"node-1": "text"},
            "tokenizer_model": "native:plain_words_en",
        },
        workspace_id=workspace_id,
    )
    manager.update_task(task_id, {"state": "successful", "data": {}})

    req_resp = await authenticated_client.get(
        f"/api/workspaces/{workspace_id}/analysis-tasks/{task_id}/request"
    )
    assert req_resp.status_code == 200

    # Task-specific result endpoints have analysis-specific processing;
    # result round-trip is covered by test_analysis_persistence.py with proper data.

    clear_resp = await authenticated_client.delete(f"/api/tasks/{task_id}")
    assert clear_resp.status_code == 200


@pytest.mark.asyncio
async def test_clear_analysis_only_task_emits_task_removed(
    authenticated_client, workspace_id
):
    """Clearing an analysis-only task should notify Task Center subscribers."""
    user_id = "test"
    analysis_manager = get_task_manager(user_id)
    task_id = analysis_manager.create_task(
        {"node_ids": ["node-1"]}, workspace_id=workspace_id
    )

    worker_manager = workspace_manager.get_task_manager(user_id)
    queue = await worker_manager.subscribe(user_id)

    try:
        clear_resp = await authenticated_client.delete(f"/api/tasks/{task_id}")
        assert clear_resp.status_code == 200

        event = await asyncio.wait_for(queue.get(), timeout=1)
        assert event["type"] == "task_removed"
        assert event["task_id"] == task_id
        assert event["workspace_id"] == workspace_id
        assert isinstance(event["timestamp"], float)
    finally:
        await worker_manager.unsubscribe(user_id, None, queue)


@pytest.mark.asyncio
async def test_worker_clear_task_emits_task_removed(workspace_id):
    """Clearing a worker task should remove the backend record and stream removal."""
    user_id = "test"
    manager = WorkerTaskManager()
    future = Future()
    future.set_result({"state": "successful"})
    task_info = TaskInfo(
        id="task-worker-clear",
        future=future,
        task_type="token_frequencies",
        user_id=user_id,
        workspace_id=workspace_id,
    )

    async with manager._lock:
        manager._tasks[task_info.id] = task_info

    queue = await manager.subscribe(user_id)

    try:
        cleared = await manager.clear_task(task_info.id)
        assert cleared is True

        event = await asyncio.wait_for(queue.get(), timeout=1)
        assert event["type"] == "task_removed"
        assert event["task_id"] == task_info.id
        assert event["workspace_id"] == workspace_id
        assert isinstance(event["timestamp"], float)
    finally:
        await manager.unsubscribe(user_id, None, queue)


@pytest.mark.asyncio
async def test_worker_clear_task_tree_removes_descendants(workspace_id):
    """Worker task tree removal should clear children before parents."""
    user_id = "test"
    manager = WorkerTaskManager()

    for task_id, parent_task_id in [
        ("task-parent", None),
        ("task-child", "task-parent"),
        ("task-grandchild", "task-child"),
    ]:
        future = Future()
        future.set_result({"state": "successful"})
        task_info = TaskInfo(
            id=task_id,
            future=future,
            task_type="concordance_materialize",
            user_id=user_id,
            workspace_id=workspace_id,
            parent_task_id=parent_task_id,
        )
        async with manager._lock:
            manager._tasks[task_info.id] = task_info

    queue = await manager.subscribe(user_id)

    try:
        cleared_ids = await manager.clear_task_tree("task-parent")

        assert cleared_ids == ["task-grandchild", "task-child", "task-parent"]
        assert await manager.get_task("task-parent") is None
        assert await manager.get_task("task-child") is None
        assert await manager.get_task("task-grandchild") is None

        events = [await asyncio.wait_for(queue.get(), timeout=1) for _ in range(3)]
        assert [event["task_id"] for event in events] == cleared_ids
        assert {event["type"] for event in events} == {"task_removed"}
    finally:
        await manager.unsubscribe(user_id, None, queue)


@pytest.mark.asyncio
async def test_clear_analysis_task_removes_child_worker_tasks(
    authenticated_client, workspace_id
):
    """Clearing an analysis parent should recursively remove worker children."""
    user_id = "test"
    analysis_manager = get_task_manager(user_id)
    parent_task_id = analysis_manager.create_task(
        {"node_ids": ["node-1"]}, workspace_id=workspace_id
    )

    child_task_id = "task-concordance-materialize-child"
    grandchild_task_id = "task-concordance-materialize-grandchild"
    analysis_manager.link_child_task(parent_task_id, child_task_id)

    worker_manager = workspace_manager.get_task_manager(user_id)
    for task_id, parent_id in [
        (child_task_id, parent_task_id),
        (grandchild_task_id, child_task_id),
    ]:
        future = Future()
        future.set_result({"state": "successful"})
        task_info = TaskInfo(
            id=task_id,
            future=future,
            task_type="concordance_materialize",
            user_id=user_id,
            workspace_id=workspace_id,
            parent_task_id=parent_id,
        )
        async with worker_manager._lock:
            worker_manager._tasks[task_info.id] = task_info

    queue = await worker_manager.subscribe(user_id)

    try:
        clear_resp = await authenticated_client.delete(f"/api/tasks/{parent_task_id}")
        assert clear_resp.status_code == 200
        data = clear_resp.json()["data"]
        assert data["cleared_task_ids"] == [
            grandchild_task_id,
            child_task_id,
            parent_task_id,
        ]

        assert analysis_manager.get_task(parent_task_id) is None
        assert await worker_manager.get_task(child_task_id) is None
        assert await worker_manager.get_task(grandchild_task_id) is None

        events = [await asyncio.wait_for(queue.get(), timeout=1) for _ in range(3)]
        assert [event["task_id"] for event in events] == [
            grandchild_task_id,
            child_task_id,
            parent_task_id,
        ]
        assert {event["type"] for event in events} == {"task_removed"}
    finally:
        await worker_manager.unsubscribe(user_id, None, queue)


@pytest.mark.asyncio
async def test_clear_analysis_task_deletes_owned_artifacts_only(
    authenticated_client, workspace_id
):
    """Clearing a task should remove transient artifacts but preserve workspace data."""
    user_id = "test"
    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
    assert workspace_dir is not None

    artifact_dir = workspace_dir / "data" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    request_artifact = artifact_dir / "materialized_child.parquet"
    result_artifact = artifact_dir / "token_frequency.parquet"
    result_artifact_dir = artifact_dir / "topic-model.artifacts"
    durable_data = workspace_dir / "data" / "added_to_workspace.parquet"

    request_artifact.write_text("request", encoding="utf-8")
    result_artifact.write_text("result", encoding="utf-8")
    result_artifact_dir.mkdir()
    (result_artifact_dir / "model.bin").write_text("model", encoding="utf-8")
    durable_data.write_text("workspace", encoding="utf-8")

    analysis_manager = get_task_manager(user_id)
    task_id = analysis_manager.create_task(
        {"materialized_paths": {"node-1": str(request_artifact)}},
        workspace_id=workspace_id,
    )
    task = analysis_manager.get_task(task_id)
    assert task is not None
    task.complete(
        GenericAnalysisResult(
            {
                "artifacts": {
                    "token_parquet_path": str(result_artifact),
                    "model_artifact_path": str(result_artifact_dir),
                    "promoted_parquet_path": str(durable_data),
                }
            }
        )
    )
    analysis_manager.save_task(task)

    clear_resp = await authenticated_client.delete(f"/api/tasks/{task_id}")

    assert clear_resp.status_code == 200
    assert not request_artifact.exists()
    assert not result_artifact.exists()
    assert not result_artifact_dir.exists()
    assert durable_data.exists()


@pytest.mark.asyncio
async def test_cancel_worker_task_uses_resource_route(authenticated_client, workspace_id):
    """Stopping a task should address the task id in the URL, not a query string."""

    user_id = "test"
    manager = workspace_manager.get_task_manager(user_id)
    future = Future()
    task_info = TaskInfo(
        id="task-worker-cancel",
        future=future,
        task_type="token_frequencies",
        user_id=user_id,
        workspace_id=workspace_id,
    )

    async with manager._lock:
        manager._tasks[task_info.id] = task_info

    response = await authenticated_client.post(f"/api/tasks/{task_info.id}/cancel")

    assert response.status_code == 200
    assert response.json()["data"]["stopped"] is True


@pytest.mark.asyncio
async def test_worker_clear_task_tree_deletes_owned_artifacts_only(workspace_id):
    """Worker task cleanup should own transient artifacts without touching data nodes."""
    user_id = "test"
    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
    assert workspace_dir is not None

    artifact_dir = workspace_dir / "data" / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    worker_artifact = artifact_dir / "worker-output.parquet"
    durable_data = workspace_dir / "data" / "worker-added-to-workspace.parquet"
    worker_artifact.write_text("worker", encoding="utf-8")
    durable_data.write_text("workspace", encoding="utf-8")

    manager = WorkerTaskManager()
    future = Future()
    future.set_result(
        {
            "state": "successful",
            "result": {
                "artifacts": {
                    "worker_parquet_path": str(worker_artifact),
                    "promoted_parquet_path": str(durable_data),
                }
            },
        }
    )
    task_info = TaskInfo(
        id="task-worker-artifact-cleanup",
        future=future,
        task_type="token_frequencies",
        user_id=user_id,
        workspace_id=workspace_id,
    )

    async with manager._lock:
        manager._tasks[task_info.id] = task_info

    cleared_ids = await manager.clear_task_tree(task_info.id)

    assert cleared_ids == [task_info.id]
    assert not worker_artifact.exists()
    assert durable_data.exists()


@pytest.mark.asyncio
async def test_task_manager_saves_result_by_task_id(workspace_id):
    """WorkerTaskManager should persist results by task_id in TaskManager."""
    user_id = "test"
    manager = get_task_manager(user_id)
    task_id = manager.create_task({"node_ids": ["node-1"]}, workspace_id=workspace_id)

    task_info = TaskInfo(id=task_id, future=Future())
    manager_instance = WorkerTaskManager()

    await manager_instance._save_analysis_result(
        user_id,
        workspace_id,
        "token_frequencies",
        task_info,
        {"state": "successful", "data": {}},
    )

    task = manager.get_task(task_id)
    assert task is not None
    assert task.result is not None
    assert task.result is not None
