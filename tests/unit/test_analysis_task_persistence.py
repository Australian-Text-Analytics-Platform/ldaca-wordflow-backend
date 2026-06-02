"""Analysis task records survive a workspace unload/reload cycle.

The analysis tab system keeps task ids on persisted tabs (``tabs.json``). For a
tab to re-show its result after the workspace is unloaded and loaded again (or
after a server restart), the underlying ``AnalysisTask`` records must be
snapshotted to disk on unload and rehydrated on load. These tests exercise that
round-trip through the real `WorkspaceManager` lifecycle hooks plus the
`analysis.persistence` helpers, mirroring the bootstrap pattern used by
``test_set_current_task_eviction.py``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import pytest
from ldaca_wordflow.analysis import persistence as analysis_persistence
from ldaca_wordflow.analysis.manager import TaskManager
from ldaca_wordflow.analysis.models import (
    AnalysisStatus,
    AnalysisTask,
    BaseAnalysisRequest,
)
from ldaca_wordflow.analysis.results import GenericAnalysisResult
from ldaca_wordflow.core import utils as core_utils
from ldaca_wordflow.core import workspace as workspace_module
from ldaca_wordflow.core.workspace import WorkspaceManager


def _bootstrap_workspace(
    manager: WorkspaceManager, user_id: str, name: str
) -> tuple[str, Path]:
    """Create a workspace dir and register it as current for the user."""
    from docworkspace import Workspace

    workspace_id = str(uuid.uuid4())
    target_dir = manager._resolve_workspace_dir(
        user_id=user_id, workspace_id=workspace_id, workspace_name=name
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "data").mkdir(parents=True, exist_ok=True)
    ws = Workspace(name=name, ws_root_dir=target_dir)
    ws.id = workspace_id
    ws.modified_at = datetime.now().isoformat()
    ws.save(target_dir)
    manager._set_cached_path(user_id, workspace_id, target_dir)
    manager.set_current_workspace(user_id, workspace_id)
    return workspace_id, target_dir


@pytest.fixture
def isolated_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(core_utils.settings, "multi_user", True, raising=False)
    monkeypatch.setattr(core_utils.settings, "user_data_folder", "users", raising=False)
    monkeypatch.setattr(core_utils.settings, "data_root", tmp_path, raising=False)

    manager = WorkspaceManager()
    monkeypatch.setattr(workspace_module, "workspace_manager", manager)
    return manager


@pytest.fixture
def reset_task_store(monkeypatch):
    """Give each test a clean per-user task store."""
    from ldaca_wordflow.analysis import manager as analysis_manager_module

    monkeypatch.setattr(analysis_manager_module, "_TASK_MANAGER_STORE", {})
    yield


def _save_concordance_task(
    tm: TaskManager, user_id: str, workspace_id: str, tab_id: str
) -> str:
    """Persist a completed concordance-style task pinned to ``tab_id``."""
    task_id = str(uuid.uuid4())
    request = BaseAnalysisRequest.model_validate(
        {"node_ids": ["node-1"], "search_word": "hello"}
    )
    tm.save_task(
        AnalysisTask(
            task_id=task_id,
            user_id=user_id,
            workspace_id=workspace_id,
            request=request,
            status=AnalysisStatus.COMPLETED,
            result=GenericAnalysisResult({"ready": True}),
        )
    )
    tm.set_current_task(tab_id, task_id)
    return task_id


def test_analysis_task_survives_unload_and_reload(
    isolated_manager, reset_task_store
):
    user_id = "tab_user"
    workspace_id, ws_dir = _bootstrap_workspace(isolated_manager, user_id, "ws")

    tm = TaskManager(user_id)
    tab_id = str(uuid.uuid4())
    task_id = _save_concordance_task(tm, user_id, workspace_id, tab_id)

    # Unload persists the snapshot to <workspace_dir>/analysis_tasks.json and
    # then wipes the in-memory store.
    assert isolated_manager.unload_workspace(user_id, save=True) is True
    assert (ws_dir / "analysis_tasks.json").exists()
    assert TaskManager(user_id).get_task(task_id) is None

    # Reload rehydrates the task record + the tab's current-task pointer.
    assert isolated_manager.set_current_workspace(user_id, workspace_id) is True

    restored = TaskManager(user_id)
    task = restored.get_task(task_id)
    assert task is not None
    assert task.workspace_id == workspace_id
    assert task.status == AnalysisStatus.COMPLETED
    assert task.request.model_dump().get("search_word") == "hello"
    assert restored.get_current_task_ids(tab_id) == [task_id]


def test_no_sidecar_written_when_workspace_has_no_tasks(
    isolated_manager, reset_task_store
):
    user_id = "empty_user"
    workspace_id, ws_dir = _bootstrap_workspace(isolated_manager, user_id, "ws")

    assert isolated_manager.unload_workspace(user_id, save=True) is True
    assert not (ws_dir / "analysis_tasks.json").exists()

    # Reloading an empty workspace is a no-op and does not error.
    assert isolated_manager.set_current_workspace(user_id, workspace_id) is True


def test_persistence_helpers_round_trip_directly(tmp_path, reset_task_store):
    """The persistence helpers serialize/restore without the workspace manager."""
    user_id = "direct_user"
    workspace_id = "ws-direct"
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()

    tm = TaskManager(user_id)
    tab_id = str(uuid.uuid4())
    task_id = _save_concordance_task(tm, user_id, workspace_id, tab_id)

    analysis_persistence.save_workspace_analysis_tasks(user_id, workspace_id, ws_dir)
    assert (ws_dir / "analysis_tasks.json").exists()

    # Wipe and restore.
    from ldaca_wordflow.analysis import manager as analysis_manager_module

    analysis_manager_module._TASK_MANAGER_STORE = {}
    analysis_persistence.load_workspace_analysis_tasks(user_id, workspace_id, ws_dir)

    restored = TaskManager(user_id)
    assert restored.get_task(task_id) is not None
    assert restored.get_current_task_ids(tab_id) == [task_id]
