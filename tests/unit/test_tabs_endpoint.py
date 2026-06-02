"""Tests for the workspace analysis-tab sidecar endpoints.

Drive both handlers directly with a monkey-patched ``workspace_manager`` that
maps workspace_id → a tmp_path, mirroring ``test_ui_state_endpoint.py``. The
endpoints load/save a JSON sidecar at ``<workspace_dir>/tabs.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ldaca_wordflow.api.workspaces import tabs as tabs_api
from ldaca_wordflow.core.exceptions import WorkspaceNotFoundError


def _state(payload: dict | None = None) -> tabs_api.WorkspaceTabsState:
    return tabs_api.WorkspaceTabsState.model_validate(payload or {})


class _FakeManager:
    def __init__(self, workspace_id: str, workspace_dir: Path):
        self.workspace_id = workspace_id
        self.workspace_dir = workspace_dir

    def get_workspace_dir(self, _user_id: str, workspace_id: str):
        if workspace_id != self.workspace_id:
            return None
        return self.workspace_dir


@pytest.fixture
def fake_workspace(tmp_path, monkeypatch):
    manager = _FakeManager(workspace_id="ws1", workspace_dir=tmp_path)
    monkeypatch.setattr(tabs_api, "workspace_manager", manager)
    return manager


def _sample_group() -> dict:
    return {
        "groups": {
            "concordance": {
                "tabs": [
                    {"tab_id": "t1", "task_id": "task-1", "title": "First"},
                    {"tab_id": "t2", "task_id": None, "title": "Untitled"},
                ],
                "active_tab_id": "t1",
            }
        }
    }


@pytest.mark.asyncio
async def test_get_returns_default_when_file_missing(fake_workspace):
    result = await tabs_api.get_workspace_tabs(
        workspace_id="ws1", current_user={"id": "u"}
    )
    assert result == tabs_api.WorkspaceTabsState()


@pytest.mark.asyncio
async def test_get_returns_parsed_contents_when_present(fake_workspace):
    payload = _sample_group()
    (fake_workspace.workspace_dir / "tabs.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = await tabs_api.get_workspace_tabs(
        workspace_id="ws1", current_user={"id": "u"}
    )
    assert result.model_dump() == payload


@pytest.mark.asyncio
async def test_get_404s_on_unknown_workspace(fake_workspace):
    with pytest.raises(WorkspaceNotFoundError):
        await tabs_api.get_workspace_tabs(
            workspace_id="does-not-exist", current_user={"id": "u"}
        )


@pytest.mark.asyncio
async def test_get_swallows_corrupt_json(fake_workspace):
    (fake_workspace.workspace_dir / "tabs.json").write_text(
        "{not-valid", encoding="utf-8"
    )
    result = await tabs_api.get_workspace_tabs(
        workspace_id="ws1", current_user={"id": "u"}
    )
    assert result == tabs_api.WorkspaceTabsState()


@pytest.mark.asyncio
async def test_get_swallows_non_object_json(fake_workspace):
    (fake_workspace.workspace_dir / "tabs.json").write_text(
        '["not", "an", "object"]', encoding="utf-8"
    )
    result = await tabs_api.get_workspace_tabs(
        workspace_id="ws1", current_user={"id": "u"}
    )
    assert result == tabs_api.WorkspaceTabsState()


@pytest.mark.asyncio
async def test_put_writes_file_and_echoes_payload(fake_workspace):
    payload = _sample_group()
    result = await tabs_api.put_workspace_tabs(
        workspace_id="ws1", payload=_state(payload), current_user={"id": "u"}
    )
    assert result.model_dump() == payload
    written = (fake_workspace.workspace_dir / "tabs.json").read_text(encoding="utf-8")
    assert json.loads(written) == payload


@pytest.mark.asyncio
async def test_put_replaces_existing_contents_not_merges(fake_workspace):
    (fake_workspace.workspace_dir / "tabs.json").write_text(
        json.dumps(_sample_group()), encoding="utf-8"
    )
    new_payload = {
        "groups": {
            "token_frequencies": {
                "tabs": [{"tab_id": "x", "task_id": None, "title": "New"}],
                "active_tab_id": "x",
            }
        }
    }
    await tabs_api.put_workspace_tabs(
        workspace_id="ws1", payload=_state(new_payload), current_user={"id": "u"}
    )
    written = (fake_workspace.workspace_dir / "tabs.json").read_text(encoding="utf-8")
    assert json.loads(written) == new_payload


@pytest.mark.asyncio
async def test_put_404s_on_unknown_workspace(fake_workspace):
    with pytest.raises(WorkspaceNotFoundError):
        await tabs_api.put_workspace_tabs(
            workspace_id="does-not-exist",
            payload=_state(),
            current_user={"id": "u"},
        )
