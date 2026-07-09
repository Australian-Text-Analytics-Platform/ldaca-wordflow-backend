import polars as pl
import pytest
from ldaca_wordflow.api.workspaces import utils as workspace_utils


class _DummyNode:
    def __init__(self, frame: pl.DataFrame):
        self.data = frame.lazy()
        self.name = "dummy"


class _DummyWorkspace:
    def __init__(self, persist_calls: dict[str, int], nodes=None):
        self.id = "00000000-0000-0000-0000-000000000201"
        self.name = "ws"
        self._persist_calls = persist_calls
        self.nodes = nodes or {}

    def set_metadata(self, *_args, **_kwargs):
        return None

    def save(self, *_args, **_kwargs):
        self._persist_calls["count"] += 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_preview_returns_masked_values(authenticated_client, monkeypatch):
    frame = pl.DataFrame({"Body": ["Invoice 123", "Order 987"]})
    node = _DummyNode(frame)
    workspace_id = "00000000-0000-0000-0000-000000000201"
    dummy_ws = _DummyWorkspace({"count": 0}, nodes={"node-123": node})

    monkeypatch.setattr(
        workspace_utils.workspace_manager,
        "get_current_workspace_id",
        lambda user_id: workspace_id,
    )
    monkeypatch.setattr(
        workspace_utils.workspace_manager,
        "get_current_workspace",
        lambda _user_id: dummy_ws,
    )

    response = await authenticated_client.post(
        f"/api/workspaces/{workspace_id}/nodes/node-123/replace/preview",
        json={
            "source_column": "Body",
            "pattern": r"\d+",
            "replacement": "#",
            "output_column_name": "Body_masked",
            "preview_limit": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"][0]["Body_masked"] == "Invoice #"
    assert payload["data"][1]["Body_masked"] == "Order #"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_apply_mutates_node_data(authenticated_client, monkeypatch):
    frame = pl.DataFrame({"Body": ["Invoice 123", "Order 987"]})
    node = _DummyNode(frame)
    workspace_id = "00000000-0000-0000-0000-000000000201"
    persist_calls = {"count": 0}
    dummy_ws = _DummyWorkspace(persist_calls, nodes={"node-123": node})

    monkeypatch.setattr(
        workspace_utils.workspace_manager,
        "get_current_workspace_id",
        lambda user_id: workspace_id,
    )
    monkeypatch.setattr(
        workspace_utils.workspace_manager,
        "get_current_workspace",
        lambda user_id: dummy_ws,
    )
    monkeypatch.setattr(
        workspace_utils.workspace_manager,
        "_resolve_workspace_dir",
        lambda user_id, workspace_id, workspace_name: "/tmp/ws",
    )
    monkeypatch.setattr(
        workspace_utils.workspace_manager,
        "_attach_workspace_dir",
        lambda workspace, path: None,
    )
    monkeypatch.setattr(
        workspace_utils.workspace_manager,
        "_set_cached_path",
        lambda user_id, workspace_id, path: None,
    )

    response = await authenticated_client.post(
        f"/api/workspaces/{workspace_id}/nodes/node-123/replace",
        json={
            "source_column": "Body",
            "pattern": r"\d+",
            "replacement": "#",
            "output_column_name": "Body_masked",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["column_name"] == "Body_masked"
    assert payload["state"] == "successful"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_preview_returns_raw_polars_error_for_missing_source_column(
    authenticated_client, monkeypatch
):
    frame = pl.DataFrame({"Body": ["Invoice 123", "Order 987"]})
    node = _DummyNode(frame)
    workspace_id = "00000000-0000-0000-0000-000000000201"
    dummy_ws = _DummyWorkspace({"count": 0}, nodes={"node-123": node})

    monkeypatch.setattr(
        workspace_utils.workspace_manager,
        "get_current_workspace_id",
        lambda user_id: workspace_id,
    )
    monkeypatch.setattr(
        workspace_utils.workspace_manager,
        "get_current_workspace",
        lambda _user_id: dummy_ws,
    )

    response = await authenticated_client.post(
        f"/api/workspaces/{workspace_id}/nodes/node-123/replace/preview",
        json={
            "source_column": "Missing",
            "pattern": r"\d+",
            "replacement": "#",
            "output_column_name": "Body_masked",
        },
    )

    assert response.status_code == 400
    detail = response.json()["message"]
    assert 'unable to find column "Missing"' in detail
    assert "Unknown column" not in detail


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_apply_returns_raw_polars_error_for_non_string_column(
    authenticated_client, monkeypatch
):
    frame = pl.DataFrame({"num": [1, 2]})
    node = _DummyNode(frame)
    workspace_id = "00000000-0000-0000-0000-000000000201"
    persist_calls = {"count": 0}
    dummy_ws = _DummyWorkspace(persist_calls, nodes={"node-123": node})

    monkeypatch.setattr(
        workspace_utils.workspace_manager,
        "get_current_workspace_id",
        lambda user_id: workspace_id,
    )
    monkeypatch.setattr(
        workspace_utils.workspace_manager,
        "get_current_workspace",
        lambda _user_id: dummy_ws,
    )

    response = await authenticated_client.post(
        f"/api/workspaces/{workspace_id}/nodes/node-123/replace",
        json={
            "source_column": "num",
            "pattern": "1",
            "replacement": "2",
            "output_column_name": "num_masked",
        },
    )

    assert response.status_code == 400
    detail = response.json()["message"]
    assert "expected String type, got: i64" in detail
    assert "must be a string column" not in detail
    collected = node.data.collect()
    assert isinstance(collected, pl.DataFrame)
    assert collected.columns == ["num"]
    assert collected.get_column("num").to_list() == [1, 2]
    assert "num_masked" not in collected.columns
    assert persist_calls["count"] == 0
