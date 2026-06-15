"""Unit tests for the slice node endpoint."""

from __future__ import annotations

import polars as pl
import pytest
from ldaca_wordflow.api.workspaces import nodes_slice as nodes_api
from ldaca_wordflow.api.workspaces import utils as workspace_utils
from ldaca_wordflow.models import SliceRequest


class DummyNode:
    """Lightweight node stub used for slice endpoint tests."""

    def __init__(
        self,
        node_id: str | None = None,
        data: pl.LazyFrame | None = None,
        name: str | None = None,
        workspace: object | None = None,
        operation: str | None = None,
        parents: list["DummyNode"] | None = None,
        **_kwargs,
    ) -> None:
        manager = getattr(workspace, "_manager", None)
        generated_index = len(getattr(manager, "add_calls", []))
        generated_id = node_id or f"generated_{generated_index}"
        self.id = generated_id
        self.node_id = generated_id
        self.name = name
        self.data = data
        self.operation = operation
        self.parents: list[DummyNode] = parents or []


class DummyWorkspace:
    def __init__(self, nodes: dict[str, DummyNode], manager: "FakeWorkspaceManager"):
        self.nodes = nodes
        self._manager = manager
        self.id = manager.workspace_id
        self.name = "dummy"

    def add_node(self, node: DummyNode):
        self.nodes[node.id] = node
        self._manager.add_calls.append({"node": node})

    def place_node_after_parent(self, _node):
        return None

    def save(self, _target_dir):
        return None


class FakeWorkspaceManager:
    """Fake workspace manager that captures slice operations."""

    def __init__(self, nodes: dict[str, DummyNode]) -> None:
        self.nodes = nodes
        self.workspace_id = "ws1"
        self.workspace = DummyWorkspace(self.nodes, self)
        self.add_calls: list[dict[str, object]] = []

    def get_current_workspace(self, _user_id: str):
        return self.workspace

    def get_current_workspace_id(self, _user_id: str):
        return self.workspace_id

    def save_workspace(self, _user_id: str, _workspace_id: str) -> None:
        pass

    def _resolve_workspace_dir(
        self,
        user_id: str,
        workspace_id: str,
        workspace_name: str,
    ):
        return "workspace_path"

    def _attach_workspace_dir(self, workspace, target_dir):
        return None

    def _set_cached_path(self, user_id: str, workspace_id: str, target_dir):
        return None

    def set_current_workspace(self, user_id: str, workspace_id: str):
        self.workspace_id = workspace_id
        return True


@pytest.fixture
def fake_workspace_manager(monkeypatch: pytest.MonkeyPatch):
    df = pl.DataFrame(
        {
            "value": [1, 2, 3, 4, 5],
            "label": ["a", "b", "c", "d", "e"],
        }
    )
    original_node = DummyNode("node_base", df.lazy(), "base_node")
    manager = FakeWorkspaceManager({"node_base": original_node})
    monkeypatch.setattr(workspace_utils, "Node", DummyNode)
    monkeypatch.setattr(workspace_utils, "workspace_manager", manager)
    return manager


@pytest.mark.asyncio
async def test_slice_node_with_offset_and_length(fake_workspace_manager):
    request = SliceRequest(offset=1, length=2, new_node_name="subset_rows")

    result = await nodes_api.slice_node(
        "node_base", request, current_user={"id": "user"}
    )

    assert result["node_name"] == "subset_rows"
    assert result["node_id"] == "generated_0"

    assert len(fake_workspace_manager.add_calls) == 1
    created = fake_workspace_manager.add_calls[0]["node"]
    collected = created.data.collect()
    assert collected.shape == (2, 2)
    assert collected.get_column("value").to_list() == [2, 3]
    assert created.parents == [fake_workspace_manager.nodes["node_base"]]
    assert created.operation == "slice(base_node, offset=1, length=2)"


@pytest.mark.asyncio
async def test_slice_node_without_length_uses_tail(fake_workspace_manager):
    request = SliceRequest(offset=3)

    result = await nodes_api.slice_node(
        "node_base", request, current_user={"id": "user"}
    )

    assert result["node_name"] == "base_node_sliced"
    assert result["node_id"] == "generated_0"

    assert len(fake_workspace_manager.add_calls) == 1
    created = fake_workspace_manager.add_calls[0]["node"]
    collected = created.data.collect()
    assert collected.shape == (2, 2)
    assert collected.get_column("value").to_list() == [4, 5]
    assert created.operation == "slice(base_node, offset=3)"


@pytest.mark.asyncio
async def test_slice_preview_respects_offset_and_length(fake_workspace_manager):
    request = SliceRequest(offset=1, length=3)

    preview = await nodes_api.slice_preview(
        "node_base",
        request,
        page=1,
        page_size=2,
        current_user={"id": "user"},
    )

    assert preview.columns == ["value", "label"]
    assert preview.pagination.total_rows == 3
    assert preview.pagination.page == 1
    assert len(preview.data) == 2
    assert [row["value"] for row in preview.data] == [2, 3]

    preview_page_two = await nodes_api.slice_preview(
        "node_base",
        request,
        page=2,
        page_size=2,
        current_user={"id": "user"},
    )

    assert preview_page_two.pagination.page == 2
    assert len(preview_page_two.data) == 1
    assert preview_page_two.data[0]["value"] == 4
    assert preview_page_two.data[0]["value"] == 4


@pytest.mark.asyncio
async def test_random_sample_node_uses_fraction_and_seed(fake_workspace_manager):
    request = SliceRequest(
        mode="random_sample",
        sample_size=0.4,
        random_seed=7,
        new_node_name="sample_rows",
    )

    result = await nodes_api.slice_node(
        "node_base", request, current_user={"id": "user"}
    )

    assert result["node_name"] == "sample_rows"
    assert result["node_id"] == "generated_0"

    assert len(fake_workspace_manager.add_calls) == 1
    created = fake_workspace_manager.add_calls[0]["node"]
    collected = created.data.collect()
    assert collected.shape == (2, 2)
    assert collected.get_column("value").to_list() == [5, 1]
    assert collected.get_column("label").to_list() == ["e", "a"]
    assert created.parents == [fake_workspace_manager.nodes["node_base"]]
    assert created.operation == "sample(base_node, fraction=0.4, seed=7)"


@pytest.mark.asyncio
async def test_random_sample_preview_respects_seed(fake_workspace_manager):
    request = SliceRequest(mode="random_sample", sample_size=0.4, random_seed=7)

    preview = await nodes_api.slice_preview(
        "node_base",
        request,
        page=1,
        page_size=10,
        current_user={"id": "user"},
    )

    assert preview.columns == ["value", "label"]
    assert preview.pagination.total_rows == 2
    assert preview.pagination.page == 1
    assert len(preview.data) == 2
    assert [row["value"] for row in preview.data] == [5, 1]
    assert [row["label"] for row in preview.data] == ["e", "a"]


@pytest.mark.asyncio
async def test_random_sample_node_uses_n_for_integer(fake_workspace_manager):
    """When sample_size >= 1, it should use n= instead of fraction=."""
    request = SliceRequest(
        mode="random_sample",
        sample_size=3,
        random_seed=7,
        new_node_name="sample_n_rows",
    )

    result = await nodes_api.slice_node(
        "node_base", request, current_user={"id": "user"}
    )

    assert result["node_name"] == "sample_n_rows"
    assert result["node_id"] == "generated_0"

    assert len(fake_workspace_manager.add_calls) == 1
    created = fake_workspace_manager.add_calls[0]["node"]
    collected = created.data.collect()
    assert collected.shape == (3, 2)
    assert created.operation == "sample(base_node, n=3, seed=7)"
    assert created.operation == "sample(base_node, n=3, seed=7)"
