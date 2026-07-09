"""Bug 2 regression: child nodes from filter / slice / sample / concat /
join / clone / expression-apply must inherit the parent's ``tokenization``
registry when the registered source column survives the operation. Pre-fix,
every Node()-creating endpoint in ``api.workspaces.nodes`` constructed the new
node WITHOUT passing ``tokenization=``, so downstream tools using
``Node.find_tokenization_column`` refused to run with "re-run Tokenise first".
"""

from __future__ import annotations

import uuid

import polars as pl
import pytest
from docworkspace.workspace.core import Workspace
from ldaca_wordflow.api.workspaces import utils as workspace_utils
from ldaca_wordflow.api.workspaces.nodes_concat import concat_nodes
from ldaca_wordflow.api.workspaces.nodes_crud import clone_node
from ldaca_wordflow.api.workspaces.nodes_expression import polars_expression_apply
from ldaca_wordflow.api.workspaces.nodes_filter import filter_node
from ldaca_wordflow.api.workspaces.nodes_join import join_nodes
from ldaca_wordflow.api.workspaces.nodes_slice import slice_node


class NodesApiMock:
    clone_node = staticmethod(clone_node)
    filter_node = staticmethod(filter_node)
    slice_node = staticmethod(slice_node)
    concat_nodes = staticmethod(concat_nodes)
    join_nodes = staticmethod(join_nodes)
    polars_expression_apply = staticmethod(polars_expression_apply)


nodes_api = NodesApiMock
from ldaca_wordflow.models.nodes import (
    ConcatRequest,
    FilterCondition,
    FilterRequest,
    SliceRequest,
)
from ldaca_wordflow.models.polars_expression import (
    PolarsExpressionContext,
    PolarsExpressionItem,
    PolarsExpressionRequest,
)

from docworkspace import Node, TokenizationMeta

TOKENIZATION_COLUMN = "tokenization.text.lindera:ja-ipadic"
TOKENIZATION_META: TokenizationMeta = {
    "column_name": TOKENIZATION_COLUMN,
    "model": "lindera:ja-ipadic",
    "language": "ja",
    "params": {"lowercase": True, "remove_punct": True},
}


def _make_node_with_tokens(name: str = "root") -> Node:
    """Create a Node with a registered token spec.

    Token arrays are hydrated dynamically in analysis paths; propagation only
    depends on whether the source column survives the child operation.
    """
    df = pl.DataFrame({"text": ["今日は", "良い天気"], "id": [1, 2]}).lazy()
    node = Node(
        data=df,
        name=name,
        tokenization={"text": TOKENIZATION_META},
    )
    return node


class _FakeManager:
    """Workspace-manager stub with a real Workspace and real Nodes."""

    def __init__(self, *nodes: Node) -> None:
        self.workspace = Workspace(name="test_ws")
        for node in nodes:
            node.workspace = self.workspace
            self.workspace.add_node(node)
        self.workspace_id = self.workspace.id

    def get_current_workspace(self, _user_id: str):
        return self.workspace

    def get_current_workspace_id(self, _user_id: str):
        return self.workspace_id

    def load_workspace(self, _user_id: str, _workspace_id: str):
        return self.workspace

    def _resolve_workspace_dir(self, *_args, **_kwargs):  # pragma: no cover
        return "/tmp/dummy"

    def _attach_workspace_dir(self, *_args, **_kwargs):  # pragma: no cover
        return None

    def _set_cached_path(self, *_args, **_kwargs):  # pragma: no cover
        return None

    def save_workspace(self, *_args, **_kwargs):  # pragma: no cover
        return None


@pytest.fixture
def single_parent(monkeypatch: pytest.MonkeyPatch):
    """Workspace with one parent node that already carries tokenization."""
    parent = _make_node_with_tokens("parent")
    manager = _FakeManager(parent)
    monkeypatch.setattr(workspace_utils, "workspace_manager", manager)
    monkeypatch.setattr(workspace_utils, "update_workspace", lambda *a, **k: None)
    from ldaca_wordflow.api.workspaces import (
        nodes_concat,
        nodes_crud,
        nodes_expression,
        nodes_filter,
        nodes_join,
        nodes_slice,
    )

    for mod in (
        nodes_crud,
        nodes_concat,
        nodes_expression,
        nodes_filter,
        nodes_join,
        nodes_slice,
    ):
        monkeypatch.setattr(mod, "update_workspace", lambda *a, **k: None, raising=False)
    return manager, parent


@pytest.fixture
def two_parents(monkeypatch: pytest.MonkeyPatch):
    """Workspace with two schema-aligned parents, both carrying tokenization."""
    parent_a = _make_node_with_tokens("parent_a")
    parent_b = _make_node_with_tokens("parent_b")
    manager = _FakeManager(parent_a, parent_b)
    monkeypatch.setattr(workspace_utils, "workspace_manager", manager)
    monkeypatch.setattr(workspace_utils, "update_workspace", lambda *a, **k: None)
    from ldaca_wordflow.api.workspaces import (
        nodes_concat,
        nodes_crud,
        nodes_expression,
        nodes_filter,
        nodes_join,
        nodes_slice,
    )

    for mod in (
        nodes_crud,
        nodes_concat,
        nodes_expression,
        nodes_filter,
        nodes_join,
        nodes_slice,
    ):
        monkeypatch.setattr(mod, "update_workspace", lambda *a, **k: None, raising=False)
    return manager, parent_a, parent_b


def _child(manager: _FakeManager, parent_ids: set[str]) -> Node:
    """Find the newest child node added to the workspace, given the parent IDs
    we expect to see in its ``parents`` list.
    """
    for node in manager.workspace.nodes.values():
        if node.id in parent_ids:
            continue
        return node
    raise AssertionError("no child node was added to the workspace")


# ---------------------------------------------------------------------------
# Single-parent, schema-preserving endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clone_inherits_tokenization(single_parent):
    manager, parent = single_parent

    await nodes_api.clone_node(
        workspace_id=uuid.UUID(manager.workspace_id),
        node_id=parent.id,
        current_user={"id": "user"},
    )

    child = _child(manager, {parent.id})
    assert child.tokenization["text"]["column_name"] == TOKENIZATION_COLUMN
    assert child.tokenization["text"]["model"] == "lindera:ja-ipadic"
    assert child.tokenization["text"]["language"] == "ja"


@pytest.mark.asyncio
async def test_filter_inherits_tokenization(single_parent):
    manager, parent = single_parent
    request = FilterRequest(
        conditions=[FilterCondition(column="id", operator="greater_than", value=0)],
    )

    await nodes_api.filter_node(
        workspace_id=uuid.UUID(manager.workspace_id),
        node_id=parent.id,
        request=request,
        current_user={"id": "user"},
    )

    child = _child(manager, {parent.id})
    assert child.tokenization["text"] == TOKENIZATION_META


@pytest.mark.asyncio
async def test_slice_inherits_tokenization(single_parent):
    manager, parent = single_parent
    request = SliceRequest(offset=0, length=1)

    await nodes_api.slice_node(
        workspace_id=uuid.UUID(manager.workspace_id),
        node_id=parent.id,
        request=request,
        current_user={"id": "user"},
    )

    child = _child(manager, {parent.id})
    assert child.tokenization["text"] == TOKENIZATION_META


@pytest.mark.asyncio
async def test_sample_inherits_tokenization(single_parent):
    manager, parent = single_parent
    request = SliceRequest(mode="random_sample", sample_size=0.5, random_seed=1)

    await nodes_api.slice_node(
        workspace_id=uuid.UUID(manager.workspace_id),
        node_id=parent.id,
        request=request,
        current_user={"id": "user"},
    )

    child = _child(manager, {parent.id})
    assert child.tokenization["text"] == TOKENIZATION_META


# ---------------------------------------------------------------------------
# Multi-parent endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concat_inherits_tokenization_from_parents(two_parents):
    manager, parent_a, parent_b = two_parents
    request = ConcatRequest(node_ids=[parent_a.id, parent_b.id], deduplicate=False)

    await nodes_api.concat_nodes(
        workspace_id=uuid.UUID(manager.workspace_id),
        request=request,
        current_user={"id": "user"},
    )

    child = _child(manager, {parent_a.id, parent_b.id})
    assert child.tokenization["text"]["model"] == "lindera:ja-ipadic"


@pytest.mark.asyncio
async def test_join_inherits_tokenization_from_both_parents(two_parents):
    manager, parent_a, parent_b = two_parents

    await nodes_api.join_nodes(
        workspace_id=uuid.UUID(manager.workspace_id),
        left_node_id=parent_a.id,
        right_node_id=parent_b.id,
        left_on="id",
        right_on="id",
        how="inner",
        current_user={"id": "user"},
    )

    child = _child(manager, {parent_a.id, parent_b.id})
    assert child.tokenization["text"] == TOKENIZATION_META


# ---------------------------------------------------------------------------
# Expression apply — schema-changing variant must filter tokenization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expression_apply_with_columns_inherits_tokenization(single_parent):
    manager, parent = single_parent
    # with_columns + new_node_name forces a new child node (the in-place
    # branch is at L1577 in nodes.py and doesn't go through our fix).
    request = PolarsExpressionRequest(
        context=PolarsExpressionContext.with_columns,
        expressions=[PolarsExpressionItem(code="pl.col('id').alias('id2')")],
        new_node_name="parent_with_id2",
    )

    await nodes_api.polars_expression_apply(
        workspace_id=uuid.UUID(manager.workspace_id),
        node_id=parent.id,
        request=request,
        current_user={"id": "user"},
    )

    child = _child(manager, {parent.id})
    assert child.tokenization["text"] == TOKENIZATION_META


@pytest.mark.asyncio
async def test_expression_apply_select_drops_tokenization_if_column_gone(single_parent):
    manager, parent = single_parent
    # A select that keeps only `id` drops the source text column. The
    # tokenization entry pointing at the now-absent source column must be filtered out.
    request = PolarsExpressionRequest(
        context=PolarsExpressionContext.select,
        expressions=[PolarsExpressionItem(code="pl.col('id')")],
        new_node_name="parent_only_id",
    )

    await nodes_api.polars_expression_apply(
        workspace_id=uuid.UUID(manager.workspace_id),
        node_id=parent.id,
        request=request,
        current_user={"id": "user"},
    )

    child = _child(manager, {parent.id})
    assert child.tokenization == {}
