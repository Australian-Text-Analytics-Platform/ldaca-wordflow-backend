"""Service-level Data Block Edit metadata and history behavior."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import anyio
import polars as pl
import pytest

from ldaca_wordflow.domain.workspace import Node, Workspace
from ldaca_wordflow.models.node_resources import (
    DeleteColumnNodeEditRequest,
    RenameColumnNodeEditRequest,
)
from ldaca_wordflow.services.nodes import NodeService


class _WorkspaceGate:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.revision = 1

    @asynccontextmanager
    async def mutation_context(self, _user_id: str, _workspace_id: str):
        lease = SimpleNamespace(
            workspace=self.workspace,
            revision=self.revision,
            commit_requested=True,
        )
        yield lease
        if lease.commit_requested:
            self.revision += 1
            lease.revision = self.revision


def _service(workspace: Workspace) -> NodeService:
    return NodeService(
        cast(Any, _WorkspaceGate(workspace)),
        cast(Any, None),
        storage_admission=cast(Any, None),
        io_limiter=anyio.CapacityLimiter(2),
        max_source_bytes=1,
        max_storage_bytes=1,
    )


@pytest.mark.anyio
async def test_forward_edits_retarget_and_reconcile_non_undoable_metadata() -> None:
    workspace = Workspace(name="metadata")
    node = workspace.add_node(
        Node(
            data=pl.DataFrame(
                {
                    "text": ["hello"],
                    "text_tokens": [["hello"]],
                    "other": [1],
                }
            ).lazy(),
            name="source",
            document="text",
            tokenization={
                "text": {
                    "column_name": "text_tokens",
                    "model": "native:plain_words_en",
                    "language": "en",
                    "params": {},
                }
            },
        )
    )
    service = _service(workspace)

    renamed_source, _revision = await service.edit(
        "user",
        workspace.id,
        node.id,
        RenameColumnNodeEditRequest(column="text", new_name="body"),
    )
    assert renamed_source.document == "body"
    assert renamed_source.tokenizer_models == {
        "body": "native:plain_words_en"
    }

    renamed_tokens, _revision = await service.edit(
        "user",
        workspace.id,
        node.id,
        RenameColumnNodeEditRequest(
            column="text_tokens",
            new_name="tokens",
        ),
    )
    assert renamed_tokens.tokenizer_models == {
        "body": "native:plain_words_en"
    }
    assert node.tokenization["body"]["column_name"] == "tokens"

    deleted_source, _revision = await service.edit(
        "user",
        workspace.id,
        node.id,
        DeleteColumnNodeEditRequest(column="body"),
    )
    assert deleted_source.document is None
    assert deleted_source.tokenizer_models == {}

    restored_plan, _revision = await service.undo("user", workspace.id, node.id)
    assert "body" in node.data.collect_schema()
    assert restored_plan.document is None
    assert restored_plan.tokenizer_models == {}
