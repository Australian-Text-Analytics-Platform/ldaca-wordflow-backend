"""Strict Concordance process-input contract tests."""

import uuid

import polars as pl
import pytest
from pydantic import ValidationError

from ldaca_wordflow.domain.workspace import Node, Workspace
from ldaca_wordflow.workers.concordance import (
    _build_concordance_response_from_snapshot,
)
from ldaca_wordflow.workers.input_snapshots import create_worker_input_snapshot


@pytest.mark.parametrize("legacy_field", ["page", "page_size", "result_node_id"])
def test_concordance_worker_rejects_legacy_query_fields(legacy_field: str) -> None:
    node_id = str(uuid.uuid4())
    payload: dict[str, object] = {
        "node_ids": [node_id],
        "node_columns": {node_id: "text"},
        "search_word": "example",
        legacy_field: 1,
    }

    with pytest.raises(ValidationError):
        _build_concordance_response_from_snapshot(
            input_snapshot_dir="unused",
            token_cache_path=None,
            request_payload=payload,
        )


def test_tokens_mode_uses_request_model_and_plain_model_bypasses_cache(
    tmp_path,
) -> None:
    node_id = str(uuid.uuid4())
    workspace = Workspace(name="request owned", workspace_id=str(uuid.uuid4()))
    workspace.add_node(
        Node(
            id=node_id,
            name="Corpus",
            data=pl.DataFrame({"text": ["Hello world", "Other"]}).lazy(),
            tokenizer_model="lindera:jieba",
        )
    )
    snapshot = create_worker_input_snapshot(
        workspace_id=workspace.id,
        node_ids=[node_id],
        workspace=workspace,
        workspace_data_dir=tmp_path,
        snapshot_dir=tmp_path / "snapshot",
        max_snapshot_bytes=1024 * 1024,
    )
    cache_path = tmp_path / "tokens.duckdb"

    result = _build_concordance_response_from_snapshot(
        input_snapshot_dir=str(snapshot),
        token_cache_path=str(cache_path),
        request_payload={
            "node_ids": [node_id],
            "node_columns": {node_id: "text"},
            "node_tokenizer_models": {node_id: "native:plain_words_en"},
            "search_word": "hello",
            "search_mode": "tokens",
        },
    )

    rows = result["sources"][0]["result"]["data"]
    assert rows[0][0]["CONC_matched_text"] == "hello"
    assert not cache_path.exists()
