"""Strict Concordance process-input contract tests."""

import uuid

import pytest
from pydantic import ValidationError

from ldaca_wordflow.workers.concordance import (
    _build_concordance_response_from_snapshot,
)


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
