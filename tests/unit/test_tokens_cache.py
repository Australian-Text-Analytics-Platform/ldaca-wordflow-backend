"""Unit tests for the per-user DuckDB-backed token cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import duckdb
import polars as pl
import polars_text  # noqa: F401
import pytest
from ldaca_wordflow.api.workspaces.analyses.generated_columns import (
    tokenization_column_name,
)
from ldaca_wordflow.core import tokens_cache as tc
from ldaca_wordflow.core.tokenization import tokenise_column

from docworkspace import Node

TEST_USER = "test_user"


@pytest.fixture(autouse=True)
def isolated_cache_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "tokens.duckdb"
    monkeypatch.setattr(tc, "tokens_cache_path", lambda _user_id: db_path)
    return db_path


def test_cache_schema_has_six_columns(isolated_cache_db: Path) -> None:
    base = pl.DataFrame({"text": ["hello world"]}).lazy()
    expr = cast(Any, pl.col("text")).text.tokenize(
        model="huggingface:bert-base-uncased",
        lowercase=True,
        remove_punct=True,
        cache=tc.tokens_cache_path(TEST_USER),
    )
    base.with_columns(expr.alias("tokens")).collect()

    assert isolated_cache_db.exists()

    with duckdb.connect(str(isolated_cache_db), read_only=True) as conn:
        rows = conn.execute("DESCRIBE token_cache").fetchall()

    assert [row[0] for row in rows] == [
        "model",
        "params_hash",
        "content_hash",
        "tokens",
        "start_offsets",
        "end_offsets",
    ]


def test_tokenise_column_registers_metadata_without_mutating_node_data() -> None:
    node = Node(
        data=pl.DataFrame({"text": ["hello world"]}).lazy(),
        name="probe",
    )

    tokenization_name = tokenise_column(
        node,
        source_column="text",
        model="huggingface:bert-base-uncased",
        language="en",
    )

    assert tokenization_name == tokenization_column_name(
        "text", "huggingface:bert-base-uncased"
    )
    assert node.tokenization["text"]["column_name"] == tokenization_name
    assert tokenization_name not in node.data.collect_schema().names()
    tokenization_meta = cast(dict[str, Any], node.tokenization["text"])
    assert "source_column" not in tokenization_meta
    assert "cache_backend" not in tokenization_meta
    assert "generated_at" not in tokenization_meta


def test_hydrate_tokenization_adds_tokens_column() -> None:
    node = Node(
        data=pl.DataFrame({"text": ["hello world", "hello again"]}).lazy(),
        name="probe",
    )
    tokenization_name = tokenise_column(
        node,
        source_column="text",
        model="huggingface:bert-base-uncased",
        language="en",
    )

    hydrated = tc.hydrate_tokenization_lazyframe(
        node=node,
        source_column="text",
        user_id=TEST_USER,
    )
    hydrated_df = cast(pl.DataFrame, hydrated.collect())

    assert tokenization_name in hydrated_df.columns
    assert tokenization_name not in node.data.collect_schema().names()
    first_tokens = hydrated_df.to_dicts()[0][tokenization_name]
    assert isinstance(first_tokens, list) and first_tokens
    assert first_tokens[0]["token"]
