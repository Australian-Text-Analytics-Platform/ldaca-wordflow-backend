"""``tokenise_column`` registers cache-backed tokenization metadata."""

from __future__ import annotations

import os
import polars as pl
import pytest
from ldaca_wordflow.analysis.generated_columns import (
    tokenization_column_name,
)
from ldaca_wordflow.analysis.tokenization import tokenise_column
from ldaca_wordflow.analysis.token_cache import hydrate_tokenization_lazyframe

from ldaca_wordflow.domain.workspace import Node

_LINDERA_JIEBA_TESTS_ENV = "POLARS_TEXT_RUN_LINDERA_JIEBA_TESTS"


def _make_node(name: str = "root") -> Node:
    df = pl.DataFrame(
        {
            "text": ["Hello world", "Goodbye world", "Hello again"],
            "value": [1, 2, 3],
        }
    ).lazy()
    return Node(data=df, name=name)


def test_tokenise_registers_metadata_without_mutating_node_data() -> None:
    node = _make_node()
    expected_name = tokenization_column_name("text", "huggingface:bert-base-uncased")

    result_name = tokenise_column(
        node,
        source_column="text",
        model="huggingface:bert-base-uncased",
        language="en",
    )

    assert result_name == expected_name
    assert expected_name not in node.data.collect_schema().names()
    assert "text" in node.tokenization
    meta = node.tokenization["text"]
    assert "source_column" not in meta
    assert meta["column_name"] == expected_name
    assert meta["model"] == "huggingface:bert-base-uncased"
    assert meta["language"] == "en"
    assert "cache_backend" not in meta
    assert meta["params"] == {"lowercase": True, "remove_punct": True}
    assert "generated_at" not in meta

    schema_names = node.data.collect_schema().names()
    assert "text" in schema_names
    assert "value" in schema_names


def test_tokenise_is_idempotent_on_source_and_model() -> None:
    node = _make_node()

    first = tokenise_column(
        node,
        source_column="text",
        model="huggingface:bert-base-uncased",
        language="en",
    )
    tokenization_count_first = len(node.tokenization)

    second = tokenise_column(
        node,
        source_column="text",
        model="huggingface:bert-base-uncased",
        language="en",
    )
    tokenization_count_second = len(node.tokenization)

    assert first == second
    assert tokenization_count_first == 1
    assert tokenization_count_second == 1
    assert len(node.tokenization) == 1


def test_tokenise_with_different_model_replaces_node_token_spec() -> None:
    node = _make_node()

    bert_name = tokenise_column(
        node,
        source_column="text",
        model="huggingface:bert-base-uncased",
        language="en",
    )
    multi_name = tokenise_column(
        node,
        source_column="text",
        model="huggingface:bert-base-multilingual-cased",
        language="en",
    )

    assert bert_name != multi_name
    assert len(node.tokenization) == 1
    assert (
        node.find_tokenization_column("text", model="huggingface:bert-base-uncased")
        is None
    )
    assert (
        node.find_tokenization_column(
            "text", model="huggingface:bert-base-multilingual-cased"
        )
        == multi_name
    )


def test_tokenise_with_different_source_preserves_existing_token_specs() -> None:
    node = _make_node()

    text_name = tokenise_column(
        node,
        source_column="text",
        model="huggingface:bert-base-uncased",
        language="en",
    )
    value_name = tokenise_column(
        node,
        source_column="value",
        model="huggingface:bert-base-uncased",
        language="en",
    )

    assert text_name != value_name
    assert len(node.tokenization) == 2
    assert node.find_tokenization_column("text") == text_name
    assert node.find_tokenization_column("value") == value_name


def test_tokenise_does_not_mutate_node_data() -> None:
    node = _make_node()
    schema_before = set(node.data.collect_schema().names())

    tokenise_column(
        node,
        source_column="text",
        model="huggingface:bert-base-uncased",
        language="en",
    )
    assert set(node.data.collect_schema().names()) == schema_before


def test_tokenise_rejects_missing_source_column() -> None:
    node = _make_node()
    with pytest.raises(KeyError):
        tokenise_column(
            node,
            source_column="nonexistent",
            model="huggingface:bert-base-uncased",
            language="en",
        )


def test_tokenise_emits_canonical_struct_dtype(tmp_path) -> None:
    """Hydrated tokenization columns use the canonical struct dtype."""
    from ldaca_wordflow.analysis.generated_columns import (
        tokens_struct_dtype,
    )

    node = _make_node()
    tokenization_name = tokenise_column(
        node,
        source_column="text",
        model="huggingface:bert-base-uncased",
        language="en",
    )
    hydrated = hydrate_tokenization_lazyframe(
        node=node,
        source_column="text",
        cache_path=tmp_path / "tokens.duckdb",
    )
    schema = hydrated.collect_schema()
    assert schema[tokenization_name] == tokens_struct_dtype()


@pytest.mark.skipif(
    _LINDERA_JIEBA_TESTS_ENV not in os.environ,
    reason=(
        f"Set {_LINDERA_JIEBA_TESTS_ENV}=1 and provide a reachable "
        "lindera:jieba dictionary archive to run Jieba download tests."
    ),
)
def test_tokenise_chinese_via_jieba_produces_word_level_tokens(tmp_path) -> None:
    """lindera:jieba is reachable through tokenise_column and produces word-level tokens."""
    df = pl.DataFrame({"text": ["今天天气很好"]}).lazy()
    node = Node(data=df, name="zh_root")

    tokenization_name = tokenise_column(
        node,
        source_column="text",
        model="lindera:jieba",
        language="zh",
    )

    hydrated = hydrate_tokenization_lazyframe(
        node=node,
        source_column="text",
        cache_path=tmp_path / "tokens.duckdb",
    )
    collected = hydrated.collect()
    tokens_lists = collected[tokenization_name].to_list()
    assert len(tokens_lists) == 1
    tokens = [entry["token"] for entry in tokens_lists[0]]
    # Word-level segmentation produces multi-character tokens, not pure chars.
    assert any(len(tok) > 1 for tok in tokens), (
        f"expected word-level Chinese tokens, got char-level: {tokens!r}"
    )
