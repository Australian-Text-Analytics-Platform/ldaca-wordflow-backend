"""Tokenization column naming helper + tokens schema contract.

Asserts:
- dynamic token column names round-trip,
- ``tokens_struct_dtype()`` lines up with the schema polars-text's
    ``tokenize`` actually emits, and
- ``is_tokenization_column`` reads from ``Node.tokenization``, not from a
    fixed magic column name.

These are the contracts every tokenization consumer relies on, so any drift
between Rust and Python schemas — or between the naming helper and its consumers
— must fail loudly here.
"""

from __future__ import annotations

from typing import Any, cast

import polars as pl
import polars_text  # noqa: F401
from ldaca_wordflow.api.workspaces.analyses.generated_columns import (
    TOKENS_END_FIELD,
    TOKENS_START_FIELD,
    TOKENS_TOKEN_FIELD,
    is_tokenization_column,
    parse_tokenization_column,
    tokenization_column_name,
    tokens_struct_dtype,
    tokens_struct_projection,
)

from docworkspace import Node

# Test fixture: canonical (source, model) we use throughout this module.
_TEXT_COLUMN = "text"
_BERT_MODEL = "huggingface:bert-base-uncased"
_TOKENS_NAME = f"tokenization.{_TEXT_COLUMN}.{_BERT_MODEL}"


def test_tokenization_column_name_builds_canonical_label() -> None:
    assert tokenization_column_name(_TEXT_COLUMN, _BERT_MODEL) == _TOKENS_NAME


def test_parse_tokenization_column_round_trips() -> None:
    assert parse_tokenization_column(_TOKENS_NAME) == (
        _TEXT_COLUMN,
        _BERT_MODEL,
    )


def test_parse_tokenization_column_rejects_non_tokenization_names() -> None:
    assert parse_tokenization_column("plain_column") is None
    # Missing prefix.
    assert parse_tokenization_column("tokens.text.lindera:jieba") is None
    # Wrong number of parts (source or model containing dots is ambiguous —
    # by design we treat it as unparseable; consult Node.tokenization instead).
    assert parse_tokenization_column("tokenization.text.foo.bar") is None


def test_tokens_struct_dtype_matches_polars_text_output() -> None:
    df = pl.DataFrame({"text": ["Hello world"]})
    out = df.select(
        cast(Any, pl.col("text")).text.tokenize(model=_BERT_MODEL).alias(_TOKENS_NAME)
    )
    assert out.schema[_TOKENS_NAME] == tokens_struct_dtype(), (
        f"polars-text emits {out.schema[_TOKENS_NAME]!r}, "
        f"but generated_columns declares {tokens_struct_dtype()!r}"
    )


def test_is_tokenization_column_reads_from_node_metadata() -> None:
    # Build a node with a token column registered in Node.tokenization.
    df = pl.DataFrame({"text": ["hi"]})
    with_tokens = df.lazy().select(
        pl.col("text"),
        cast(Any, pl.col("text")).text.tokenize(model=_BERT_MODEL).alias(_TOKENS_NAME),
    )
    node = Node(data=with_tokens, name="tokens_root")
    node.register_tokenization(
        _TEXT_COLUMN,
        {
            "column_name": _TOKENS_NAME,
            "model": _BERT_MODEL,
            "language": "en",
            "params": {"lowercase": True, "remove_punct": True},
        },
    )
    assert is_tokenization_column(node, _TOKENS_NAME)


def test_is_tokenization_column_rejects_unregistered_column() -> None:
    df = pl.DataFrame({"text": ["hi"]})
    out = df.lazy().select(
        cast(Any, pl.col("text")).text.tokenize(model=_BERT_MODEL).alias(_TOKENS_NAME)
    )
    node = Node(data=out, name="unregistered")
    # Column exists in schema but isn't in Node.tokenization → not a tokens column.
    assert not is_tokenization_column(node, _TOKENS_NAME)


def test_is_tokenization_column_rejects_non_token_column() -> None:
    df = pl.DataFrame({"text": ["hi"]}).lazy()
    pos_name = "text.pos.spacy-en"
    node = Node(data=df, name="pos_root")
    assert not is_tokenization_column(node, pos_name)


def test_tokens_struct_projection_unpacks_fields() -> None:
    df = pl.DataFrame({"text": ["hello world"]})
    tokens_df = df.select(
        cast(Any, pl.col("text")).text.tokenize(model=_BERT_MODEL).alias(_TOKENS_NAME)
    ).explode(_TOKENS_NAME)
    unpacked = tokens_df.select(*tokens_struct_projection(_TOKENS_NAME))
    assert set(unpacked.columns) == {
        TOKENS_TOKEN_FIELD,
        TOKENS_START_FIELD,
        TOKENS_END_FIELD,
    }
    assert unpacked.height > 0
