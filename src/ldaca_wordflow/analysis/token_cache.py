"""Analysis integration for the polars-text token cache.

Nodes store per-column tokenisation specs in ``Node.tokenization``. Analyses
attach those specs to a LazyFrame with ``hydrate_tokenization_lazyframe``.
The generic DuckDB cache mechanics live in ``pl.col(...).text.tokenize(...,
cache=...)``. Callers pass the runtime-owned cache path explicitly so worker
processes and independently configured app instances never consult globals.

Used by Analysis preparation, process entrypoints, and focused backend tests.

Flow: resolve tokenization preferences, hydrate or create token columns, aggregate
    frequencies, and persist derived artifacts for result queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import polars as pl
import polars_text  # noqa: F401

from ..domain.workspace import Node

TOKENS_CACHE_FILENAME = "tokens.duckdb"


def tokens_cache_path(cache_root: str | Path) -> Path:
    """Return the token-cache file below an explicitly owned cache root."""
    return Path(cache_root) / TOKENS_CACHE_FILENAME


def hydrate_tokenization_lazyframe(
    *,
    node: Node,
    source_column: str,
    cache_path: str | Path,
) -> pl.LazyFrame:
    """Lazily attach a tokenization column registered on ``node``.

    Short-circuits if the column is already physically present. Otherwise reads
    the model, token column, and tokenisation params from
    ``node.tokenization[source_column]`` and attaches a cache-backed elementwise
    expression keyed on the explicit runtime-owned cache path.

    The hydrated column exists only in the returned plan; it is never written
    back to the Workspace aggregate.
    """
    tokenization_meta = node.tokenization.get(source_column)
    if tokenization_meta is None:
        return node.data

    tokenization_column = tokenization_meta["column_name"]
    model = tokenization_meta["model"]
    params = tokenization_meta["params"]

    if tokenization_column in node.data.collect_schema().names():
        return node.data

    return node.data.with_columns(
        cast(Any, pl.col(source_column))
        .text.tokenize(
            lowercase=bool(params.get("lowercase", True)),
            remove_punct=bool(params.get("remove_punct", True)),
            model=model,
            cache=Path(cache_path),
        )
        .alias(tokenization_column)
    )


__all__ = [
    "TOKENS_CACHE_FILENAME",
    "hydrate_tokenization_lazyframe",
    "tokens_cache_path",
]
