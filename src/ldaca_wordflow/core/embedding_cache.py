"""Wordflow integration for the Rust embedding cache.

The actual DuckDB schema and vector reuse live in ``polars_text`` on the Rust
side. This module owns only Wordflow's per-user cache path so topic modeling can
use ``embeddings.duckdb`` without sharing or mutating the tokenization cache.

Used by:
- ``worker_tasks_topic`` because topic-modeling workers need a stable per-user
  cache location to pass into the Rust pipeline.

Flow: resolve the user cache folder and append the embedding-cache filename.
"""

from __future__ import annotations

from pathlib import Path

from .utils import get_user_cache_folder

EMBEDDINGS_CACHE_FILENAME = "embeddings.duckdb"


def embeddings_cache_path(user_id: str) -> Path:
    """Return the per-user DuckDB embedding cache path.

    Used by:
    - topic-modeling worker tests and runtime code because both need the same
      observable ``embeddings.duckdb`` location while leaving ``tokens.duckdb``
      reserved for tokenization.

    Flow: resolve Wordflow's per-user cache folder, then append the embedding
        cache filename.
    """
    return get_user_cache_folder(user_id) / EMBEDDINGS_CACHE_FILENAME


__all__ = ["EMBEDDINGS_CACHE_FILENAME", "embeddings_cache_path"]
