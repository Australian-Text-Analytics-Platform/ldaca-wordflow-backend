"""Canonical physical types for Topic Distribution values."""

from __future__ import annotations

import polars as pl

TOPIC_DISTRIBUTION_ENTRY_DTYPE = pl.Struct(
    [
        pl.Field("topic_id", pl.Int64),
        pl.Field("proportion", pl.Float64),
    ]
)


def topic_distribution_storage_dtype(topic_count: int) -> pl.Array:
    """Return storage for outlier ``-1`` plus every real topic."""

    if topic_count < 0:
        raise ValueError("Topic count cannot be negative")
    return pl.Array(TOPIC_DISTRIBUTION_ENTRY_DTYPE, topic_count + 1)


def is_topic_distribution_storage_dtype(dtype: pl.DataType) -> bool:
    """Whether ``dtype`` is the canonical fixed-size physical representation."""

    return isinstance(dtype, pl.Array) and dtype.inner == TOPIC_DISTRIBUTION_ENTRY_DTYPE


def topic_count_from_storage_dtype(dtype: pl.DataType) -> int:
    """Read the real-topic count from canonical storage or fail clearly."""

    if not isinstance(dtype, pl.Array) or dtype.inner != TOPIC_DISTRIBUTION_ENTRY_DTYPE:
        raise ValueError("Topic Distribution storage is not a canonical fixed-size array")
    return dtype.size - 1


__all__ = [
    "TOPIC_DISTRIBUTION_ENTRY_DTYPE",
    "is_topic_distribution_storage_dtype",
    "topic_count_from_storage_dtype",
    "topic_distribution_storage_dtype",
]
