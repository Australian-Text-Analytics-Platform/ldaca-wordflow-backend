"""Canonical physical dtype used for per-document topic distributions."""

from __future__ import annotations

import polars as pl


TM_DISTRIBUTION_POLARS_DTYPE = pl.List(
    pl.Struct(
        [
            pl.Field("topic_id", pl.Int64),
            pl.Field("proportion", pl.Float64),
        ]
    )
)


__all__ = ["TM_DISTRIBUTION_POLARS_DTYPE"]
