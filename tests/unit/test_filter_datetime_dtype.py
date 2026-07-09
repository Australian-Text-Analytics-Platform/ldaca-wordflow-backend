"""Regression tests for datetime dtype handling in filter expressions.

Polars rejects comparisons between Datetime values whose time-unit or
time-zone differ. The frontend always sends ISO strings with an explicit
``+00:00`` offset, so ``_parse_temporal`` produces tz-aware ``datetime``
objects. ``pl.lit`` on such a value yields ``Datetime("us", "UTC")``.

If the column is tz-naive (e.g. ``Datetime("ns")`` from social-data dumps
where timestamps are conceptually UTC but stored without a tz attached),
the comparison fails with::

    could not evaluate '<' comparison between series ... of dtype:
    Datetime('ns') and series 'literal' of dtype: Datetime('us', 'UTC')

`_make_temporal_literal` casts the literal to match the column dtype.
"""

from __future__ import annotations

from typing import cast

import polars as pl
from ldaca_wordflow.api.workspaces.nodes_filter import _build_filter_expression
from ldaca_wordflow.models.nodes import FilterCondition, FilterRequest


def _filter(df: pl.DataFrame, request: FilterRequest) -> pl.DataFrame:
    schema_map = dict(df.schema)
    expr = _build_filter_expression(request, column_dtypes=schema_map)
    return cast(pl.DataFrame, df.lazy().filter(expr).collect())


def test_lt_naive_ns_column_against_utc_literal() -> None:
    """Reproduces the bug: tz-naive ns column vs tz-aware UTC literal."""
    df = pl.DataFrame(
        {
            "created_utc_max": pl.Series(
                ["2025-03-01T00:00:00", "2025-04-01T00:00:00", "2025-05-01T00:00:00"]
            ).str.to_datetime(time_unit="ns")
        }
    )
    assert df.schema["created_utc_max"] == pl.Datetime("ns")

    request = FilterRequest(
        conditions=[
            FilterCondition(
                column="created_utc_max",
                operator="lt",
                value="2025-04-15T00:00:00+00:00",
            )
        ],
        logic="and",
    )
    result = _filter(df, request)
    assert result.height == 2


def test_between_naive_ns_column_against_utc_literals() -> None:
    df = pl.DataFrame(
        {
            "created_utc": pl.Series(
                [
                    "2025-03-01T00:00:00",
                    "2025-03-15T00:00:00",
                    "2025-04-01T00:00:00",
                    "2025-05-01T00:00:00",
                ]
            ).str.to_datetime(time_unit="ns")
        }
    )

    request = FilterRequest(
        conditions=[
            FilterCondition(
                column="created_utc",
                operator="between",
                value={
                    "start": "2025-03-10T00:00:00+00:00",
                    "end": "2025-04-15T00:00:00+00:00",
                },
            )
        ],
        logic="and",
    )
    result = _filter(df, request)
    assert result.height == 2


def test_eq_tz_aware_column_against_utc_literal() -> None:
    """Tz-aware columns still work after the fix."""
    df = pl.DataFrame(
        {
            "ts": pl.Series(
                ["2025-04-01T00:00:00+00:00", "2025-04-02T00:00:00+00:00"]
            ).str.to_datetime(time_unit="us", time_zone="UTC")
        }
    )
    assert df.schema["ts"] == pl.Datetime("us", "UTC")

    request = FilterRequest(
        conditions=[
            FilterCondition(
                column="ts",
                operator="eq",
                value="2025-04-01T00:00:00+00:00",
            )
        ],
        logic="and",
    )
    result = _filter(df, request)
    assert result.height == 1
