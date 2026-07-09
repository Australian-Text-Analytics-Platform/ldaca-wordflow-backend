"""Workspace node join helpers.

Used by:
- ``nodes_join.join_nodes_preview`` and ``nodes_join.join_nodes`` because both
  endpoints need the same join-type validation and LazyFrame join construction.

Flow:
- Normalize and validate the requested join type.
- Build the joined LazyFrame, requiring join columns for non-cross joins.
- For preview, collect a page plus pagination metadata without persisting a node.
"""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import polars as pl

from ...core.exceptions import InternalServiceError, InvalidInputError

JOIN_HOWS = {"inner", "left", "right", "full", "semi", "anti", "cross"}
JoinHow = Literal["inner", "left", "right", "full", "semi", "anti", "cross"]


def _join_how(value: str | None) -> JoinHow:
    """Normalize and validate one join type value."""

    how = (value or "inner").lower()
    if how not in JOIN_HOWS:
        raise InvalidInputError(
            "Invalid join type. Allowed values: inner, left, right, full, semi, anti, cross",
        )
    return cast(JoinHow, how)


def joined_lazyframe_for_nodes(
    left_node: Any,
    right_node: Any,
    *,
    left_on: str | None,
    right_on: str | None,
    how: str,
) -> pl.LazyFrame:
    """Build the joined LazyFrame for two workspace nodes.

    Used by:
    - join preview and apply routes so they share one validation path.
    """

    join_how = _join_how(how)
    if join_how == "cross":
        return left_node.data.join(right_node.data, how="cross")
    if not left_on or not right_on:
        raise InvalidInputError("left_on and right_on must be provided for non-cross joins")
    return left_node.data.join(
        right_node.data,
        left_on=left_on,
        right_on=right_on,
        how=join_how,
    )


def preview_joined_nodes(
    *,
    left_node: Any,
    right_node: Any,
    left_on: str | None,
    right_on: str | None,
    how: str,
    page: int,
    page_size: int,
) -> dict[str, Any]:
    """Return a paginated preview for a node join."""

    joined_lazy = joined_lazyframe_for_nodes(
        left_node,
        right_node,
        left_on=left_on,
        right_on=right_on,
        how=how,
    )
    try:
        total_rows_df = cast(
            pl.DataFrame,
            joined_lazy.select(pl.len().alias("_len")).collect(),
        )
        total_rows = int(total_rows_df.to_series(0).item())
    except Exception:
        total_rows = None

    offset = (page - 1) * page_size
    try:
        preview_df = cast(pl.DataFrame, joined_lazy.slice(offset, page_size).collect())
    except Exception as exc:
        raise InternalServiceError(str(exc)) from exc

    preview_rows = preview_df.to_dicts()
    if total_rows is None:
        has_next = len(preview_rows) == page_size
        total_rows_value = offset + len(preview_rows) + (page_size if has_next else 0)
        total_pages = max(1, page + (1 if has_next else 0))
    else:
        has_next = offset + page_size < total_rows
        total_rows_value = total_rows
        total_pages = max(1, math.ceil(total_rows / page_size))

    return {
        "data": preview_rows,
        "columns": list(preview_df.columns),
        "dtypes": {column: str(dtype) for column, dtype in preview_df.schema.items()},
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows_value,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": page > 1,
        },
    }
