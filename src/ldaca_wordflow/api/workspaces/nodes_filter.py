"""Filter node endpoints: build filter expressions, preview, and apply.

Used by:
- Frontend and API clients through the FastAPI filter routes.

Flow:
- Resolve workspace and node, build Polars filter expressions from request conditions,
- Return preview rows or persist a new filtered child node.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, cast

import polars as pl
from fastapi import APIRouter, Depends, Query

from ...core.auth import get_current_user
from ...core.docworkspace_data_types import TM_DISTRIBUTION_POLARS_DTYPE
from ...models.analysis_common import PaginationInfo
from ...models.nodes import FilterPreviewResponse, FilterRequest, NodeOperationResponse
from ...core.exceptions import InvalidInputError
from .utils import (
    _coerce_scalar,
    _create_and_persist_child_node,
    _is_string_list_dtype,
    _make_temporal_literal,
    _paginated_lazy_preview,
    _parse_temporal,
    require_workspace,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
    tags=["nodes"],
)

# Comparison operators supported for the TMDist topic-proportion filter.
_TMDIST_OPERATORS = {"gt", "gte", "lt", "lte", "eq", "ne"}


def _tmdist_proportion_expr(column: str, topic_id: int) -> pl.Expr:
    """Expr giving one topic's proportion within a TMDist (distribution) column.

    Maps each ``{topic_id, proportion}`` struct in the per-row list to its
    proportion when it matches ``topic_id`` (else 0), then reduces with
    ``list.max`` so a topic absent from the row's distribution reads as 0.

    Called by: ``_build_filter_expression`` for TMDist topic-proportion filters.
    """
    return (
        pl.col(column)
        .list.eval(
            pl.when(pl.element().struct.field("topic_id") == int(topic_id))
            .then(pl.element().struct.field("proportion"))
            .otherwise(0.0)
        )
        .list.max()
        .fill_null(0.0)
    )


def _tmdist_condition_expr(condition: Any) -> pl.Expr:
    """Build a boolean predicate for one TMDist filter condition.

    The condition value is ``{topic_id, threshold}`` where ``threshold`` is a
    proportion in [0, 1]; ``operator`` is a numeric comparison. Semantics:
    "keep rows where topic <topic_id>'s proportion <op> <threshold>".

    Called by: ``_build_filter_expression`` when the target column is a TMDist
    (topic-distribution) column.
    """
    raw = condition.value
    if not isinstance(raw, dict):
        raise InvalidInputError(
            "Topic-distribution filter value must be {topic_id, threshold}"
        )
    try:
        topic_id = int(raw["topic_id"])
        threshold = float(raw["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidInputError(
            "Topic-distribution filter needs an integer topic_id and numeric threshold"
        ) from exc
    op = condition.operator
    if op not in _TMDIST_OPERATORS:
        raise InvalidInputError(f"Unsupported topic-distribution operator: {op}")
    proportion = _tmdist_proportion_expr(condition.column, topic_id)
    if op == "gt":
        return proportion > threshold
    if op == "gte":
        return proportion >= threshold
    if op == "lt":
        return proportion < threshold
    if op == "lte":
        return proportion <= threshold
    if op == "ne":
        return proportion != threshold
    return proportion == threshold


def _build_filter_expression(
    request: FilterRequest,
    column_dtypes: dict[str, Any] | None = None,
) -> pl.Expr:
    """Build a Polars filter expression from request conditions.

    Called by:
    - filter_node, filter_preview.
    """
    logic = (request.logic or "and").lower()
    filter_expr = None
    schema_map = column_dtypes or {}

    for condition in request.conditions:
        column_expr = pl.col(condition.column)
        column_dtype = schema_map.get(condition.column)
        is_string_list_column = _is_string_list_dtype(column_dtype)
        is_tmdist_column = column_dtype == TM_DISTRIBUTION_POLARS_DTYPE
        op = condition.operator
        raw_value = condition.value
        expr = None

        if is_tmdist_column:
            # Topic-distribution column: compare one topic's proportion against
            # a threshold (value = {topic_id, threshold}). Handled before the
            # generic scalar comparisons, which can't operate on list-of-struct.
            expr = _tmdist_condition_expr(condition)
        elif op in {
            "eq", "equals", "ne", "gt", "greater_than", "gte", "lt", "less_than", "lte",
        }:
            value = _coerce_scalar(_parse_temporal(raw_value))
            lit_val = (
                _make_temporal_literal(value, column_dtype)
                if isinstance(value, datetime)
                else value
            )
            if op in {"eq", "equals"}:
                expr = column_expr == lit_val
            elif op == "ne":
                expr = column_expr != lit_val
            elif op in {"gt", "greater_than"}:
                expr = column_expr > lit_val
            elif op == "gte":
                expr = column_expr >= lit_val
            elif op in {"lt", "less_than"}:
                expr = column_expr < lit_val
            elif op == "lte":
                expr = column_expr <= lit_val
        elif op == "in":
            include_null = False
            values: list[Any] = []

            if isinstance(raw_value, (list, tuple, set)):
                for item in raw_value:
                    if item is None:
                        include_null = True
                        continue
                    values.append(_coerce_scalar(_parse_temporal(item)))
            elif raw_value is None:
                include_null = True
            else:
                values = [_coerce_scalar(_parse_temporal(raw_value))]

            if is_string_list_column:
                string_values = [str(item) for item in values if item is not None]
                if string_values:
                    expr = column_expr.list.eval(
                        pl.element().cast(pl.String).is_in(string_values),
                        parallel=False,
                    ).list.any()
                else:
                    expr = pl.lit(False)
            else:
                if values:
                    expr = column_expr.is_in(values)
                    if include_null:
                        expr = expr | column_expr.is_null()
                elif include_null:
                    expr = column_expr.is_null()
        elif op == "contains":
            pattern = str(raw_value)
            case_sensitive = bool(getattr(condition, "case_sensitive", False))
            if getattr(condition, "regex", False):
                effective_pattern = pattern if case_sensitive else f"(?i){pattern}"
                expr = column_expr.str.contains(effective_pattern)
            elif case_sensitive:
                expr = column_expr.str.contains(pl.lit(pattern), literal=True)
            else:
                expr = column_expr.str.to_lowercase().str.contains(
                    pl.lit(pattern.lower()), literal=True
                )
        elif op == "startswith":
            expr = column_expr.str.starts_with(str(raw_value))
        elif op == "endswith":
            expr = column_expr.str.ends_with(str(raw_value))
        elif op == "is_null":
            expr = column_expr.is_null()
        elif op == "is_not_null":
            expr = column_expr.is_not_null()
        elif op == "between":
            expr = pl.lit(True)
            if isinstance(raw_value, dict):
                start_val = (
                    _parse_temporal(raw_value.get("start"))
                    if raw_value.get("start") is not None
                    else None
                )
                end_val = (
                    _parse_temporal(raw_value.get("end"))
                    if raw_value.get("end") is not None
                    else None
                )
                if start_val is not None and end_val is not None:
                    if isinstance(start_val, datetime):
                        start_val = _make_temporal_literal(start_val, column_dtype)
                    if isinstance(end_val, datetime):
                        end_val = _make_temporal_literal(end_val, column_dtype)
                    expr = column_expr.is_between(start_val, end_val, closed="both")
                elif start_val is not None:
                    if isinstance(start_val, datetime):
                        start_val = _make_temporal_literal(start_val, column_dtype)
                    expr = column_expr >= start_val
                elif end_val is not None:
                    if isinstance(end_val, datetime):
                        end_val = _make_temporal_literal(end_val, column_dtype)
                    expr = column_expr <= end_val
        else:
            expr = column_expr.str.contains(str(raw_value))

        if getattr(condition, "negate", False) and expr is not None:
            try:
                expr = expr.not_()
            except Exception:
                logger.debug("not_() failed for expression, falling back to ~ operator")
                expr = ~expr

        if expr is None:
            continue

        if filter_expr is None:
            filter_expr = expr
        else:
            filter_expr = (
                (filter_expr | expr) if logic == "or" else (filter_expr & expr)
            )

    if filter_expr is None:
        raise ValueError("No valid filter conditions provided")

    return filter_expr


@router.post("/nodes/{node_id}/filter", response_model=NodeOperationResponse)
async def filter_node(
    workspace_id: uuid.UUID,
    node_id: str,
    request: FilterRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a new child node by filtering the source node's data."""
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    workspace = require_workspace(user_id, workspace_id_str)
    node = workspace.nodes[node_id]
    lazy_data = node.data
    schema_map: dict[str, Any] = dict(lazy_data.collect_schema().items())
    filter_expr = _build_filter_expression(request, column_dtypes=schema_map)
    filtered_data = lazy_data.filter(filter_expr)
    new_node_name = request.new_node_name or f"{node.name}_filtered"
    new_node = _create_and_persist_child_node(
        workspace=workspace,
        data=filtered_data,
        name=new_node_name,
        operation=f"filter({node.name})",
        parents=[node],
        user_id=user_id,
        workspace_id=workspace_id_str,
    )
    return {
        "node_name": new_node.name,
        "node_id": new_node.id,
    }


@router.post("/nodes/{node_id}/filter/preview")
async def filter_preview(
    workspace_id: uuid.UUID,
    node_id: str,
    request: FilterRequest,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=500),
    current_user: dict = Depends(get_current_user),
) -> FilterPreviewResponse:
    """Preview the result of a filter operation on the source node."""
    user_id = current_user["id"]
    workspace = require_workspace(user_id, str(workspace_id))
    lazy_data = workspace.nodes[node_id].data

    try:
        schema_map: dict[str, Any] = dict(lazy_data.collect_schema().items())
        filter_expr = _build_filter_expression(request, column_dtypes=schema_map)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    try:
        filtered_lazy = lazy_data.filter(filter_expr)
        data_rows, columns, dtypes, pagination = _paginated_lazy_preview(
            filtered_lazy, page, page_size
        )
    except Exception as exc:
        raise InvalidInputError(str(exc)) from exc
    return FilterPreviewResponse(
        data=data_rows,
        columns=columns,
        dtypes=dtypes,
        pagination=pagination,
    )
