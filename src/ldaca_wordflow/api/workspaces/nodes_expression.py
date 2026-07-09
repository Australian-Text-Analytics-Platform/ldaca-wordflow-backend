"""Unified Polars expression endpoints: execute arbitrary Polars expressions.

Used by:
- Frontend and API clients through the FastAPI expression routes.

Flow:
- Validate and execute user-provided Polars expression code,
- Return preview rows or persist the result as a new child node (or mutate in-place).
"""

from __future__ import annotations

import logging
import uuid
from typing import cast

import polars as pl
from fastapi import APIRouter, Depends, Query

from ...core.auth import get_current_user
from ...core.polars_expr_validator import (
    PolarsExprValidationError,
    ValidationResult,
    validate_polars_expr_code,
)
from ...core.utils import stringify_unsafe_integers
from ...core.exceptions import InvalidInputError
from ...models import (
    FilterPreviewResponse,
    PolarsExpressionApplyResponse,
    PolarsExpressionContext,
    PolarsExpressionRequest,
)
from .utils import (
    _create_and_persist_child_node,
    _paginated_lazy_preview,
    require_workspace,
    update_workspace,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
    tags=["nodes"],
)


def _split_top_level_commas(code: str) -> list[str]:
    """Split code at commas that are not inside parentheses, brackets, braces, or strings.

    Called by:
    - _exec_polars_expr.
    """
    segments: list[str] = []
    depth = 0
    current: list[str] = []
    in_string: str | None = None
    escape = False

    for char in code:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\":
            current.append(char)
            escape = True
            continue
        if in_string:
            current.append(char)
            if char == in_string:
                in_string = None
            continue
        if char in ('"', "'"):
            in_string = char
            current.append(char)
            continue
        if char in ("(", "[", "{"):
            depth += 1
        elif char in (")", "]", "}"):
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            segments.append("".join(current).strip())
            current = []
            continue
        current.append(char)

    remainder = "".join(current).strip()
    if remainder:
        segments.append(remainder)
    return segments


def _exec_polars_expr(code: str) -> list[pl.Expr]:
    """Execute polars expression code string(s) and return resulting pl.Expr(s).

    Supports:
    * Plain expression: ``pl.col("a")``
    * Assignment: ``name = pl.col("a")`` → ``pl.col("a").alias(name)``

    Called by:
    - _apply_expression_context.
    """
    segments = _split_top_level_commas(code.strip())
    if not segments:
        raise ValueError("Expression code cannot be empty")

    exprs: list[pl.Expr] = []
    for segment in segments:
        vr: ValidationResult = validate_polars_expr_code(segment)

        if vr.mode == "assign":
            expr_source = segment.split("=", 1)[1].strip()
        else:
            expr_source = segment

        ns: dict[str, object] = {"pl": pl}
        exec(f"_result = {expr_source}", {"__builtins__": {}}, ns)  # noqa: S102
        result = ns["_result"]

        if not isinstance(result, pl.Expr):
            raise ValueError(f"Expected pl.Expr, got {type(result).__name__}")

        if vr.mode == "assign" and vr.alias:
            result = result.alias(vr.alias)

        exprs.append(result)

    return exprs


def _apply_expression_context(
    lazy: pl.LazyFrame,
    request: PolarsExpressionRequest,
) -> pl.LazyFrame:
    """Apply the requested polars context + expressions to a LazyFrame.

    Called by:
    - polars_expression_preview, polars_expression_apply.
    """
    context = request.context
    items = request.expressions

    if context == PolarsExpressionContext.filter:
        exprs = _exec_polars_expr(items[0].code)
        return lazy.filter(exprs[0])

    if context == PolarsExpressionContext.with_columns:
        exprs = [e for item in items for e in _exec_polars_expr(item.code)]
        return lazy.with_columns(exprs)

    if context == PolarsExpressionContext.select:
        exprs = [e for item in items for e in _exec_polars_expr(item.code)]
        return lazy.select(exprs)

    if context == PolarsExpressionContext.sort:
        pairs = [
            (e, bool(item.descending))
            for item in items
            for e in _exec_polars_expr(item.code)
        ]
        by = [p[0] for p in pairs]
        descending = [p[1] for p in pairs]
        return lazy.sort(by, descending=descending)

    if context == PolarsExpressionContext.group_by_agg:
        keys = [
            e for k in (request.group_by_keys or []) for e in _exec_polars_expr(k.code)
        ]
        aggs = [e for item in items for e in _exec_polars_expr(item.code)]
        return lazy.group_by(keys).agg(aggs)

    raise InvalidInputError(f"Unknown context: {context}")
@router.post("/nodes/{node_id}/expression/preview")
async def polars_expression_preview(
    workspace_id: uuid.UUID,
    node_id: str,
    request: PolarsExpressionRequest,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=500),
    current_user: dict = Depends(get_current_user),
) -> FilterPreviewResponse:
    """Preview a Polars expression applied to the source node."""
    user_id = current_user["id"]
    workspace = require_workspace(user_id, str(workspace_id))
    lazy_data = workspace.nodes[node_id].data

    try:
        result_lazy = _apply_expression_context(lazy_data, request)
    except PolarsExprValidationError as exc:
        raise InvalidInputError(str(exc)) from exc
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    data_rows, columns, dtypes, pagination = _paginated_lazy_preview(
        result_lazy, page, page_size
    )
    data = stringify_unsafe_integers(data_rows)

    return FilterPreviewResponse(
        data=data,
        columns=columns,
        dtypes=dtypes,
        pagination=pagination,
    )


@router.post("/nodes/{node_id}/expression/apply")
async def polars_expression_apply(
    workspace_id: uuid.UUID,
    node_id: str,
    request: PolarsExpressionRequest,
    current_user: dict = Depends(get_current_user),
) -> PolarsExpressionApplyResponse:
    """Apply a Polars expression and persist the result."""
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    workspace = require_workspace(user_id, workspace_id_str)
    node = workspace.nodes[node_id]

    try:
        result_lazy = _apply_expression_context(node.data, request)
    except PolarsExprValidationError as exc:
        raise InvalidInputError(str(exc)) from exc
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    if (
        request.context == PolarsExpressionContext.with_columns
        and not request.new_node_name
    ):
        node.data = result_lazy
        update_workspace(user_id, workspace_id_str, workspace)
        return PolarsExpressionApplyResponse(node_id=node_id, node_name=node.name)

    new_node_name = request.new_node_name or f"{node.name}_{request.context}"
    new_node = _create_and_persist_child_node(
        workspace=workspace,
        data=result_lazy,
        name=new_node_name,
        operation=f"expression({request.context}, {node.name})",
        parents=[node],
        user_id=user_id,
        workspace_id=workspace_id_str,
    )
    return PolarsExpressionApplyResponse(node_id=new_node.id, node_name=new_node.name)
