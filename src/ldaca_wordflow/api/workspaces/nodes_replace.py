"""Replace node endpoints: build regex-replace expressions, preview, and apply.

Used by:
- Frontend and API clients through the FastAPI replace routes.

Flow:
- Resolve workspace and node, build Polars replace expressions,
- Return preview rows or persist the updated node data.
"""

from __future__ import annotations

import math
import re
import uuid
from typing import Any, cast

import polars as pl
from fastapi import APIRouter, Depends, Query

from ...core.auth import get_current_user
from ...models import (
    FilterPreviewResponse,
    PaginationInfo,
    ReplaceApplyResponse,
    ReplaceRequest,
)
from .utils import (
    _paginated_lazy_preview,
    require_workspace,
    update_workspace,
)
from ...core.exceptions import InternalServiceError, InvalidInputError

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
    tags=["nodes"],
)


def _sanitize_column_alias(label: str) -> str:
    """Sanitize a column alias/label for use as a Polars column name."""
    sanitized = re.sub(r"\s+", " ", label or "").strip()
    if not sanitized:
        return "computed_column"
    return sanitized[:120]


def _resolve_replace_column_name(request: ReplaceRequest) -> str:
    """Resolve the final output column name for replace operations."""
    candidate = (request.output_column_name or request.source_column or "").strip()
    return _sanitize_column_alias(candidate)


def _build_replace_expression(request: ReplaceRequest) -> tuple[str, pl.Expr]:
    """Build a Polars replace expression from the request.

    Called by:
    - replace_preview, replace_apply.
    """
    source_column = request.source_column.strip()
    column_name = _resolve_replace_column_name(request)

    col_expr = pl.col(source_column)
    connector = request.connector

    if request.mode == "extract":
        extracted = col_expr.str.extract_all(request.pattern)
        if request.count == "first":
            n = request.n if request.n is not None else 1
            extracted = extracted.list.head(n)
        joined = extracted.list.join(connector)
        expr = pl.when(extracted.list.len() > 0).then(joined).otherwise(pl.lit(None))
    else:
        if request.count == "all":
            expr = col_expr.str.replace_all(request.pattern, request.replacement)
        else:
            n = request.n if request.n is not None else 1
            expr = col_expr.str.replace(request.pattern, request.replacement, n=n)

    return column_name, expr.alias(column_name)


@router.post("/nodes/{node_id}/replace/preview", response_model=FilterPreviewResponse)
async def replace_preview(
    workspace_id: uuid.UUID,
    node_id: str,
    request: ReplaceRequest,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=500),
    current_user: dict = Depends(get_current_user),
):
    """Preview a regex replace operation on the source node."""
    user_id = current_user["id"]
    workspace = require_workspace(user_id, str(workspace_id))
    lazy_data = workspace.nodes[node_id].data

    try:
        _, replace_expr = _build_replace_expression(request)
        replaced_lazy = lazy_data.with_columns(replace_expr)
        data_rows, columns, dtypes, pagination = _paginated_lazy_preview(
            replaced_lazy, page, page_size
        )
    except Exception as exc:
        raise InvalidInputError(str(exc)) from exc
    return FilterPreviewResponse(
        data=data_rows,
        columns=columns,
        dtypes=dtypes,
        pagination=pagination,
    )


@router.post("/nodes/{node_id}/replace", response_model=ReplaceApplyResponse)
async def replace_apply(
    workspace_id: uuid.UUID,
    node_id: str,
    request: ReplaceRequest,
    current_user: dict = Depends(get_current_user),
):
    """Apply a regex replace operation and persist the updated node."""
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    workspace = require_workspace(user_id, workspace_id_str)
    node = workspace.nodes[node_id]
    dtype_str: str | None = None

    try:
        lazy_data = node.data
        column_name, replace_expr = _build_replace_expression(request)
        updated_data = lazy_data.with_columns(replace_expr)
        updated_schema = dict(updated_data.collect_schema().items())
        dtype = updated_schema.get(column_name)
        if dtype is not None:
            dtype_str = str(dtype)
    except Exception as exc:
        raise InvalidInputError(str(exc)) from exc
    try:
        node.data = updated_data
        update_workspace(user_id, workspace_id_str, workspace)
    except Exception as exc:
        raise InternalServiceError(str(exc)) from exc
    return ReplaceApplyResponse(
        state="successful",
        node_id=node_id,
        column_name=column_name,
        dtype=dtype_str,
        message=f"Updated column '{column_name}' with regex replacement",
    )
