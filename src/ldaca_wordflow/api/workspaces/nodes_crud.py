"""CRUD node endpoints: get, update, delete, clone, column operations.

Used by:
- Frontend and API clients through the FastAPI node CRUD routes.

Flow:
- Resolve workspace nodes for read, update, delete, clone operations,
- Handle column statistics, descriptions, unique values, tokenization settings.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, cast

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Query

from ...core.auth import get_current_user
from ...core.polars_operations import get_operations_for_dtype
from ...core.tokenization import tokenise_column
from ...core.utils import stringify_unsafe_integers
from ...models import (
    ColumnDescribeResponse,
    ColumnOperationsResponse,
    ColumnUniqueValuesResponse,
    NodeActionResponse,
    NodeDataResponse,
    NodeDocumentColumnUpdateRequest,
    NodeQueryPlanResponse,
    NodeShapeResponse,
    NodeTokenizationPreferenceRequest,
    TokenizerModelsResponse,
    WorkspaceNodeInfo,
)
from polars_text.models import PREDEFINED_MODELS, predefined_model_records

from .schema_filter import frontend_node_info, project_visible
from ...core.docworkspace_data_types import TM_DISTRIBUTION_POLARS_DTYPE
from ...core.exceptions import InternalServiceError, InvalidInputError, NodeNotFoundError, NotFoundError, ValidationError
from .utils import (
    Node,
    _is_string_list_dtype,
    _propagated_tokenization,
    _serialize_column_scalar,
    _validate_existing_column,
    require_current_workspace,
    require_current_workspace_id,
    update_workspace,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspaces", tags=["nodes"])


def _normalise_iso6391_language_code(code: str | None) -> str | None:
    """Normalize an ISO 639-1 language code, raising 422 on invalid input."""
    import re

    if code is None:
        return None
    trimmed = code.strip().lower()
    if not trimmed:
        return None
    primary = re.split(r"[-_]", trimmed, maxsplit=1)[0]
    if not re.fullmatch(r"[a-z]{2}", primary):
        raise ValidationError("language must be an ISO 639-1 two-letter code",)
    return primary


# ── Tokenizer models ────────────────────────────────────────────────────


@router.get("/tokenizer-models", response_model=TokenizerModelsResponse)
async def get_tokenizer_models(
    _current_user: dict = Depends(get_current_user),
):
    """Return available tokenizer model records."""
    return {"models": predefined_model_records()}


# ── Node CRUD ───────────────────────────────────────────────────────────


@router.get("/nodes/{node_id}", response_model=WorkspaceNodeInfo)
async def get_node_info(
    node_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return workspace node metadata."""
    user_id = current_user["id"]
    ws = require_current_workspace(user_id)
    return frontend_node_info(ws.nodes[node_id])


@router.get("/nodes/{node_id}/data", response_model=NodeDataResponse)
async def get_node_data(
    node_id: str,
    page: int = 1,
    page_size: int = 20,
    sort_by: str | None = None,
    descending: bool = False,
    filter_column: str | None = None,
    filter_value: str | None = None,
    filter_op: str = "contains",
    current_user: dict = Depends(get_current_user),
):
    """Return paginated node data rows."""
    user_id = current_user["id"]
    lf = project_visible(require_current_workspace(user_id).nodes[node_id].data)
    schema = {col: str(dtype) for col, dtype in lf.collect_schema().items()}
    columns = list(schema.keys())

    if filter_column and filter_value is not None and filter_column in columns:
        dtype_str = schema.get(filter_column, "")
        is_string_like = any(
            t in dtype_str.lower() for t in ("utf8", "string", "categorical")
        )
        if is_string_like:
            col_expr = pl.col(filter_column).cast(pl.Utf8)
            val = filter_value
            if filter_op == "eq":
                lf = lf.filter(col_expr == val)
            elif filter_op == "startswith":
                lf = lf.filter(col_expr.str.starts_with(val))
            elif filter_op == "endswith":
                lf = lf.filter(col_expr.str.ends_with(val))
            else:
                lf = lf.filter(col_expr.str.contains(val, literal=True))

    total_rows: int = cast(pl.DataFrame, lf.select(pl.len()).collect()).item()

    if sort_by and sort_by in columns:
        lf = lf.sort(sort_by, descending=descending, nulls_last=True)

    start_idx = (page - 1) * page_size
    page_df = cast(pl.DataFrame, lf.slice(start_idx, page_size).collect())

    return {
        "data": stringify_unsafe_integers(page_df.to_dicts()),
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": (total_rows + page_size - 1) // page_size,
            "has_next": start_idx + page_size < total_rows,
            "has_prev": page > 1,
        },
        "columns": columns,
        "dtypes": schema,
        "sorting": {
            "sort_by": sort_by if sort_by and sort_by in columns else None,
            "descending": descending,
        },
        "filtering": {
            "column": filter_column
            if filter_column and filter_column in columns
            else None,
            "value": filter_value,
            "op": filter_op,
        },
    }


@router.get("/nodes/{node_id}/shape", response_model=NodeShapeResponse)
async def get_node_shape(
    node_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return the row x column shape of a workspace node."""
    user_id = current_user["id"]
    return {"shape": require_current_workspace(user_id).nodes[node_id].shape}


@router.delete("/nodes/{node_id}", response_model=NodeActionResponse)
async def delete_node(node_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a node from the workspace."""
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    workspace_id = workspace.id
    success = workspace.remove_node(node_id)
    if success:
        update_workspace(user_id, workspace_id)
    if not success:
        raise NodeNotFoundError("Node not found")
    return {"state": "successful", "message": "Node deleted successfully"}


@router.put("/nodes/{node_id}/name", response_model=WorkspaceNodeInfo)
async def update_node_name(
    node_id: str,
    new_name: str,
    current_user: dict = Depends(get_current_user),
):
    """Rename a workspace node."""
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    workspace_id = workspace.id
    node = workspace.nodes[node_id]
    node.name = new_name
    update_workspace(user_id, workspace_id, best_effort=True)
    try:
        return frontend_node_info(node)
    except Exception:
        logger.debug("node.info() failed for %s, returning minimal dict", node_id)
        return {"id": getattr(node, "id", node_id), "name": new_name}


@router.post("/nodes/{node_id}/clone", response_model=WorkspaceNodeInfo)
async def clone_node(
    node_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Clone a workspace node (deep-copy its LazyFrame)."""
    user_id = current_user["id"]
    workspace = require_current_workspace(user_id)
    workspace_id = workspace.id
    node = workspace.nodes[node_id]

    def _unique_clone_name(original: str) -> str:
        base = original or node_id
        candidate = f"{base}_clone"
        existing = {getattr(n, "name", None) for n in workspace.nodes.values()}
        if candidate not in existing:
            return candidate
        suffix = 2
        while f"{base}_clone_{suffix}" in existing:
            suffix += 1
        return f"{base}_clone_{suffix}"

    try:
        source_lazy = node.data
        cloned_lazy = source_lazy.clone()
        new_name = _unique_clone_name(getattr(node, "name", node_id))
        new_node = Node(
            data=cloned_lazy,
            name=new_name,
            workspace=workspace,
            operation=f"clone({getattr(node, 'name', node_id)})",
            parents=[node],
            tokenization=_propagated_tokenization(node, cloned_lazy),
        )
        workspace.add_node(new_node)
        update_workspace(user_id, workspace_id)
        try:
            return frontend_node_info(new_node)
        except Exception:
            logger.debug("new_node.info() failed after clone, returning minimal dict")
            return {"id": getattr(new_node, "id", None), "name": new_name}
    except Exception as exc:
        raise InternalServiceError(str(exc)) from exc
@router.put("/nodes/{node_id}/document-column", response_model=WorkspaceNodeInfo)
async def set_node_document_column(
    node_id: str,
    request: NodeDocumentColumnUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Set or clear the document column on a workspace node."""
    user_id = current_user["id"]
    workspace_id = require_current_workspace_id(user_id)
    ws = require_current_workspace(user_id)
    node = ws.nodes[node_id]
    document_column = (request.document_column or "").strip()

    if document_column:
        _validate_existing_column(node, document_column)
        node.document = document_column
    else:
        node.document = None

    update_workspace(user_id, workspace_id, best_effort=True)
    return frontend_node_info(node)


@router.put(
    "/nodes/{node_id}/tokenization-preference", response_model=WorkspaceNodeInfo
)
async def set_node_tokenization_preference(
    node_id: str,
    request: NodeTokenizationPreferenceRequest,
    current_user: dict = Depends(get_current_user),
):
    """Set or clear a tokenization preference on a workspace node."""
    user_id = current_user["id"]
    workspace_id = require_current_workspace_id(user_id)
    ws = require_current_workspace(user_id)
    node = ws.nodes[node_id]
    source_column = request.source_column.strip()
    model = (request.model or "").strip()

    if not source_column:
        raise ValidationError("source_column is required")
    if not model:
        _validate_existing_column(node, source_column)
        node.unregister_tokenization(source_column)
        update_workspace(user_id, workspace_id, best_effort=True)
        return frontend_node_info(node)
    if model not in PREDEFINED_MODELS:
        raise InvalidInputError(f"Unknown tokenizer model: {model}")
    language = _normalise_iso6391_language_code(request.language)
    try:
        tokenise_column(
            node,
            source_column=source_column,
            model=model,
            language=language,
        )
    except KeyError as exc:
        raise InvalidInputError(str(exc)) from exc
    update_workspace(user_id, workspace_id, best_effort=True)
    return frontend_node_info(node)


@router.get("/nodes/{node_id}/query-plan", response_model=NodeQueryPlanResponse)
async def get_node_query_plan(
    node_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return the Polars query plan for a workspace node."""
    user_id = current_user["id"]
    ws = require_current_workspace(user_id)
    lazyframe = ws.nodes[node_id].data
    plan = lazyframe.explain(format="tree")
    return {"plan": plan}


# ── Column operations ───────────────────────────────────────────────────


@router.get(
    "/nodes/{node_id}/columns/{column_name}/unique",
    response_model=ColumnUniqueValuesResponse,
)
async def get_column_unique_values(
    node_id: str,
    column_name: str,
    current_user: dict = Depends(get_current_user),
):
    """Return unique values for a column."""
    user_id = current_user["id"]
    try:
        lazyframe = require_current_workspace(user_id).nodes[node_id].data
        schema = lazyframe.collect_schema()
        schema_map: dict[str, Any] = dict(schema.items())
        if schema_map.get(column_name) == TM_DISTRIBUTION_POLARS_DTYPE:
            # Topic-distribution column: return the distinct topic ids so the
            # filter UI can offer a topic dropdown instead of a free-text id.
            topic_df = cast(
                pl.DataFrame,
                lazyframe.select(
                    pl.col(column_name).explode().struct.field("topic_id").alias("topic_id")
                )
                .unique(maintain_order=True)
                .collect(),
            )
            topic_ids = sorted(
                {int(value) for value in topic_df.get_column("topic_id").to_list() if value is not None}
            )
            return {
                "column_name": column_name,
                "unique_count": len(topic_ids),
                "unique_values": topic_ids,
                "has_null": False,
            }

        if _is_string_list_dtype(schema_map.get(column_name)):
            unique_df = cast(
                pl.DataFrame,
                lazyframe.select(pl.col(column_name).explode().alias(column_name))
                .unique(maintain_order=True)
                .collect(),
            )
            raw_values = unique_df.get_column(column_name).to_list()
            has_null = any(value is None for value in raw_values)
            deduped_values = [str(value) for value in raw_values if value is not None]
            return {
                "column_name": column_name,
                "unique_count": len(deduped_values) + (1 if has_null else 0),
                "unique_values": deduped_values,
                "has_null": has_null,
            }

        unique_df = cast(
            pl.DataFrame,
            lazyframe.select(pl.col(column_name).alias(column_name))
            .unique(maintain_order=True)
            .collect(),
        )
        raw_values = unique_df.get_column(column_name).to_list()
        has_null = any(value is None for value in raw_values)
        non_null_values = [
            _serialize_column_scalar(value) for value in raw_values if value is not None
        ]
        return {
            "column_name": column_name,
            "unique_count": len(raw_values),
            "unique_values": non_null_values,
            "has_null": has_null,
        }
    except Exception as exc:
        raise InternalServiceError(str(exc)) from exc
@router.get(
    "/nodes/{node_id}/columns/{column_name}/describe",
    response_model=ColumnDescribeResponse,
)
async def describe_column(
    node_id: str,
    column_name: str,
    current_user: dict = Depends(get_current_user),
):
    """Get descriptive statistics for a column using Polars describe."""
    user_id = current_user["id"]

    try:
        lazyframe = require_current_workspace(user_id).nodes[node_id].data
        df = cast(pl.DataFrame, lazyframe.collect())

        column_dtype = df.schema[column_name]
        is_datetime_column = column_dtype in (
            pl.Datetime,
            pl.Datetime("ms"),
            pl.Datetime("us"),
            pl.Datetime("ns"),
        )

        desc_df = df.select(column_name).describe(interpolation="nearest")

        desc_dict = {}
        for row in desc_df.iter_rows(named=True):
            stat_name = row.get("statistic") or row.get("describe")
            if stat_name:
                desc_dict[stat_name] = row[column_name]

        def serialize_value(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val.isoformat()
            if is_datetime_column and isinstance(val, str) and val != "null":
                try:
                    dt = datetime.fromisoformat(val.replace(" ", "T"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.isoformat()
                except (ValueError, AttributeError):
                    return val
            try:
                return float(val)
            except (TypeError, ValueError):
                return val

        return ColumnDescribeResponse(
            column_name=column_name,
            count=int(desc_dict.get("count", 0))
            if desc_dict.get("count") is not None
            else None,
            null_count=int(desc_dict.get("null_count", 0))
            if desc_dict.get("null_count") is not None
            else None,
            mean=serialize_value(desc_dict.get("mean")),
            std=serialize_value(desc_dict.get("std")),
            min=serialize_value(desc_dict.get("min")),
            percentile_25=serialize_value(desc_dict.get("25%")),
            median=serialize_value(desc_dict.get("50%")),
            percentile_75=serialize_value(desc_dict.get("75%")),
            max=serialize_value(desc_dict.get("max")),
        )
    except Exception as exc:
        raise InternalServiceError(str(exc)) from exc
@router.get(
    "/nodes/{node_id}/columns/{column_name}/operations",
    response_model=ColumnOperationsResponse,
)
async def column_operations(
    node_id: str,
    column_name: str,
    current_user: dict = Depends(get_current_user),
):
    """Return available no-arg Polars operations for a column, filtered by dtype."""
    user_id = current_user["id"]
    ws = require_current_workspace(user_id)
    node = ws.nodes[node_id]
    schema = dict(node.data.collect_schema().items())
    dtype = schema.get(column_name)
    if dtype is None:
        raise NotFoundError(f"Column '{column_name}' not found")
    return {"operations": get_operations_for_dtype(dtype)}
