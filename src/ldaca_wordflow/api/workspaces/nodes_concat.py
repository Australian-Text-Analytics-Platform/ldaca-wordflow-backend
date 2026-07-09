"""Concat node endpoints: validate, align, concatenate, preview.

Used by:
- Frontend and API clients through the FastAPI concat routes.

Flow:
- Resolve workspace nodes, validate schema alignment, concatenate vertically,
- Return preview rows or persist a new concatenated child node.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, cast

import polars as pl
from fastapi import APIRouter, Depends, Query

from ...core.auth import get_current_user
from ...models import (
    ConcatPreviewRequest,
    ConcatRequest,
    FilterPreviewResponse,
    WorkspaceNodeInfo,
)
from .schema_filter import frontend_node_info
from ...core.exceptions import AppError, InternalServiceError, InvalidInputError
from .utils import (
    Node,
    _create_and_persist_child_node,
    _extract_lazy_schema,
    require_workspace,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
    tags=["nodes"],
)


def _get_concat_nodes(workspace, node_ids: list[str]) -> list[Node]:
    """Resolve node IDs to workspace Node objects, validating count and duplicates."""
    if not node_ids:
        raise InvalidInputError("At least two node IDs are required")
    nodes: list[Node] = []
    seen: set[str] = set()
    for raw_node_id in node_ids:
        node_id = raw_node_id.strip()
        if not node_id:
            continue
        if node_id in seen:
            raise InvalidInputError(f"Duplicate node id '{node_id}' provided",)
        node = workspace.nodes[node_id]
        nodes.append(node)
        seen.add(node_id)
    if len(nodes) < 2:
        raise InvalidInputError("At least two distinct nodes are required for concatenation",)
    return nodes


def _validate_and_align_concat_nodes(
    nodes: list[Node],
) -> tuple[list[pl.LazyFrame], list[str], dict[str, str]]:
    """Validate schema compatibility across nodes and align column selections."""
    lazy_frames: list[pl.LazyFrame] = [node.data for node in nodes]
    base_columns, base_dtypes = _extract_lazy_schema(lazy_frames[0])
    if not base_columns:
        raise InvalidInputError("Unable to determine schema for the first node.",)
    select_expr = [pl.col(column) for column in base_columns]
    aligned_frames: list[pl.LazyFrame] = [lazy_frames[0].select(select_expr)]

    for node, lazy_frame in zip(nodes[1:], lazy_frames[1:]):
        columns, dtypes = _extract_lazy_schema(lazy_frame)
        missing = [col for col in base_columns if col not in columns]
        extra = [col for col in columns if col not in base_columns]
        mismatched = [
            col
            for col in base_columns
            if col in dtypes and base_dtypes.get(col) != dtypes.get(col)
        ]

        if missing or extra or mismatched:
            detail_parts: list[str] = []
            if missing:
                detail_parts.append("missing columns: " + ", ".join(sorted(missing)))
            if extra:
                detail_parts.append("unexpected columns: " + ", ".join(sorted(extra)))
            if mismatched:
                mismatch_details = ", ".join(
                    f"{col} ({base_dtypes.get(col)} vs {dtypes.get(col)})"
                    for col in sorted(mismatched)
                )
                detail_parts.append(f"type mismatches: {mismatch_details}")
            detail = (
                "Schema mismatch for node '"
                + node.name
                + "': "
                + "; ".join(detail_parts)
            )
            raise InvalidInputError(detail)
        aligned_frames.append(lazy_frame.select(select_expr))

    return aligned_frames, base_columns, base_dtypes


def _calculate_concat_row_count(
    aligned_frames: list[pl.LazyFrame],
) -> int | None:
    """Calculate total row count for concat preview."""
    total = 0
    for lazy_frame in aligned_frames:
        try:
            count_df = cast(
                pl.DataFrame, lazy_frame.select(pl.len().alias("_len")).collect(),
            )
            total += int(count_df.to_series(0).item())
        except Exception:
            return None
    return total


def _derive_concat_node_name(nodes: list[Node], desired_name: str | None) -> str:
    """Derive a human-readable name for the concatenated node."""
    if desired_name:
        return desired_name
    labels = [node.name for node in nodes]
    if not labels:
        return "Stack Result"
    if len(labels) <= 3:
        label_str = ", ".join(labels)
    else:
        label_str = ", ".join(labels[:3]) + ", ..."
    return f"Stack({label_str})"


@router.post("/nodes/concat/preview", response_model=FilterPreviewResponse)
async def concat_nodes_preview(
    workspace_id: uuid.UUID,
    request: ConcatPreviewRequest,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=500),
    current_user: dict = Depends(get_current_user),
):
    """Preview the result of concatenating multiple nodes vertically."""
    user_id = current_user["id"]
    workspace = require_workspace(user_id, str(workspace_id))
    try:
        nodes = _get_concat_nodes(workspace, request.node_ids)
        aligned_frames, columns, dtypes = _validate_and_align_concat_nodes(nodes)
        concat_lazy = pl.concat(aligned_frames, how="vertical")
        if request.deduplicate:
            concat_lazy = concat_lazy.unique(maintain_order=True)
        total_rows = (
            None if request.deduplicate else _calculate_concat_row_count(aligned_frames)
        )

        normalized_page_size = page_size
        if total_rows is not None:
            total_pages = (
                math.ceil(total_rows / normalized_page_size) if total_rows else 0
            )
            normalized_page = min(max(page, 1), total_pages or 1)
            offset = (normalized_page - 1) * normalized_page_size if total_rows else 0
        else:
            total_pages = None
            normalized_page = max(page, 1)
            offset = (normalized_page - 1) * normalized_page_size

        preview_df = cast(
            pl.DataFrame,
            concat_lazy.slice(offset, normalized_page_size).collect(),
        )
        data_rows = preview_df.to_dicts()

        if total_rows is None:
            has_next = len(data_rows) == normalized_page_size
            inferred_total = (
                offset + len(data_rows) + (normalized_page_size if has_next else 0)
            )
            total_rows_value = inferred_total
            total_pages_value = max(1, normalized_page + (1 if has_next else 0))
        else:
            has_next = offset + normalized_page_size < total_rows
            total_rows_value = total_rows
            total_pages_value = total_pages

        pagination = {
            "page": normalized_page,
            "page_size": normalized_page_size,
            "total_rows": total_rows_value,
            "total_pages": total_pages_value,
            "has_next": has_next,
            "has_prev": normalized_page > 1 and (total_rows is None or total_rows > 0),
        }

        return {
            "data": data_rows,
            "columns": columns,
            "dtypes": dtypes,
            "pagination": pagination,
        }
    except AppError:
        raise
    except Exception as exc:
        raise InternalServiceError(str(exc)) from exc
@router.post("/nodes/concat", response_model=WorkspaceNodeInfo)
async def concat_nodes(
    workspace_id: uuid.UUID,
    request: ConcatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a new child node by concatenating multiple nodes vertically."""
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    workspace = require_workspace(user_id, workspace_id_str)
    try:
        nodes = _get_concat_nodes(workspace, request.node_ids)
        aligned_frames, _, _ = _validate_and_align_concat_nodes(nodes)
        concat_lazy = pl.concat(aligned_frames, how="vertical")
        if request.deduplicate:
            concat_lazy = concat_lazy.unique(maintain_order=True)
        node_name = _derive_concat_node_name(nodes, request.new_node_name)
        labels = [node.name for node in nodes]
        parent_nodes: list = list(nodes)
        if len(labels) > 3:
            operation_args = ", ".join(labels[:3]) + ", ..."
        else:
            operation_args = ", ".join(labels)
        op_name = "concat_unique" if request.deduplicate else "concat"
        operation_label = f"{op_name}({operation_args})"
        new_node = _create_and_persist_child_node(
            workspace=workspace,
            data=concat_lazy,
            name=node_name,
            operation=operation_label,
            parents=parent_nodes,
            user_id=user_id,
            workspace_id=workspace_id_str,
        )
        return frontend_node_info(new_node)
    except AppError:
        raise
    except Exception as exc:
        raise InternalServiceError(str(exc)) from exc
