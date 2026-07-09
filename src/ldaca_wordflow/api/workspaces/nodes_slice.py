"""Slice node endpoints: build slice/sample/shuffle LazyFrame, preview, and apply.

Used by:
- Frontend and API clients through the FastAPI slice routes.

Flow:
- Resolve workspace and node, build a sliced/sampled/shuffled LazyFrame,
- Return preview rows or persist a new sliced child node.
"""

from __future__ import annotations

import math
import uuid
from typing import cast

import polars as pl
from fastapi import APIRouter, Depends, Query

from ...core.auth import get_current_user
from ...core.exceptions import InvalidInputError, ValidationError
from ...models.analysis_common import PaginationInfo
from ...models.nodes import (
    FilterPreviewResponse,
    NodeOperationResponse,
    SliceRequest,
)
from .utils import (
    _create_and_persist_child_node,
    _extract_lazy_schema,
    _paginated_lazy_preview,
    require_workspace,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
    tags=["nodes"],
)


def _build_slice_or_sample_lazy(
    lazy_data: pl.LazyFrame,
    node_name: str,
    request: SliceRequest,
) -> tuple[pl.LazyFrame, str, str]:
    """Build a sliced, sampled, or shuffled LazyFrame.

    Called by:
    - slice_node, slice_preview.
    """
    if request.mode == "shuffle":
        seed_args = (
            f", seed={request.random_seed}" if request.random_seed is not None else ""
        )
        shuffle_indices = pl.int_range(pl.len()).sample(
            fraction=1.0,
            shuffle=True,
            seed=request.random_seed,
        )
        shuffled_data = lazy_data.select(pl.all().gather(shuffle_indices))
        return (
            shuffled_data,
            f"{node_name}_shuffled",
            f"shuffle({node_name}{seed_args})",
        )

    if request.mode == "random_sample":
        if request.sample_size is None:
            raise ValidationError("sample_size is required when mode is 'random_sample'",)
        if request.sample_size < 1:
            sample_indices = pl.int_range(pl.len()).sample(
                fraction=request.sample_size,
                seed=request.random_seed,
            )
            sample_args = f"fraction={request.sample_size}"
        else:
            n = int(request.sample_size)
            sample_indices = pl.int_range(pl.len()).sample(
                n=n,
                seed=request.random_seed,
            )
            sample_args = f"n={n}"
        if request.random_seed is not None:
            sample_args = f"{sample_args}, seed={request.random_seed}"
        sampled_data = lazy_data.select(pl.all().gather(sample_indices))
        return (
            sampled_data,
            f"{node_name}_sampled",
            f"sample({node_name}, {sample_args})",
        )

    offset = int(request.offset or 0)
    length = request.length
    sliced_data = lazy_data.slice(offset, length)
    slice_args = f"offset={offset}"
    if length is not None:
        slice_args = f"{slice_args}, length={length}"
    return sliced_data, f"{node_name}_sliced", f"slice({node_name}, {slice_args})"


@router.post("/nodes/{node_id}/slice", response_model=NodeOperationResponse)
async def slice_node(
    workspace_id: uuid.UUID,
    node_id: str,
    request: SliceRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a new child node by slicing/sampling/shuffling the source."""
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    workspace = require_workspace(user_id, workspace_id_str)
    node = workspace.nodes[node_id]
    output_data, default_node_name, operation = _build_slice_or_sample_lazy(
        node.data,
        node.name,
        request,
    )
    new_node_name = request.new_node_name or default_node_name
    new_node = _create_and_persist_child_node(
        workspace=workspace,
        data=output_data,
        name=new_node_name,
        operation=operation,
        parents=[node],
        user_id=user_id,
        workspace_id=workspace_id_str,
    )
    return {
        "node_name": new_node.name,
        "node_id": new_node.id,
    }


@router.post("/nodes/{node_id}/slice/preview")
async def slice_preview(
    workspace_id: uuid.UUID,
    node_id: str,
    request: SliceRequest,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=500),
    current_user: dict = Depends(get_current_user),
) -> FilterPreviewResponse:
    """Preview the result of a slice/sample/shuffle on the source node."""
    user_id = current_user["id"]
    workspace = require_workspace(user_id, str(workspace_id))

    try:
        preview_lazy, _default_node_name, _operation = _build_slice_or_sample_lazy(
            workspace.nodes[node_id].data,
            workspace.nodes[node_id].name,
            request,
        )
        data_rows, columns, dtypes, pagination = _paginated_lazy_preview(
            preview_lazy, page, page_size
        )
    except Exception as exc:
        raise InvalidInputError(str(exc)) from exc
    return FilterPreviewResponse(
        data=data_rows,
        columns=columns,
        dtypes=dtypes,
        pagination=pagination,
    )
