"""AI Annotation analysis endpoints.

Includes:
    - POST /workspaces/ai-annotation/models  (list models for a given endpoint)
    - POST /workspaces/ai-annotation          (run classification)
    - DELETE /workspaces/ai-annotation        (clear results)
    - GET  /workspaces/ai-annotation/tasks/{task_id}/request
    - GET  /workspaces/ai-annotation/tasks/{task_id}/result
    - POST /workspaces/ai-annotation/tasks/{task_id}/result
    - POST /workspaces/nodes/{node_id}/ai-annotation/detach
    - POST /workspaces/nodes/{node_id}/ai-annotation/save
    - GET  /workspaces/nodes/{node_id}/ai-annotation/providers
    - GET  /workspaces/nodes/{node_id}/ai-annotation/categories

Used by:
- FastAPI workspace analysis routers, frontend analysis features, and backend tests because they need this unit's "AI Annotation analysis endpoints" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers validate model/category requests, node selections, and task records.
- Helpers submit classification work, page stored results, and attach/detach generated columns.
- Responses return model lists, task metadata, paged annotations, or saved workspace updates.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Query

from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....analysis.results import GenericAnalysisResult
from ....core.auth import get_current_user
from ....core.exceptions import (
    InternalServiceError,
    InvalidInputError,
    NoActiveWorkspaceError,
    TaskNotFoundError,
    WorkspaceNotFoundError,
)
from ....core.workspace import workspace_manager
from ....models import (
    AiAnnotationCategoriesResponse,
    AiAnnotationDetachRequest,
    AiAnnotationDetachResponse,
    AiAnnotationModelsRequest,
    AiAnnotationModelsResponse,
    AiAnnotationProvidersResponse,
    AiAnnotationRequest,
    AiAnnotationResponse,
    AiAnnotationResultQuery,
    AiAnnotationSaveRequest,
    AiAnnotationSaveResponse,
    AnalysisClearResponse,
)
from ..utils import update_workspace
from .ai_annotation_core import classify_texts, list_models

router = APIRouter(prefix="/workspaces", tags=["ai-annotation"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _paginate_results(
    items: List[Dict[str, Any]],
    page: int = 1,
    page_size: int = 20,
    sort_by: Optional[str] = None,
    descending: bool = True,
    total_source_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Support AI annotation routes with a paginate results helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support AI annotation routes with a paginate results helper" behavior.
    """

    total = total_source_rows if total_source_rows is not None else len(items)
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))

    if sort_by:
        items = sorted(
            items,
            key=lambda r: (r.get(sort_by) is None, r.get(sort_by, "")),
            reverse=descending,
        )

    # Items may carry a row_index from on-demand classification across pages.
    # Filter by the row_index range that corresponds to the requested page.
    start = (page - 1) * page_size
    end = start + page_size
    if any("row_index" in item for item in items):
        page_items = [
            item for item in items if start <= item.get("row_index", -1) < end
        ]
    else:
        page_items = items[start:end]

    return {
        "data": page_items,
        "columns": list(page_items[0].keys()) if page_items else [],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_source_rows": total,
            "total_source_pages": total_pages,
            "result_count": len(page_items),
            "has_next": page < total_pages,
            "has_prev": page > 1,
        },
        "sorting": {
            "sort_by": sort_by,
            "descending": descending,
        },
    }


async def _build_response(
    request: AiAnnotationRequest,
    task_id: str,
    page: int,
    page_size: int,
    user_id: str,
    sort_by: Optional[str] = None,
    descending: bool = True,
) -> Dict[str, Any]:
    """Compute and return only the requested page for AI annotation.

    Results are generated on demand for the requested page and are not written
    back into the stored task.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Compute and return only the requested page for AI annotation" behavior.
    """
    ws = workspace_manager.get_current_workspace(user_id)
    if ws is None:
        return {
            "state": "failed",
            "message": "No active workspace selected",
            "data": None,
        }

    data: Dict[str, Any] = {}
    start = (page - 1) * page_size

    classes = [c.model_dump() for c in request.classes]
    examples = [e.model_dump() for e in request.examples] if request.examples else None

    for node_id in request.node_ids:
        items: List[Dict[str, Any]] = []

        node = ws.nodes[node_id]
        node_data = node.data

        column_name = request.node_columns[node_id]

        total_rows = cast(pl.DataFrame, node_data.select(pl.len()).collect()).item()

        sel_df = node_data.select(pl.col(column_name)).slice(start, page_size).collect()
        texts = [
            str(v) if v is not None else ""
            for v in sel_df.get_column(column_name).to_list()
        ]

        results = await classify_texts(
            texts=texts,
            classes=classes,
            examples=examples,
            model=request.model,
            api_key=request.api_key,
            base_url=request.base_url,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
            batch_size=page_size,
            text_column_name=column_name,
        )

        for r in results:
            r["row_index"] = start + r["row_index"]
        items.extend(results)

        data[node_id] = _paginate_results(
            items,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            descending=descending,
            total_source_rows=total_rows,
        )

    return {
        "state": "successful",
        "message": "AI annotation results",
        "data": data,
        "analysis_params": {
            "model": request.model,
            "base_url": request.base_url,
            "node_columns": request.node_columns,
        },
        "metadata": {
            "task_id": task_id,
            "annotation_columns": ["classification"],
        },
    }


def _workspace_data_dir(user_id: str, workspace_id: str) -> Path:
    """Support AI annotation routes with a workspace data dir helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support AI annotation routes with a workspace data dir helper" behavior.
    """

    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
    if workspace_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    workspace_data_dir = workspace_dir / "data"
    workspace_data_dir.mkdir(parents=True, exist_ok=True)
    return workspace_data_dir


def _distinct_annotation_values(
    user_id: str,
    node_id: str,
    annotation_column: str,
) -> list[str]:
    """Support AI annotation routes with a distinct annotation values helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support AI annotation routes with a distinct annotation values helper" behavior.
    """

    ws = workspace_manager.get_current_workspace(user_id)
    if ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    node_data = ws.nodes[node_id].data
    values = (
        node_data.select(pl.col(annotation_column))
        .collect()
        .get_column(annotation_column)
        .drop_nulls()
        .unique()
        .to_list()
    )
    return sorted(str(value) for value in values)


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------


@router.post("/ai-annotation/models", response_model=AiAnnotationModelsResponse)
async def get_ai_annotation_models(
    request: AiAnnotationModelsRequest,
    current_user: dict = Depends(get_current_user),
):
    """List available models from an OpenAI-compatible endpoint.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /ai-annotation/models route because they need this unit's "List available models from an OpenAI-compatible endpoint" behavior.
    """
    models = await list_models(
        base_url=request.base_url,
        api_key=request.api_key,
    )
    if models:
        return {
            "state": "successful",
            "message": f"Found {len(models)} models",
            "data": {"models": models},
        }
    return {
        "state": "failed",
        "message": "Could not retrieve models from the endpoint",
        "data": {"models": []},
    }


# ---------------------------------------------------------------------------
# Run / Clear
# ---------------------------------------------------------------------------


@router.post("/ai-annotation", response_model=AiAnnotationResponse)
async def run_ai_annotation(
    request: AiAnnotationRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run AI annotation classification on selected nodes.

    Calls the OpenAI-compatible endpoint, stores results in
    the analysis task manager, and returns paginated output.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /ai-annotation route because they need this unit's "Run AI annotation classification on selected nodes" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    if not request.node_ids:
        raise InvalidInputError("At least one node ID must be provided")
    for node_id in request.node_ids:
        node = ws.nodes[node_id]
        node.data

    task_id = str(uuid4())
    task_manager = get_task_manager(user_id)

    analysis_result = GenericAnalysisResult(
        {
            "analysis_params": {
                "model": request.model,
                "base_url": request.base_url,
                "node_columns": request.node_columns,
            },
            "metadata": {
                "task_id": task_id,
                "annotation_columns": ["classification"],
            },
        }
    )

    task_manager.save_task(
        AnalysisTask(
            task_id=task_id,
            user_id=user_id,
            workspace_id=workspace_id,
            request=request,
            status=AnalysisStatus.COMPLETED,
            result=analysis_result,
        )
    )
    saved_task = task_manager.get_task(task_id)
    if saved_task is None:
        raise InternalServiceError("Failed to save task")
    response_data = await _build_response(
        request=request,
        task_id=task_id,
        page=request.page,
        page_size=request.page_size,
        user_id=user_id,
        sort_by=request.sort_by,
        descending=request.descending,
    )
    existing_meta = response_data.get("metadata") or {}
    existing_meta["task_id"] = task_id
    response_data["metadata"] = existing_meta
    return response_data


@router.delete("/ai-annotation", response_model=AnalysisClearResponse)
async def clear_ai_annotation(
    current_user: dict = Depends(get_current_user),
):
    """Clear stored AI annotation task state.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI DELETE /ai-annotation route because they need this unit's "Clear stored AI annotation task state" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    task_manager = get_task_manager(user_id)
    for task in list(task_manager.get_all_tasks()):
        if (
            task.workspace_id == workspace_id
            and task.request.__class__.__name__ == "AiAnnotationRequest"
        ):
            task_manager.clear_task(task.task_id)

    return {
        "state": "successful",
        "message": "AI annotation results have been cleared.",
    }


# ---------------------------------------------------------------------------
# Task result endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/ai-annotation/tasks/{task_id}/request",
    response_model=AiAnnotationRequest,
)
async def ai_annotation_task_request(
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return stored request payload for an AI annotation task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /ai-annotation/tasks/{task_id}/request route because they need this unit's "Return stored request payload for an AI annotation task" behavior.
    """
    user_id = current_user["id"]
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if task is None:
        raise TaskNotFoundError("Task not found")
    return task.request.model_dump()


@router.get(
    "/ai-annotation/tasks/{task_id}/result", response_model=AiAnnotationResponse | None
)
async def ai_annotation_task_result(
    task_id: str,
    page: Optional[int] = Query(None),
    page_size: Optional[int] = Query(None),
    sort_by: Optional[str] = Query(None),
    descending: Optional[bool] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """Read AI annotation result with optional pagination/sort overrides.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /ai-annotation/tasks/{task_id}/result route because they need this unit's "Read AI annotation result with optional pagination/sort overrides" behavior.
    """
    user_id = current_user["id"]
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if task is None:
        raise TaskNotFoundError("Task not found")
    return await _build_response(
        request=task.request,
        task_id=task_id,
        page=page or 1,
        page_size=page_size or 20,
        user_id=user_id,
        sort_by=sort_by,
        descending=descending if descending is not None else True,
    )


@router.post(
    "/ai-annotation/tasks/{task_id}/result", response_model=AiAnnotationResponse
)
async def ai_annotation_task_result_post(
    task_id: str,
    query: AiAnnotationResultQuery,
    current_user: dict = Depends(get_current_user),
):
    """Read AI annotation result using POST body overrides.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /ai-annotation/tasks/{task_id}/result route because they need this unit's "Read AI annotation result using POST body overrides" behavior.
    """
    user_id = current_user["id"]
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if task is None:
        raise TaskNotFoundError("Task not found")
    page = query.page or 1
    page_size = query.page_size or 20
    sort_by = query.sort_by
    descending = query.descending if query.descending is not None else True

    request = task.request.model_copy(
        update={
            "page": page,
            "page_size": page_size,
            "sort_by": sort_by,
            "descending": descending,
        }
    )

    return await _build_response(
        request=request,
        task_id=task_id,
        page=page,
        page_size=page_size,
        user_id=user_id,
        sort_by=sort_by,
        descending=descending,
    )


# ---------------------------------------------------------------------------
# Detach
# ---------------------------------------------------------------------------


@router.post(
    "/nodes/{node_id}/ai-annotation/detach",
    response_model=AiAnnotationDetachResponse,
)
async def detach_ai_annotation(
    node_id: str,
    request: AiAnnotationDetachRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run AI annotation on a node and detach the result as a new child node.

    Classifies texts using the OpenAI-compatible endpoint, writes the result
    as a parquet file, and adds it as a new node in the workspace.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /nodes/{node_id}/ai-annotation/detach route because they need this unit's "Run AI annotation on a node and detach the result as a new child node" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    node = ws.nodes[node_id]
    node_data = node.data

    sel_df = node_data.select(pl.col(request.column)).collect()
    texts = [
        str(v) if v is not None else ""
        for v in sel_df.get_column(request.column).to_list()
    ]

    classes = [c.model_dump() for c in request.classes]
    examples = [e.model_dump() for e in request.examples] if request.examples else None

    results = await classify_texts(
        texts=texts,
        classes=classes,
        examples=examples,
        model=request.model,
        api_key=request.api_key,
        base_url=request.base_url,
        temperature=request.temperature,
        top_p=request.top_p,
        seed=request.seed,
        batch_size=request.batch_size,
        text_column_name=request.column,
    )

    annotation_col = request.annotation_column or "ai_annotation"
    result_df = pl.DataFrame(
        {
            request.column: [r[request.column] for r in results],
            annotation_col: [r["classification"] for r in results],
        }
    )

    # Detach output must live in the workspace-owned top-level `data/`
    # directory, NOT in `data/artifacts/`. Artifacts get wiped on workspace
    # unload (`clear_workspace_artifacts_dir`), so a detached node scanning an
    # artifact path would silently corrupt on next reload. The
    # workspace GC (`_garbage_collect_workspace_data`) keeps top-level
    # parquets alive while they're referenced by any node's plbin.
    workspace_data_dir = _workspace_data_dir(user_id, workspace_id)
    detach_parquet_path = (
        workspace_data_dir / f"ai_annotation_detach_{uuid4().hex}.parquet"
    )
    result_df.write_parquet(str(detach_parquet_path))

    from docworkspace import Node as DwNode

    new_name = request.new_node_name or f"{getattr(node, 'name', node_id)}_annotated"
    lazy_result = pl.scan_parquet(str(detach_parquet_path))
    new_node = DwNode(
        data=lazy_result,
        name=new_name,
        operation="ai_annotation_detach",
        parents=[node],
    )

    try:
        ws.add_node(new_node)
    except Exception as exc:
        raise InternalServiceError(
            f"Failed to add detached node: {exc}",
        )
    # Smart insertion: keep the annotated node directly below its source node.
    ws.place_node_after_parent(new_node)
    update_workspace(user_id, workspace_id, best_effort=True)

    return {
        "state": "successful",
        "message": f"AI annotation detached as '{new_name}'",
        "data": {"new_node_name": new_name, "record_count": len(result_df)},
    }


# ---------------------------------------------------------------------------
# Save annotations to existing node
# ---------------------------------------------------------------------------


@router.post(
    "/nodes/{node_id}/ai-annotation/save",
    response_model=AiAnnotationSaveResponse,
)
async def save_ai_annotation(
    node_id: str,
    request: AiAnnotationSaveRequest,
    current_user: dict = Depends(get_current_user),
):
    """Save AI annotation edits back to the source node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /nodes/{node_id}/ai-annotation/save route because they need this unit's "Save AI annotation edits back to the source node" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    node = ws.nodes[node_id]
    node_data = node.data

    annotation_col = request.annotation_column or "ai_annotation"
    df = node_data.collect()

    if annotation_col not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(annotation_col))

    annotations = df[annotation_col].to_list()
    for edit in request.edits:
        if 0 <= edit.row_index < len(annotations):
            annotations[edit.row_index] = edit.annotation

    df = df.with_columns(pl.Series(annotation_col, annotations))

    # Same rationale as `detach_ai_annotation`: the updated node's data
    # must live under the workspace-owned `data/` dir, not the transient
    # `data/artifacts/` dir. Reassigning `node.data` to a scan of a
    # to-be-deleted artifact would corrupt the existing node on the next
    # workspace unload or analysis submit.
    workspace_data_dir = _workspace_data_dir(user_id, workspace_id)
    save_parquet_path = workspace_data_dir / f"ai_annotation_save_{uuid4().hex}.parquet"
    df.write_parquet(str(save_parquet_path))
    node.data = pl.scan_parquet(str(save_parquet_path))

    update_workspace(user_id, workspace_id, best_effort=True)

    return {
        "state": "successful",
        "message": "Annotations saved",
        "data": {
            "annotation_column": annotation_col,
            "edits_applied": len(request.edits),
        },
    }


# ---------------------------------------------------------------------------
# Node-scoped metadata
# ---------------------------------------------------------------------------


@router.get(
    "/nodes/{node_id}/ai-annotation/providers",
    response_model=AiAnnotationProvidersResponse,
)
async def get_ai_annotation_providers(
    node_id: str,
    annotation_column: str = Query("ai_annotation"),
    current_user: dict = Depends(get_current_user),
):
    """Return distinct provider values from the annotation column of a node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /nodes/{node_id}/ai-annotation/providers route because they need this unit's "Return distinct provider values from the annotation column of a node" behavior.
    """
    user_id = current_user["id"]
    return {
        "state": "successful",
        "data": {
            "providers": _distinct_annotation_values(
                user_id, node_id, annotation_column
            )
        },
    }


@router.get(
    "/nodes/{node_id}/ai-annotation/categories",
    response_model=AiAnnotationCategoriesResponse,
)
async def get_ai_annotation_categories(
    node_id: str,
    annotation_column: str = Query("ai_annotation"),
    current_user: dict = Depends(get_current_user),
):
    """Return distinct category values from the annotation column of a node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /nodes/{node_id}/ai-annotation/categories route because they need this unit's "Return distinct category values from the annotation column of a node" behavior.
    """
    user_id = current_user["id"]
    return {
        "state": "successful",
        "data": {
            "categories": _distinct_annotation_values(
                user_id, node_id, annotation_column
            )
        },
    }
