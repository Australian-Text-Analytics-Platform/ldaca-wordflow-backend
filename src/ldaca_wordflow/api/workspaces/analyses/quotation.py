"""Quotation analysis endpoints with on-demand paginated result retrieval.

Used by:
- FastAPI workspace analysis routers, frontend analysis features, and backend tests because they need this unit's "Quotation analysis endpoints with on-demand paginated result retrieval" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers validate quotation requests, and manage current task state.
- Helpers submit extraction work, read on-demand result pages, and attach/detach generated columns.
- Responses return task metadata, quotation pages, preference updates, or saved workspace changes.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional, cast
from uuid import uuid4

import polars as pl
from fastapi import APIRouter, Depends, HTTPException

from ....analysis.implementations.quotation import (
    QuotationRequest as AnalysisQuotationRequest,
)
from ....analysis.manager import get_task_manager
from ....analysis.results import GenericAnalysisResult
from ....core.auth import get_current_user
from ....core.exceptions import (
    BadGatewayError,
    InternalServiceError,
    InvalidInputError,
    NoActiveWorkspaceError,
    NotFoundError,
    ResourceConflictError,
    TaskNotFoundError,
    WorkspaceNotFoundError,
)
from ....core.services.quotation_client import (
    QuotationServiceError,
    extract_remote_quotations,
)
from ....core.workspace import workspace_manager
from ....models import (
    AnalysisTaskActionResponse,
    CurrentAnalysisTasksResponse,
    QuotationAnalysisResponse,
    QuotationDetachNodeOption,
    QuotationDetachOptionsResponse,
    QuotationDetachRequest,
    QuotationEngineConfig,
    QuotationMaterializeRequest,
    QuotationPreferenceUpdateResponse,
    QuotationRequest,
    QuotationResultQuery,
)
from ....settings import settings
from ..utils import _build_detach_options
from . import quotation_core as qcore
from .current_tasks import get_current_task_ids_for_analysis
from .generated_columns import QUOTE_EXTRACTION_COLUMN, is_tokenization_column_name

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_LENGTH = qcore.DEFAULT_CONTEXT_LENGTH
DEFAULT_PAGE_SIZE = qcore.DEFAULT_PAGE_SIZE
DEFAULT_DESCENDING = qcore.DEFAULT_DESCENDING
CORE_QUOTATION_COLUMNS = list(qcore.CORE_QUOTATION_COLUMNS)


async def _compute_on_demand_page(
    node: Any,
    column: str,
    engine: QuotationEngineConfig,
    *,
    page: int,
    page_size: Optional[int],
    sort_by: Optional[str],
    descending: bool,
    materialized_path: Optional[str] = None,
) -> dict[str, Any]:
    """Compute paged quotation payloads via shared quotation-core helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Compute paged quotation payloads via shared quotation-core helper" behavior.
    """
    compute_quote_dataframe_fn = partial(
        qcore.compute_quote_dataframe,
        extract_remote_fn=extract_remote_quotations,
        quotation_service_max_batch_size=settings.quotation_service_max_batch_size,
        quotation_service_timeout=settings.quotation_service_timeout,
    )

    return await qcore.compute_on_demand_page(
        node,
        column,
        engine,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        descending=descending,
        compute_quote_dataframe_fn=compute_quote_dataframe_fn,
        materialized_path=materialized_path,
    )


router = APIRouter(prefix="/workspaces", tags=["quotation"])


@router.get("/quotation/tasks/current", response_model=CurrentAnalysisTasksResponse)
async def quotation_current_tasks(
    current_user: dict = Depends(get_current_user),
):
    """Return current task IDs for quotation analysis.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /quotation/tasks/current route because they need this unit's "Return current task IDs for quotation analysis" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    return await get_current_task_ids_for_analysis(
        user_id, ["quotation_analysis", "quotation-analysis", "quotation"]
    )


@router.get(
    "/quotation/tasks/{task_id}/request",
    response_model=AnalysisQuotationRequest,
)
async def quotation_task_request(
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return stored request payload for a quotation task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /quotation/tasks/{task_id}/request route because they need this unit's "Return stored request payload for a quotation task" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if task is None:
        raise TaskNotFoundError("Task not found")
    request = task.request
    return request.model_dump()


@router.get(
    "/quotation/tasks/{task_id}/result", response_model=QuotationAnalysisResponse | None
)
async def quotation_task_result(
    task_id: str,
    page: Optional[int] = None,
    page_size: Optional[int] = None,
    sort_by: Optional[str] = None,
    descending: Optional[bool] = None,
    current_user: dict = Depends(get_current_user),
):
    """Return stored quotation result, optionally recomputed for new page params.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend polling route for quotation result panels because they need this unit's "Return stored quotation result, optionally recomputed for new page params" behavior.

    Why:
    - Supports cheap preference-only reads and on-demand page recomputation.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if not task or not task.result:
        return None

    base_result = task.result.to_json()
    req_dict = task.request.model_dump()

    if any(v is not None for v in (page, page_size, sort_by, descending)):
        node_id = req_dict.get("node_id")
        column = req_dict.get("column")
        if not node_id or not column:
            return base_result

        engine_dict = req_dict.get("engine") or {}
        engine_dict = {
            k: v for k, v in engine_dict.items() if k not in ("api_key", "model")
        }
        try:
            engine = QuotationEngineConfig.model_validate(engine_dict)
        except Exception:
            return base_result

        node = ws.nodes[node_id]

        normalized_page = max(1, int(page)) if isinstance(page, int) and page else 1

        return await _compute_on_demand_page(
            node,
            column,
            engine,
            page=normalized_page,
            page_size=page_size,
            sort_by=sort_by or None,
            descending=descending if descending is not None else DEFAULT_DESCENDING,
            materialized_path=req_dict.get("materialized_path"),
        )

    return base_result


@router.post(
    "/quotation/tasks/{task_id}/result",
    response_model=QuotationAnalysisResponse | QuotationPreferenceUpdateResponse,
)
async def update_quotation_task_result(
    task_id: str,
    query: QuotationResultQuery,
    current_user: dict = Depends(get_current_user),
):
    """Persist quotation display preferences and optional page overrides.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend preference updates for context length/sort/page controls because they need this unit's "Persist quotation display preferences and optional page overrides" behavior.

    Why:
    - Lets UI tune quotation presentation without rerunning analysis creation.

    Refactor note:
    - Shares substantial logic with `quotation_task_result`; both could delegate
      to a single internal read/update orchestrator.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if not task or not task.result:
        raise NotFoundError("No quotation analysis found")
    base_request = task.request.model_dump()
    base_result = task.result.to_json()

    context_length_value = qcore.extract_context_preference(base_result)
    if query.context_length is not None:
        context_length_value = qcore.normalize_context_length(query.context_length)

    preferences = {
        **(
            base_result.get("preferences")
            if isinstance(base_result.get("preferences"), dict)
            else {}
        ),
        "context_length": context_length_value,
    }

    needs_pagination = (
        any(
            value is not None
            for value in (query.page, query.page_size, query.sort_by, query.descending)
        )
        and not query.update_only
    )

    if not needs_pagination:
        base_result["preferences"] = preferences
        try:
            task.complete(GenericAnalysisResult(base_result))
            task_manager.save_task(task)
        except Exception as exc:  # pragma: no cover
            raise InternalServiceError(
                f"Failed to persist quotation preferences: {exc}",
            )
        return {
            "state": "successful",
            "message": "saved",
            "data": {"context_length": context_length_value},
        }

    node_id = base_request.get("node_id")
    column = base_request.get("column")
    if not node_id or not column:
        raise NotFoundError("No quotation analysis found for this workspace")
    engine_dict = base_request.get("engine") or {}
    engine_dict = {
        k: v for k, v in engine_dict.items() if k not in ("api_key", "model")
    }
    try:
        engine = QuotationEngineConfig.model_validate(engine_dict)
    except Exception as exc:  # pragma: no cover
        raise InvalidInputError(f"Invalid engine config: {exc}")
    node = ws.nodes[node_id]

    normalized_page = (
        max(1, int(query.page)) if isinstance(query.page, int) and query.page else 1
    )

    effective_page_size = int(query.page_size) if query.page_size is not None else None

    page_payload = await _compute_on_demand_page(
        node,
        column,
        engine,
        page=normalized_page,
        page_size=effective_page_size,
        sort_by=query.sort_by or None,
        descending=(
            query.descending if query.descending is not None else DEFAULT_DESCENDING
        ),
        materialized_path=base_request.get("materialized_path"),
    )

    updated_result = {**page_payload, "preferences": preferences}

    try:
        task.complete(GenericAnalysisResult(updated_result))
        if hasattr(task.request, "page"):
            task.request.page = normalized_page
            task.request.page_size = page_payload.get("pagination", {}).get("page_size")
            task.request.sort_by = query.sort_by or None
            task.request.descending = (
                query.descending if query.descending is not None else DEFAULT_DESCENDING
            )

        task_manager.save_task(task)
    except Exception as exc:  # pragma: no cover
        raise InternalServiceError(
            f"Failed to persist quotation pagination update: {exc}",
        )
    return updated_result


@router.post("/nodes/{node_id}/quotation", response_model=QuotationAnalysisResponse)
async def get_quotation(
    node_id: str,
    request: QuotationRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run quotation extraction on selected node and store latest task payload.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend quotation run/search action because they need this unit's "Run quotation extraction on selected node and store latest task payload" behavior.

    Why:
    - Produces immediate result payload and persists it as current quotation task.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    task_manager = get_task_manager(user_id)

    try:
        workspace = workspace_manager.get_current_workspace(user_id)
        if workspace is None:
            raise NoActiveWorkspaceError("No active workspace selected")
        node = workspace.nodes[node_id]

        engine = request.engine or QuotationEngineConfig()

        page = (
            max(1, int(request.page))
            if isinstance(request.page, int) and request.page
            else 1
        )

        page_payload = await _compute_on_demand_page(
            node,
            request.column,
            engine,
            page=page,
            page_size=request.page_size,
            sort_by=request.sort_by or None,
            descending=request.descending,
            materialized_path=None,
        )
        resolved_page_size = page_payload.get("pagination", {}).get(
            "page_size", DEFAULT_PAGE_SIZE
        )

        context_length_pref = DEFAULT_CONTEXT_LENGTH
        try:
            prev_task_ids = task_manager.get_current_task_ids("quotation")
            prev_task = (
                task_manager.get_task(prev_task_ids[0]) if prev_task_ids else None
            )
            if prev_task and prev_task.result:
                prev_result = prev_task.result.to_json()
                context_length_pref = qcore.extract_context_preference(prev_result)
        except Exception:  # pragma: no cover
            context_length_pref = DEFAULT_CONTEXT_LENGTH

        result_payload: dict[str, Any] = {
            **page_payload,
            "preferences": {"context_length": context_length_pref},
        }

        analysis_request = AnalysisQuotationRequest(
            node_id=node_id,
            column=request.column,
            engine=request.engine.model_dump(mode="json") if request.engine else None,
            page=page,
            page_size=resolved_page_size,
            sort_by=request.sort_by or None,
            descending=request.descending,
            context_length=context_length_pref,
        )

        existing_task_ids = task_manager.get_current_task_ids("quotation")
        existing_task = (
            task_manager.get_task(existing_task_ids[0]) if existing_task_ids else None
        )

        if existing_task:
            existing_req = existing_task.request
            if existing_req.node_id != node_id or existing_req.column != request.column:
                raise ResourceConflictError(
                    "Clear current quotation results before starting a new quotation analysis",
                )
            task = existing_task

        else:
            task_id = task_manager.create_task(analysis_request)
            task = task_manager.get_task(task_id)
            task_manager.set_current_task("quotation", task_id)

        if task is None:
            raise InternalServiceError(
                "Failed to load quotation task",
            )
        task.request = analysis_request
        task.complete(GenericAnalysisResult(result_payload))
        task_manager.save_task(task)

        result_payload["task_id"] = task.task_id
        return result_payload
    except HTTPException:
        raise
    except QuotationServiceError as exc:
        raise BadGatewayError(str(exc))
    except ValueError as exc:
        raise InvalidInputError(str(exc))
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected quotation error")
        raise InternalServiceError(f"Internal server error: {exc}")


@router.get(
    "/nodes/{node_id}/quotation/detach-options",
    response_model=QuotationDetachOptionsResponse,
)
async def quotation_detach_options(
    node_id: str,
    column: str,
    current_user: dict = Depends(get_current_user),
):
    """Return detachable quotation columns for one node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend quotation detach dialog because they need this unit's "Return detachable quotation columns for one node" behavior.

    Why:
    - Keeps mandatory generated quotation columns and optional source columns
      aligned with backend detach behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    node = ws.nodes[node_id]

    return _build_detach_options(
        workspace=ws,
        node=node,
        node_id=node_id,
        column=column,
        mandatory_columns=list(CORE_QUOTATION_COLUMNS),
        extraction_column=QUOTE_EXTRACTION_COLUMN,
        node_option_class=QuotationDetachNodeOption,
        detach_options_response_class=QuotationDetachOptionsResponse,
        message="Quotation detach options loaded",
        schema_filter=lambda c: not is_tokenization_column_name(c),
    )


@router.post(
    "/nodes/{node_id}/quotation/detach",
    response_model=AnalysisTaskActionResponse,
)
async def detach_quotation(
    node_id: str,
    request: QuotationDetachRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit background task to detach quotations into a new workspace node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend quotation detach action because they need this unit's "Submit background task to detach quotations into a new workspace node" behavior.

    Why:
    - Offloads potentially expensive extraction/materialization to worker tasks.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    node = ws.nodes[node_id]
    tm = workspace_manager.get_task_manager(user_id)
    node_data = node.data

    include_document_column = False
    include_extraction = False
    columns_to_select: list[str] = []
    # The generated quote columns are now user-choosable like any other column.
    # Record exactly which ones the client kept so the worker drops the rest.
    # `None` (no selection sent) preserves the legacy "keep all generated"
    # behavior for older callers.
    generated_names = set(CORE_QUOTATION_COLUMNS)
    selected_generated_columns: list[str] | None = None
    if request.selected_columns is not None:
        selected_generated_columns = []
        for col in request.selected_columns:
            if col == request.column:
                include_document_column = True
                continue
            # QUOTE_extraction is a generated column, not a source schema
            # column — translate to a worker flag and skip source-selection.
            if col == QUOTE_EXTRACTION_COLUMN:
                include_extraction = True
                continue
            if col in generated_names:
                selected_generated_columns.append(col)
                continue
            columns_to_select.append(col)

    node_corpus, extra_columns_data, extra_columns_dtypes = (
        qcore.collect_non_empty_quotation_corpus(
            node_data, request.column, columns_to_select
        )
    )

    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
    if workspace_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    try:
        task_info = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type="quotation_detach",
            task_args={
                "workspace_dir": str(workspace_dir),
                "node_corpus": node_corpus,
                "parent_node_id": node_id,
                "document_column": request.column,
                "engine_config": request.engine.model_dump() if request.engine else {},
                "new_node_name": request.new_node_name,
                "include_document_column": include_document_column,
                "include_extraction": include_extraction,
                "selected_generated_columns": selected_generated_columns,
                "extra_columns_data": extra_columns_data or None,
                "extra_columns_dtypes": extra_columns_dtypes or None,
                "materialized_path": request.materialized_path,
            },
            task_name="Detach Quotation",
        )

        return {
            "state": "running",
            "message": "Quotation detach started",
            "data": None,
            "metadata": {"task_id": task_info.id},
        }

    except Exception as exc:
        logger.exception("Error submitting detach quotation task")
        raise InternalServiceError(f"Error submitting detach task: {exc}")


@router.post(
    "/nodes/{node_id}/quotation/materialize", response_model=AnalysisTaskActionResponse
)
async def materialize_quotation(
    node_id: str,
    request: QuotationMaterializeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit a background task that writes the full flattened quotation parquet.

    Unlike detach, this does not add a node to the workspace. On completion the
    parent quotation analysis task's `materialized_path` is updated so subsequent
    pagination and detach reuse the cached parquet.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /nodes/{node_id}/quotation/materialize route because they need this unit's "Submit a background task that writes the full flattened quotation parquet" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    if node_id not in ws.nodes:
        raise NotFoundError(f"Node {node_id} not found")
    node = ws.nodes[node_id]
    tm = workspace_manager.get_task_manager(user_id)
    node_data = node.data

    # Materialise should preserve EVERY source-node column so the table
    # view's metadata-column selector still has the original columns to
    # pick from after Process All (the live unmaterialised path joins
    # the source rows back to each quote-row, so they're visible there).
    # Without this, the materialised parquet only carries the document
    # column + QUOTE_* derivatives, and the table loses all metadata
    # the moment the user clicks Process All.
    source_schema = node_data.collect_schema()
    source_columns = list(source_schema.names())
    extra_metadata_columns = [
        col
        for col in source_columns
        if col != request.column and not is_tokenization_column_name(col)
    ]

    node_corpus, extra_columns_data, extra_columns_dtypes = (
        qcore.collect_non_empty_quotation_corpus(
            node_data, request.column, extra_metadata_columns
        )
    )

    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
    if workspace_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    try:
        child_task_id = str(uuid4())
        task_info = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type="quotation_materialize",
            task_id=child_task_id,
            task_args={
                "workspace_dir": str(workspace_dir),
                "node_corpus": node_corpus,
                "child_task_id": child_task_id,
                "parent_task_id": request.parent_task_id,
                "parent_node_id": node_id,
                "document_column": request.column,
                "engine_config": request.engine.model_dump() if request.engine else {},
                "extra_columns_data": extra_columns_data or None,
                "extra_columns_dtypes": extra_columns_dtypes or None,
            },
            task_name="Materialize Quotation",
        )
        get_task_manager(user_id).link_child_task(request.parent_task_id, task_info.id)
        return {
            "state": "running",
            "message": "Quotation materialize started",
            "data": None,
            "metadata": {"task_id": task_info.id},
        }
    except Exception as exc:
        logger.exception("Error submitting materialize quotation task")
        raise InternalServiceError(f"Error submitting materialize task: {exc}")
