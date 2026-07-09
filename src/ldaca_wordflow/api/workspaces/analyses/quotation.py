"""Quotation analysis endpoints with on-demand paginated result retrieval.

Used by:
- FastAPI workspace analysis routers, frontend analysis features, and backend tests because they need this unit's "Quotation analysis endpoints with on-demand paginated result retrieval" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers validate quotation requests and manage task records.
- Helpers submit extraction work, read on-demand result pages, and attach/detach generated columns.
- Responses return task metadata, quotation pages, preference updates, or saved workspace changes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast
from uuid import UUID, uuid4

import polars as pl
from fastapi import APIRouter, Depends

from ....analysis.manager import get_task_manager
from ....analysis.results import GenericAnalysisResult
from ....core.auth import get_current_user
from ....core.exceptions import (
    InternalServiceError,
    InvalidInputError,
    NotFoundError,
    ResourceConflictError,
    TaskNotFoundError,
    WorkspaceNotFoundError,
)
from ....core.worker_input_snapshots import create_worker_input_snapshot
from ....core.workspace import workspace_manager
from ....models.analysis_common import AnalysisTaskActionResponse, DetachNodeOption
from ....models.quotation import (
    QuotationAnalysisResponse,
    QuotationDetachOptionsResponse,
    QuotationDetachRequest,
    QuotationEngineConfig,
    QuotationMaterializeRequest,
    QuotationPreferenceUpdateResponse,
    QuotationRequest,
    QuotationResultQuery,
)
from ..utils import _build_detach_options, require_workspace
from . import quotation_core as qcore
from .generated_columns import QUOTE_EXTRACTION_COLUMN, is_tokenization_column_name
from .quotation_submission import submit_quotation_analysis

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = qcore.DEFAULT_PAGE_SIZE
DEFAULT_DESCENDING = qcore.DEFAULT_DESCENDING
CORE_QUOTATION_COLUMNS = list(qcore.CORE_QUOTATION_COLUMNS)


router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
    tags=["quotation"],
)


async def quotation_task_result(
    workspace_id: str,
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
    - shared analysis-task result route because quotation result reads need
      optional page recomputation without rerunning extraction.

    Why:
    - Supports cheap preference-only reads and on-demand page recomputation.
    """
    user_id = current_user["id"]
    ws = require_workspace(user_id, workspace_id)
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if not task:
        return None
    if not task.result:
        return {
            "state": "running",
            "message": "Quotation analysis running",
            "data": None,
            "metadata": {"task_id": task_id},
        }

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

        return await qcore.compute_remote_on_demand_page(
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


async def update_quotation_task_result(
    workspace_id: str,
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
    - shared analysis-task result-query and preferences routes because
      quotation supports both page recomputation and context-length preference
      updates.

    Why:
    - Lets UI tune quotation presentation or fetch another page without rerunning
      analysis creation.

    Refactor note:
    - Shares substantial logic with `quotation_task_result`; both could delegate
      to a single internal read/update orchestrator.
    """
    user_id = current_user["id"]
    ws = require_workspace(user_id, workspace_id)
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

    page_payload = await qcore.compute_remote_on_demand_page(
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


@router.post(
    "/nodes/{node_id}/quotation",
    response_model=QuotationAnalysisResponse | AnalysisTaskActionResponse,
)
async def get_quotation(
    workspace_id: UUID,
    node_id: str,
    request: QuotationRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit quotation extraction for selected node and store task payload.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - frontend quotation run/search action because they need this unit's "Run quotation extraction on selected node and store latest task payload" behavior.

    Why:
    - Keeps the initial request responsive by moving extraction/page collection
      into a worker task.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    workspace = require_workspace(user_id, workspace_id_str)
    return await submit_quotation_analysis(
        user_id=user_id,
        workspace_id=workspace_id_str,
        node_id=node_id,
        request=request,
        workspace=workspace,
    )


async def quotation_detach_options(
    workspace_id: UUID,
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
    - shared analysis-task detach-options route because quotation detach
      dialogs need node and column metadata while staying under the parent task
      namespace.

    Why:
    - Keeps mandatory generated quotation columns and optional source columns
      aligned with backend detach behavior.
    """
    user_id = current_user["id"]
    ws = require_workspace(user_id, str(workspace_id))
    node = ws.nodes[node_id]

    return _build_detach_options(
        workspace=ws,
        node=node,
        node_id=node_id,
        column=column,
        mandatory_columns=list(CORE_QUOTATION_COLUMNS),
        extraction_column=QUOTE_EXTRACTION_COLUMN,
        node_option_class=DetachNodeOption,
        detach_options_response_class=QuotationDetachOptionsResponse,
        message="Quotation detach options loaded",
        schema_filter=lambda c: not is_tokenization_column_name(c),
    )


async def detach_quotation(
    workspace_id: UUID,
    node_id: str,
    request: QuotationDetachRequest,
    current_user: dict = Depends(get_current_user),
    parent_task_id: str | None = None,
):
    """Submit background task to detach quotations into a new workspace node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - shared analysis-task detachments route because quotation detaches are
      task-scoped child actions that create workspace nodes asynchronously.

    Why:
    - Offloads potentially expensive extraction/materialization to worker tasks.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    if node_id not in ws.nodes:
        raise NotFoundError(f"Node {node_id} not found")
    tm = workspace_manager.get_task_manager(user_id)

    include_document_column = False
    include_extraction = False
    columns_to_select: list[str] = []
    # The generated quote columns are now user-choosable like any other column.
    # Record exactly which ones the client kept so the worker drops the rest.
    generated_names = set(CORE_QUOTATION_COLUMNS)
    selected_generated_columns: list[str] = []
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

    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id_str)
    if workspace_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    try:
        task_id = str(uuid4())
        input_snapshot_dir = create_worker_input_snapshot(
            workspace_id=workspace_id_str,
            task_id=task_id,
            node_ids=[node_id],
            workspace=ws,
            artifact_dir=workspace_dir / "data" / "artifacts",
        )
        task_info = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id_str,
            task_type="quotation_detach",
            task_id=task_id,
            task_args={
                "workspace_dir": str(workspace_dir),
                "node_corpus": [],
                "parent_node_id": node_id,
                "document_column": request.column,
                "engine_config": request.engine.model_dump() if request.engine else {},
                "new_node_name": request.new_node_name,
                "include_document_column": include_document_column,
                "include_extraction": include_extraction,
                "selected_generated_columns": selected_generated_columns,
                "extra_columns_data": None,
                "extra_columns_dtypes": None,
                "extra_column_names": columns_to_select,
                "materialized_path": request.materialized_path,
                "input_snapshot_dir": str(input_snapshot_dir),
            },
            task_name="Detach Quotation",
        )
        if parent_task_id:
            get_task_manager(user_id).link_child_task(parent_task_id, task_info.id)

        return {
            "state": "running",
            "message": "Quotation detach started",
            "data": None,
            "metadata": {"task_id": task_info.id},
        }

    except Exception as exc:
        logger.exception("Error submitting detach quotation task")
        raise InternalServiceError(f"Error submitting detach task: {exc}")


async def materialize_quotation(
    workspace_id: UUID,
    node_id: str,
    request: QuotationMaterializeRequest,
    current_user: dict = Depends(get_current_user),
    parent_task_id: str | None = None,
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
    - shared analysis-task materializations route because quotation
      materialization is a child operation of the analysis task selected by the
      URL.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    if node_id not in ws.nodes:
        raise NotFoundError(f"Node {node_id} not found")
    tm = workspace_manager.get_task_manager(user_id)

    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id_str)
    if workspace_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    try:
        child_task_id = str(uuid4())
        input_snapshot_dir = create_worker_input_snapshot(
            workspace_id=workspace_id_str,
            task_id=child_task_id,
            node_ids=[node_id],
            workspace=ws,
            artifact_dir=workspace_dir / "data" / "artifacts",
        )
        task_info = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id_str,
            task_type="quotation_materialize",
            task_id=child_task_id,
            task_args={
                "workspace_dir": str(workspace_dir),
                "node_corpus": [],
                "child_task_id": child_task_id,
                "parent_task_id": parent_task_id,
                "parent_node_id": node_id,
                "document_column": request.column,
                "engine_config": request.engine.model_dump() if request.engine else {},
                "extra_columns_data": None,
                "extra_columns_dtypes": None,
                "input_snapshot_dir": str(input_snapshot_dir),
            },
            task_name="Materialize Quotation",
        )
        if parent_task_id:
            get_task_manager(user_id).link_child_task(parent_task_id, task_info.id)
        return {
            "state": "running",
            "message": "Quotation materialize started",
            "data": None,
            "metadata": {"task_id": task_info.id},
        }
    except Exception as exc:
        logger.exception("Error submitting materialize quotation task")
        raise InternalServiceError(f"Error submitting materialize task: {exc}")
