"""Concordance analysis endpoints.

Includes:
    - POST /workspaces/{workspace_id}/concordance
    - GET  /workspaces/{workspace_id}/concordance/tasks/{task_id}/result
    - POST /workspaces/{workspace_id}/concordance/tasks/{task_id}/result
    - POST /workspaces/{workspace_id}/nodes/{node_id}/concordance/detach

Used by:
- FastAPI workspace analysis routers, frontend analysis features, and backend tests because they need this unit's "Concordance analysis endpoints" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers validate concordance requests, current task state, and saved result overrides.
- Helpers hydrate tokenized nodes, submit worker tasks, read artifacts, and detach generated columns.
- Responses return task metadata, paged concordance rows, dispersion bins, or workspace updates.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast
from uuid import uuid4

import polars as pl
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ....analysis.implementations.concordance import (
    ConcordanceRequest as AnalysisConcordanceRequest,
)
from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....analysis.results import GenericAnalysisResult
from ....core.auth import get_current_user
from ....core.tokens_cache import hydrate_tokenization_lazyframe
from ....core.workspace import workspace_manager
from ....models import (
    AnalysisTaskActionResponse,
    ConcordanceAnalysisRequest,
    ConcordanceAnalysisResponse,
    ConcordanceDetachNodeOption,
    ConcordanceDetachOptionsResponse,
    ConcordanceDetachRequest,
    ConcordanceDispersionBinsResponse,
    ConcordanceDispersionDetachRequest,
    ConcordanceMaterializeRequest,
    CurrentAnalysisTasksResponse,
)
from .cleanup import clear_previous_completed_analysis_task
from .concordance_core import (
    CORE_CONCORDANCE_COLUMNS,
    DEFAULT_CONCORDANCE_PAGE,
    DEFAULT_CONCORDANCE_PAGE_SIZE,
    build_concordance_response,
    normalize_saved_request,
    read_dispersion_bins,
)
from .current_tasks import get_current_task_ids_for_analysis
from .generated_columns import (
    CONC_EXTRACTION_COLUMN,
)
from ..utils import _build_detach_options
from ....core.exceptions import InternalServiceError, InvalidInputError, NoActiveWorkspaceError, NotFoundError, TaskNotFoundError, WorkspaceNotFoundError

router = APIRouter(prefix="/workspaces", tags=["concordance"])
logger = logging.getLogger(__name__)


class ConcordanceResultQuery(BaseModel):
    """Query overrides for reading persisted concordance results.

    Used by:
    - `concordance_task_result` because they need this unit's "Query overrides for reading persisted concordance results" behavior.
    - `concordance_task_result_post` because they need this unit's "Query overrides for reading persisted concordance results" behavior.

    Why:
    - Allows pagination and sorting updates without recomputing concordance.

        Flow:
        - FastAPI/Pydantic parses optional GET query parameters or POST body overrides.
        - Result endpoints merge provided values into the stored concordance request.
        - Downstream response builders receive normalized pagination, sorting, and visibility flags.
    """

    node_id: Optional[str] = None
    page: Optional[int] = None
    page_number: Optional[int] = None
    page_size: Optional[int] = None
    sort_by: Optional[str] = None
    descending: Optional[bool] = None
    show_metadata: Optional[bool] = None
    update_only: bool = False


def _apply_result_query_overrides(
    normalized_request: dict[str, Any],
    query: ConcordanceResultQuery,
) -> dict[str, Any]:
    """Apply request overrides from query parameters.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `concordance_task_result` because they need this unit's "Apply request overrides from query parameters" behavior.
    - `concordance_task_result_post` because they need this unit's "Apply request overrides from query parameters" behavior.

    Why:
    - Reuses one normalization path for GET and POST result retrieval APIs.
    """
    page = query.page_number if query.page_number is not None else query.page
    if page is not None:
        normalized_request["page"] = page
    if query.page_size is not None:
        normalized_request["page_size"] = query.page_size
    if query.sort_by is not None:
        normalized_request["sort_by"] = query.sort_by
    if query.descending is not None:
        normalized_request["descending"] = query.descending
    # Scope the page/sort override to a single node when the client targets one.
    # Each table paginates independently, so a per-node page or sort change must
    # not re-page its sibling. build_concordance_response honors this key to
    # recompute only that node, leaving the other node's client-side data
    # untouched after the merge. The combined comparison view is synthesized
    # client-side by fetching both nodes at the same page.
    if query.node_id is not None:
        normalized_request["result_node_id"] = query.node_id
    return normalized_request


def _failed_concordance_result(message: str) -> dict[str, Any]:
    """Support concordance analysis routes with a failed concordance result helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support concordance analysis routes with a failed concordance result helper" behavior.
    """

    return {
        "state": "failed",
        "message": message,
        "data": None,
    }


def _build_concordance_task_result(
    user_id: str,
    workspace_id: str,
    task_id: str,
    query: ConcordanceResultQuery,
) -> tuple[dict[str, Any] | None, str | None]:
    """Build concordance task result values used by concordance analysis routes.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Build concordance task result values used by concordance analysis routes" behavior.
    """

    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if not task:
        return None, "No analysis found for concordance"
    if not task.request:
        return None, "No concordance request available"

    normalized_request = normalize_saved_request(task.request.model_dump()) or {}
    _apply_result_query_overrides(normalized_request, query)
    return build_concordance_response(user_id, workspace_id, normalized_request), None


@router.post("/concordance", response_model=ConcordanceAnalysisResponse)
async def run_concordance(
    request: ConcordanceAnalysisRequest,
    current_user: dict = Depends(get_current_user),
):
    """Run concordance immediately and store task metadata for retrieval.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend run route: `POST /workspaces/{id}/concordance` because they need this unit's "Run concordance immediately and store task metadata for retrieval" behavior.

    Why:
    - Keeps API behavior aligned with other analyses by returning task-linked
        responses while using shared concordance response builders.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    task_manager = get_task_manager(user_id)

    if not request.node_ids:
        raise InvalidInputError("At least one node ID must be provided")
    try:
        analysis_request = AnalysisConcordanceRequest(
            node_ids=request.node_ids,
            node_columns=request.node_columns,
            search_word=request.search_word,
            num_left_tokens=request.num_left_tokens,
            num_right_tokens=request.num_right_tokens,
            regex=request.regex,
            whole_word=request.whole_word,
            case_sensitive=request.case_sensitive,
            search_mode=request.search_mode,
        )

        task_id = str(uuid4())
        # Drop any prior completed/failed concordance task before recording the
        # new one to keep the per-user analysis store bounded.
        await clear_previous_completed_analysis_task(
            user_id, workspace_id, ["concordance", "concordance_analysis"]
        )
        task_manager.save_task(
            AnalysisTask(
                task_id=task_id,
                user_id=user_id,
                workspace_id=workspace_id,
                request=analysis_request,
                status=AnalysisStatus.COMPLETED,
                result=GenericAnalysisResult({"ready": True}),
            )
        )
        # Key the current-task pointer by tab when the frontend supplies a
        # tab_id so re-running inside a tab supersedes that tab's previous
        # task (set_current_task clears the superseded task + its artifacts).
        # Fall back to the analysis-type key for older clients.
        task_manager.set_current_task(request.tab_id or "concordance", task_id)

        normalized_request = (
            normalize_saved_request(analysis_request.model_dump()) or {}
        )
        normalized_request.setdefault("page", DEFAULT_CONCORDANCE_PAGE)
        if request.sort_by:
            normalized_request["sort_by"] = request.sort_by
        normalized_request["descending"] = request.descending

        response = build_concordance_response(
            user_id,
            workspace_id,
            normalized_request,
        )
        response["metadata"] = {"task_id": task_id}
        return response
    except Exception as exc:
        raise InternalServiceError(f"Failed to run concordance: {exc}")
@router.get("/concordance/tasks/current", response_model=CurrentAnalysisTasksResponse)
async def concordance_current_tasks(
    current_user: dict = Depends(get_current_user),
):
    """Return current task IDs for concordance analysis.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /concordance/tasks/current route because they need this unit's "Return current task IDs for concordance analysis" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    return await get_current_task_ids_for_analysis(
        user_id,
        ["concordance_analysis", "concordance-analysis", "concordance"],
    )


@router.get(
    "/concordance/tasks/{task_id}/request",
    response_model=AnalysisConcordanceRequest,
)
async def concordance_task_request(
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return stored request payload for a concordance task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /concordance/tasks/{task_id}/request route because they need this unit's "Return stored request payload for a concordance task" behavior.
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
    "/concordance/tasks/{task_id}/bins",
    response_model=ConcordanceDispersionBinsResponse,
)
async def concordance_task_dispersion_bins(
    task_id: str,
    node_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return 100-bucket dispersion histogram for one materialised concordance node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend dispersion summary plot, when a node has been materialised by because they need this unit's "Return 100-bucket dispersion histogram for one materialised concordance node" behavior.
      "Process All". The frontend re-aggregates these 100 buckets into a
      smaller number of display bins (4, 5, 10, 20, 25, 50, 100) without
      another network round-trip.

    Why:
    - Server-side pre-binning collapses the response from one row per hit
      (potentially tens of MB on large blocks) to ~100 rows per matched-text
      term, regardless of corpus size.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if task is None or not task.request:
        raise TaskNotFoundError("Task not found")
    materialized_paths = getattr(task.request, "materialized_paths", None) or {}
    path = materialized_paths.get(node_id)
    if not path:
        raise NotFoundError(f"No materialised concordance for node {node_id}",)
    node_columns = getattr(task.request, "node_columns", None) or {}
    document_column = node_columns.get(node_id)

    payload = read_dispersion_bins(path, document_column=document_column)
    return {
        "node_id": node_id,
        **payload,
    }


@router.get(
    "/concordance/tasks/{task_id}/result",
    response_model=ConcordanceAnalysisResponse | None,
)
async def concordance_task_result(
    task_id: str,
    query: ConcordanceResultQuery = Depends(),
    current_user: dict = Depends(get_current_user),
):
    """Read concordance result with optional pagination/sort overrides.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend polling route: `GET /workspaces/{id}/concordance/tasks/{id}/result` because they need this unit's "Read concordance result with optional pagination/sort overrides" behavior.

    Why:
    - Hydrates saved concordance state while allowing query-time view changes.

    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    result, _failure_message = _build_concordance_task_result(
        user_id, workspace_id, task_id, query
    )
    return result


@router.post(
    "/concordance/tasks/{task_id}/result",
    response_model=ConcordanceAnalysisResponse | None,
)
async def concordance_task_result_post(
    task_id: str,
    query: ConcordanceResultQuery,
    current_user: dict = Depends(get_current_user),
):
    """Read concordance result using POST body overrides.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend state-sync route: because they need this unit's "Read concordance result using POST body overrides" behavior.
        `POST /workspaces/{id}/concordance/tasks/{id}/result`

    Why:
    - Preserves compatibility with clients that send result preferences in body
        payloads instead of query parameters.

    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    result, failure_message = _build_concordance_task_result(
        user_id, workspace_id, task_id, query
    )
    if failure_message:
        return _failed_concordance_result(failure_message)
    return result


@router.post(
    "/nodes/{node_id}/concordance/detach",
    response_model=AnalysisTaskActionResponse,
)
async def detach_concordance(
    node_id: str,
    request: ConcordanceDetachRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit a background task to create a concordance-detached node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend detach action: because they need this unit's "Submit a background task to create a concordance-detached node" behavior.
        `POST /workspaces/{id}/nodes/{node_id}/concordance/detach`

    Why:
    - Runs potentially expensive row extraction out-of-band and returns task id
        for progress tracking.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    tm = workspace_manager.get_task_manager(user_id)
    node = ws.nodes[node_id]
    node_data = node.data

    include_document_column = False
    include_extraction = False
    columns_to_select: list[str] = []
    if request.selected_columns:
        for col in request.selected_columns:
            if col == request.column:
                include_document_column = True
                continue
            # CONC_extraction is a generated column, not a source schema
            # column — translate the tick into a worker-side flag and skip
            # source selection.
            if col == CONC_EXTRACTION_COLUMN:
                include_extraction = True
                continue
            columns_to_select.append(col)

    corpus_df = (
        node_data.select(
            [pl.col(request.column)] + [pl.col(c) for c in columns_to_select]
        )
        .filter(
            pl.col(request.column)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.len_chars()
            .fill_null(0)
            > 0
        )
        .collect()
    )
    node_corpus = [
        str(value) if value is not None else ""
        for value in corpus_df.get_column(request.column).to_list()
    ]

    extra_columns_data: dict[str, list] = {}
    extra_columns_dtypes: dict[str, Any] = {}
    for col in columns_to_select:
        if col != request.column:
            series = corpus_df.get_column(col)
            extra_columns_data[col] = series.to_list()
            extra_columns_dtypes[col] = series.dtype

    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
    if workspace_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    try:
        task_info = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type="concordance_detach",
            task_args={
                "workspace_dir": str(workspace_dir),
                "node_corpus": node_corpus,
                "parent_node_id": node_id,
                "document_column": request.column,
                "search_word": request.search_word,
                "num_left_tokens": request.num_left_tokens,
                "num_right_tokens": request.num_right_tokens,
                "regex": request.regex,
                "whole_word": request.whole_word,
                "case_sensitive": request.case_sensitive,
                "new_node_name": request.new_node_name,
                "include_document_column": include_document_column,
                "include_extraction": include_extraction,
                "extra_columns_data": extra_columns_data
                if extra_columns_data
                else None,
                "extra_columns_dtypes": extra_columns_dtypes
                if extra_columns_dtypes
                else None,
                "materialized_path": request.materialized_path,
            },
        )

        return {
            "state": "running",
            "message": "Concordance detach started",
            "data": None,
            "metadata": {"task_id": task_info.id},
        }

    except Exception as exc:
        raise InternalServiceError(f"Error submitting detach task: {exc}")
@router.post(
    "/nodes/{node_id}/concordance/dispersion-detach",
    response_model=AnalysisTaskActionResponse,
)
async def detach_concordance_dispersion(
    node_id: str,
    request: ConcordanceDispersionDetachRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit a per-document aggregated detach (dispersion view).

    Output shape differs from the per-hit detach: one row per source document,
    matched-text/L1/R1/freq columns become `List<T>`, plus a `CONC_extraction`
    string column that joins each hit's character slice with newline + asterisk
    bullets in document-flow order. Used by the dispersion summary chart so the
    user can pull a per-document view (optionally limited to bin-filtered hits)
    into the workspace as a new data block.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /nodes/{node_id}/concordance/dispersion-detach route because they need this unit's "Submit a per-document aggregated detach (dispersion view)" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    tm = workspace_manager.get_task_manager(user_id)
    node = ws.nodes[node_id]
    node_data = node.data

    # Source columns to project — same shape as the per-hit detach so the
    # caller can opt-in to metadata columns and opt-out of the document
    # column.
    include_document_column = False
    columns_to_select: list[str] = []
    if request.selected_columns:
        for col in request.selected_columns:
            if col == request.column:
                include_document_column = True
                continue
            # `CONC_extraction` is the dispersion-detach worker's own output
            # column (the per-document joined raw-window string); a stale
            # client that picks it would otherwise crash this endpoint with
            # `ColumnNotFoundError` on the source-frame select.
            if col == CONC_EXTRACTION_COLUMN:
                continue
            columns_to_select.append(col)

    corpus_df = (
        node_data.select(
            [pl.col(request.column)] + [pl.col(c) for c in columns_to_select]
        )
        .filter(
            pl.col(request.column)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.len_chars()
            .fill_null(0)
            > 0
        )
        .collect()
    )
    node_corpus = [
        str(value) if value is not None else ""
        for value in corpus_df.get_column(request.column).to_list()
    ]

    extra_columns_data: dict[str, list] = {}
    extra_columns_dtypes: dict[str, Any] = {}
    for col in columns_to_select:
        if col != request.column:
            series = corpus_df.get_column(col)
            extra_columns_data[col] = series.to_list()
            extra_columns_dtypes[col] = series.dtype

    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
    if workspace_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    if request.selected_bins is not None and (
        request.total_bins is None or request.total_bins <= 0
    ):
        raise InvalidInputError("total_bins must be a positive integer when selected_bins is provided",)
    try:
        child_task_id = str(uuid4())
        task_info = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type="concordance_dispersion_detach",
            task_id=child_task_id,
            task_name=request.new_node_name or None,
            task_args={
                "workspace_dir": str(workspace_dir),
                "node_corpus": node_corpus,
                "parent_node_id": node_id,
                "child_task_id": child_task_id,
                "parent_task_id": request.parent_task_id,
                "document_column": request.column,
                "search_word": request.search_word,
                "num_left_tokens": request.num_left_tokens,
                "num_right_tokens": request.num_right_tokens,
                "regex": request.regex,
                "whole_word": request.whole_word,
                "case_sensitive": request.case_sensitive,
                "new_node_name": request.new_node_name,
                "include_document_column": include_document_column,
                "extra_columns_data": extra_columns_data or None,
                "extra_columns_dtypes": extra_columns_dtypes or None,
                "materialized_path": request.materialized_path,
                "selected_bins": request.selected_bins,
                "total_bins": request.total_bins,
                "selected_matched_texts": request.selected_matched_texts,
                "match_case_insensitive": request.match_case_insensitive,
            },
        )
        if request.parent_task_id:
            get_task_manager(user_id).link_child_task(
                request.parent_task_id, task_info.id
            )

        return {
            "state": "running",
            "message": "Concordance dispersion detach started",
            "data": None,
            "metadata": {"task_id": task_info.id},
        }
    except Exception as exc:
        raise InternalServiceError(f"Error submitting dispersion detach task: {exc}",)
@router.post(
    "/nodes/{node_id}/concordance/materialize",
    response_model=AnalysisTaskActionResponse,
)
async def materialize_concordance(
    node_id: str,
    request: ConcordanceMaterializeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit a background task that writes the full flattened occurrence parquet.

    Unlike detach, this does not add a node to the workspace. On completion the
    parent concordance analysis task's `materialized_paths` is updated so
    subsequent pagination and detach reuse the cached parquet.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /nodes/{node_id}/concordance/materialize route because they need this unit's "Submit a background task that writes the full flattened occurrence parquet" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    tm = workspace_manager.get_task_manager(user_id)

    if node_id not in ws.nodes:
        raise NotFoundError(f"Node {node_id} not found")
    node = ws.nodes[node_id]
    node_data = node.data

    # Tokens-mode materialize needs the tokenization column alongside the
    # text. Look it up via the node's tokenization registry so we know exactly
    # which column to pull. Empty/missing → 400 so the caller can fall back
    # or prompt the user to re-tokenise.
    tokenization_column: Optional[str] = None
    if request.search_mode == "tokens":
        if hasattr(node, "find_tokenization_column"):
            tokenization_column = node.find_tokenization_column(request.column)
        if tokenization_column is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No tokens column registered on node {node_id!r} for "
                    f"source column {request.column!r}" + "; re-run Tokenise first."
                ),
            )
        node_data = hydrate_tokenization_lazyframe(
            node=node,
            source_column=request.column,
            user_id=user_id,
        )

    # Collect all source columns so the materialized parquet includes metadata.
    # This allows the detach fast path to select only user-chosen columns later.
    all_schema_columns = list(node_data.collect_schema().names())
    extra_source_columns = [
        c
        for c in all_schema_columns
        if c != request.column and c != tokenization_column
    ]
    select_exprs: list[pl.Expr] = [pl.col(request.column)] + [
        pl.col(c) for c in extra_source_columns
    ]
    if tokenization_column is not None:
        select_exprs.append(pl.col(tokenization_column))

    corpus_df = cast(
        pl.DataFrame,
        (
            node_data.select(select_exprs)
            .filter(
                pl.col(request.column)
                .cast(pl.Utf8, strict=False)
                .str.strip_chars()
                .str.len_chars()
                .fill_null(0)
                > 0
            )
            .collect()
        ),
    )
    node_corpus = [
        str(value) if value is not None else ""
        for value in corpus_df.get_column(request.column).to_list()
    ]
    node_tokens: Optional[list[Any]] = None
    if tokenization_column is not None:
        node_tokens = corpus_df.get_column(tokenization_column).to_list()

    extra_columns_data: dict[str, list] | None = None
    extra_columns_dtypes: dict[str, Any] | None = None
    if extra_source_columns:
        extra_columns_data = {}
        extra_columns_dtypes = {}
        for col in extra_source_columns:
            series = corpus_df.get_column(col)
            extra_columns_data[col] = series.to_list()
            extra_columns_dtypes[col] = series.dtype

    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
    if workspace_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    try:
        child_task_id = str(uuid4())
        task_info = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type="concordance_materialize",
            task_id=child_task_id,
            task_args={
                "workspace_dir": str(workspace_dir),
                "node_corpus": node_corpus,
                "child_task_id": child_task_id,
                "parent_task_id": request.parent_task_id,
                "parent_node_id": node_id,
                "document_column": request.column,
                "search_word": request.search_word,
                "num_left_tokens": request.num_left_tokens,
                "num_right_tokens": request.num_right_tokens,
                "regex": request.regex,
                "whole_word": request.whole_word,
                "case_sensitive": request.case_sensitive,
                "extra_columns_data": extra_columns_data,
                "extra_columns_dtypes": extra_columns_dtypes,
                "search_mode": request.search_mode,
                "node_tokens": node_tokens,
            },
        )
        get_task_manager(user_id).link_child_task(request.parent_task_id, task_info.id)
        return {
            "state": "running",
            "message": "Concordance materialize started",
            "data": None,
            "metadata": {"task_id": task_info.id},
        }
    except Exception as exc:
        raise InternalServiceError(f"Error submitting materialize task: {exc}")
@router.get(
    "/nodes/{node_id}/concordance/detach-options",
    response_model=ConcordanceDetachOptionsResponse,
)
async def concordance_detach_options(
    node_id: str,
    column: str,
    current_user: dict = Depends(get_current_user),
):
    """Return detachable concordance columns for one node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend concordance detach dialog because they need this unit's "Return detachable concordance columns for one node" behavior.

    Why:
    - Keeps mandatory generated concordance columns and optional metadata
      columns aligned with backend detach behavior.
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
        mandatory_columns=list(CORE_CONCORDANCE_COLUMNS),
        extraction_column=CONC_EXTRACTION_COLUMN,
        node_option_class=ConcordanceDetachNodeOption,
        detach_options_response_class=ConcordanceDetachOptionsResponse,
        message="Concordance detach options loaded",
    )
