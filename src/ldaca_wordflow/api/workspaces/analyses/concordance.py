"""Concordance analysis endpoints.

Includes:
    - POST /workspaces/{workspace_id}/concordance
    - shared GET/POST analysis-task result routes
    - analysis-task detach, materialize, and dispersion helpers

Used by:
- FastAPI workspace analysis routers, frontend analysis features, and backend tests because they need this unit's "Concordance analysis endpoints" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers validate concordance requests, task records, and saved result overrides.
- Helpers hydrate tokenized nodes, submit worker tasks, read artifacts, and detach generated columns.
- Responses return task metadata, paged concordance rows, dispersion bins, or workspace updates.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....core.auth import get_current_user
from ....core.exceptions import (
    AccessDeniedError,
    InternalServiceError,
    InvalidInputError,
    NotFoundError,
    TaskNotFoundError,
    WorkspaceNotFoundError,
)
from ....core.worker_input_snapshots import create_worker_input_snapshot
from ....core.workspace import workspace_manager
from ....models import (
    AnalysisTaskActionResponse,
    ConcordanceAnalysisRequest,
    ConcordanceAnalysisResponse,
    ConcordanceDetachOptionsResponse,
    ConcordanceDetachRequest,
    ConcordanceDispersionBinsResponse,
    ConcordanceDispersionDetachRequest,
    ConcordanceMaterializeRequest,
    DetachNodeOption,
)
from ..utils import _build_detach_options, require_workspace
from .concordance_core import (
    CORE_CONCORDANCE_COLUMNS,
    build_concordance_response,
    normalize_saved_request,
    read_dispersion_bins,
)
from .concordance_submission import submit_concordance_analysis
from .generated_columns import (
    CONC_EXTRACTION_COLUMN,
    MATERIALIZED_CONCORDANCE_COLUMNS,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
    tags=["concordance"],
)
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
    model_config = ConfigDict(extra="forbid")


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
    if task.result is None:
        return {
            "state": "running",
            "message": "Concordance analysis running",
            "data": {},
            "metadata": {"task_id": task_id},
        }, None

    has_query_overrides = any(
        value is not None
        for value in (
            query.node_id,
            query.page,
            query.page_number,
            query.page_size,
            query.sort_by,
            query.descending,
            query.show_metadata,
        )
    )
    if not has_query_overrides:
        result = task.result.to_json() if hasattr(task.result, "to_json") else task.result
        if isinstance(result, dict):
            result = dict(result)
            result["metadata"] = {"task_id": task_id}
            return result, None

    normalized_request = normalize_saved_request(task.request.model_dump()) or {}
    _apply_result_query_overrides(normalized_request, query)
    return build_concordance_response(user_id, workspace_id, normalized_request), None


@router.post("/concordance", response_model=ConcordanceAnalysisResponse)
async def run_concordance(
    workspace_id: UUID,
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
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    return await submit_concordance_analysis(
        user_id=user_id,
        workspace_id=workspace_id_str,
        workspace=ws,
        request=request,
    )


async def concordance_task_dispersion_bins(
    workspace_id: UUID,
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
    - shared analysis-task dispersion-bins route, when a node has been
      materialised by "Process All". The frontend re-aggregates these buckets into a
      smaller number of display bins (4, 5, 10, 20, 25, 50, 100) without
      another network round-trip.

    Why:
    - Server-side pre-binning collapses the response from one row per hit
      (potentially tens of MB on large blocks) to ~100 rows per matched-text
      term, regardless of corpus size.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if task is None or not task.request:
        raise TaskNotFoundError("Task not found")
    if task.workspace_id != workspace_id_str:
        raise AccessDeniedError("Task does not belong to this workspace")
    materialized_paths = getattr(task.request, "materialized_paths", None) or {}
    path = materialized_paths.get(node_id)
    if not path:
        raise NotFoundError(
            f"No materialised concordance for node {node_id}",
        )
    node_columns = getattr(task.request, "node_columns", None) or {}
    document_column = node_columns.get(node_id)

    payload = read_dispersion_bins(path, document_column=document_column)
    return {
        "node_id": node_id,
        **payload,
    }


async def concordance_task_result(
    workspace_id: str,
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
    - shared analysis-task result route because concordance result reads need
      the existing optional pagination/sort override behavior.

    Why:
    - Hydrates saved concordance state while allowing query-time view changes.

    """
    user_id = current_user["id"]
    result, _failure_message = _build_concordance_task_result(
        user_id, workspace_id, task_id, query
    )
    return result


async def concordance_task_result_post(
    workspace_id: str,
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
    - shared analysis-task result-query route because concordance per-node
      pagination and sort overrides are body-shaped result reads.

    Why:
    - Keeps richer result reads separate from presentation-preference updates.

    """
    user_id = current_user["id"]
    result, failure_message = _build_concordance_task_result(
        user_id, workspace_id, task_id, query
    )
    if failure_message:
        return _failed_concordance_result(failure_message)
    return result


async def detach_concordance(
    workspace_id: UUID,
    node_id: str,
    request: ConcordanceDetachRequest,
    current_user: dict = Depends(get_current_user),
    parent_task_id: str | None = None,
):
    """Submit a background task to create a concordance-detached node.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - shared analysis-task detachments route because concordance detaches are
      task-scoped child actions that create workspace nodes asynchronously.

    Why:
    - Runs potentially expensive row extraction out-of-band and returns task id
        for progress tracking.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    tm = workspace_manager.get_task_manager(user_id)
    if node_id not in ws.nodes:
        raise NotFoundError(f"Node {node_id} not found")

    include_document_column = False
    include_extraction = False
    columns_to_select: list[str] = []
    # The generated CONC_* columns are user-choosable like any other column.
    # Record exactly which ones the client kept so the worker drops the rest.
    # Generated columns must NOT be projected from the source node here — they
    # don't exist in `node_data` and are produced by the detach worker;
    # selecting them off source raises ColumnNotFound.
    generated_names = set(MATERIALIZED_CONCORDANCE_COLUMNS)
    selected_generated_columns: list[str] = []
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
            user_id=user_id,
            workspace_id=workspace_id_str,
            task_id=task_id,
            node_ids=[node_id],
            workspace=ws,
            artifact_dir=workspace_dir / "data" / "artifacts",
        )
        task_info = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id_str,
            task_type="concordance_detach",
            task_id=task_id,
            task_args={
                "workspace_dir": str(workspace_dir),
                "node_corpus": [],
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
                "selected_generated_columns": selected_generated_columns,
                "extra_columns_data": None,
                "extra_columns_dtypes": None,
                "extra_column_names": columns_to_select,
                "materialized_path": request.materialized_path,
                "input_snapshot_dir": str(input_snapshot_dir),
            },
        )
        if parent_task_id:
            get_task_manager(user_id).link_child_task(parent_task_id, task_info.id)

        return {
            "state": "running",
            "message": "Concordance detach started",
            "data": None,
            "metadata": {"task_id": task_info.id},
        }

    except Exception as exc:
        raise InternalServiceError(f"Error submitting detach task: {exc}")


async def detach_concordance_dispersion(
    workspace_id: UUID,
    node_id: str,
    request: ConcordanceDispersionDetachRequest,
    current_user: dict = Depends(get_current_user),
    parent_task_id: str | None = None,
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
    - shared analysis-task dispersion-detachments route because dispersion
      detaches need the parent task path for materialization events and task
      child links.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    tm = workspace_manager.get_task_manager(user_id)
    if node_id not in ws.nodes:
        raise NotFoundError(f"Node {node_id} not found")

    # Source columns to project — same shape as the per-hit detach so the
    # caller can opt-in to metadata columns and opt-out of the document
    # column.
    include_document_column = False
    columns_to_select: list[str] = []
    for col in request.selected_columns:
        if col == request.column:
            include_document_column = True
            continue
        # `CONC_extraction` is the dispersion-detach worker's own output
        # column (the per-document joined raw-window string); selecting it
        # would otherwise crash this endpoint with `ColumnNotFoundError` on
        # the source-frame select.
        if col == CONC_EXTRACTION_COLUMN:
            continue
        columns_to_select.append(col)

    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id_str)
    if workspace_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    if request.selected_bins is not None and (
        request.total_bins is None or request.total_bins <= 0
    ):
        raise InvalidInputError(
            "total_bins must be a positive integer when selected_bins is provided",
        )
    try:
        child_task_id = str(uuid4())
        input_snapshot_dir = create_worker_input_snapshot(
            user_id=user_id,
            workspace_id=workspace_id_str,
            task_id=child_task_id,
            node_ids=[node_id],
            workspace=ws,
            artifact_dir=workspace_dir / "data" / "artifacts",
        )
        task_info = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id_str,
            task_type="concordance_dispersion_detach",
            task_id=child_task_id,
            task_name=request.new_node_name or None,
            task_args={
                "workspace_dir": str(workspace_dir),
                "node_corpus": [],
                "parent_node_id": node_id,
                "child_task_id": child_task_id,
                "parent_task_id": parent_task_id,
                "document_column": request.column,
                "search_word": request.search_word,
                "num_left_tokens": request.num_left_tokens,
                "num_right_tokens": request.num_right_tokens,
                "regex": request.regex,
                "whole_word": request.whole_word,
                "case_sensitive": request.case_sensitive,
                "new_node_name": request.new_node_name,
                "include_document_column": include_document_column,
                "extra_columns_data": None,
                "extra_columns_dtypes": None,
                "extra_column_names": columns_to_select,
                "materialized_path": request.materialized_path,
                "selected_bins": request.selected_bins,
                "total_bins": request.total_bins,
                "selected_matched_texts": request.selected_matched_texts,
                "match_case_insensitive": request.match_case_insensitive,
                "input_snapshot_dir": str(input_snapshot_dir),
            },
        )
        if parent_task_id:
            get_task_manager(user_id).link_child_task(parent_task_id, task_info.id)

        return {
            "state": "running",
            "message": "Concordance dispersion detach started",
            "data": None,
            "metadata": {"task_id": task_info.id},
        }
    except Exception as exc:
        raise InternalServiceError(
            f"Error submitting dispersion detach task: {exc}",
        )


async def materialize_concordance(
    workspace_id: UUID,
    node_id: str,
    request: ConcordanceMaterializeRequest,
    current_user: dict = Depends(get_current_user),
    parent_task_id: str | None = None,
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
    - shared analysis-task materializations route because concordance
      materialization is a child operation of the analysis task selected by the
      URL.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    tm = workspace_manager.get_task_manager(user_id)

    if node_id not in ws.nodes:
        raise NotFoundError(f"Node {node_id} not found")
    node = ws.nodes[node_id]

    # Tokens-mode materialize needs the tokenization column alongside the
    # text. Look it up via the node's tokenization registry so we know exactly
    # which column to pull. Empty/missing → 400 so the caller can fall back
    # or prompt the user to re-tokenise.
    tokenization_column: Optional[str] = None
    if request.search_mode == "tokens":
        if hasattr(node, "find_tokenization_column"):
            tokenization_column = node.find_tokenization_column(request.column)
        if tokenization_column is None:
            raise InvalidInputError(
                f"No tokens column registered on node {node_id!r} for "
                f"source column {request.column!r}; re-run Tokenise first."
            )

    workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id_str)
    if workspace_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    try:
        child_task_id = str(uuid4())
        input_snapshot_dir = create_worker_input_snapshot(
            user_id=user_id,
            workspace_id=workspace_id_str,
            task_id=child_task_id,
            node_ids=[node_id],
            workspace=ws,
            artifact_dir=workspace_dir / "data" / "artifacts",
        )
        task_info = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id_str,
            task_type="concordance_materialize",
            task_id=child_task_id,
            task_args={
                "workspace_dir": str(workspace_dir),
                "node_corpus": [],
                "child_task_id": child_task_id,
                "parent_task_id": parent_task_id,
                "parent_node_id": node_id,
                "document_column": request.column,
                "search_word": request.search_word,
                "num_left_tokens": request.num_left_tokens,
                "num_right_tokens": request.num_right_tokens,
                "regex": request.regex,
                "whole_word": request.whole_word,
                "case_sensitive": request.case_sensitive,
                "extra_columns_data": None,
                "extra_columns_dtypes": None,
                "search_mode": request.search_mode,
                "node_tokens": None,
                "input_snapshot_dir": str(input_snapshot_dir),
            },
        )
        if parent_task_id:
            get_task_manager(user_id).link_child_task(parent_task_id, task_info.id)
        return {
            "state": "running",
            "message": "Concordance materialize started",
            "data": None,
            "metadata": {"task_id": task_info.id},
        }
    except Exception as exc:
        raise InternalServiceError(f"Error submitting materialize task: {exc}")


async def concordance_detach_options(
    workspace_id: UUID,
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
    - shared analysis-task detach-options route because concordance detach
      dialogs need node and column metadata while staying under the parent task
      namespace.

    Why:
    - Keeps mandatory generated concordance columns and optional metadata
      columns aligned with backend detach behavior.
    """
    user_id = current_user["id"]
    ws = require_workspace(user_id, str(workspace_id))
    node = ws.nodes[node_id]

    return _build_detach_options(
        workspace=ws,
        node=node,
        node_id=node_id,
        column=column,
        mandatory_columns=list(CORE_CONCORDANCE_COLUMNS),
        extraction_column=CONC_EXTRACTION_COLUMN,
        node_option_class=DetachNodeOption,
        detach_options_response_class=ConcordanceDetachOptionsResponse,
        message="Concordance detach options loaded",
    )
