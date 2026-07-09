"""Shared analysis-task lifecycle endpoints.

Used by:
- the workspace router package because request/result reads, preferences,
  detachments, materializations, and task artifact reads are task-scoped
  resources regardless of which analysis originally created the task.

Why:
- Keeps the common task lifecycle contract in one API namespace while leaving
  analysis-specific result shaping and worker submission details in the owning
  analysis modules.

Flow:
- Resolve the explicit workspace path and authenticated user.
- Fetch the analysis task from the per-user task manager and verify workspace
  ownership.
- Return stored task state directly or dispatch result/action work to the
  owning analysis helper based on the persisted request type.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ....analysis.implementations.concordance import (
    ConcordanceRequest as AnalysisConcordanceRequest,
)
from ....analysis.implementations.quotation import (
    QuotationRequest as AnalysisQuotationRequest,
)
from ....analysis.implementations.sequential_analysis import (
    SequentialAnalysisRequest as AnalysisSequentialAnalysisRequest,
)
from ....analysis.implementations.token_frequency import (
    TokenFrequencyRequest as AnalysisTokenFrequencyRequest,
)
from ....analysis.implementations.topic_modeling import (
    TopicModelingRequest as AnalysisTopicModelingRequest,
)
from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisTask
from ....core.auth import get_current_user
from ....core.exceptions import AccessDeniedError, InvalidInputError, TaskNotFoundError
from ....models import (
    AnalysisTaskActionResponse,
    ConcordanceDetachOptionsResponse,
    ConcordanceDetachRequest,
    ConcordanceDispersionBinsResponse,
    ConcordanceDispersionDetachRequest,
    ConcordanceMaterializeRequest,
    QuotationResultQuery,
    QuotationDetachOptionsResponse,
    QuotationDetachRequest,
    QuotationMaterializeRequest,
    SequentialAnalysisPreferenceUpdateRequest,
    SequentialAnalysisDetachResponse,
    TokenFrequencyPreferenceUpdateRequest,
    TopicModelingDetachOptionsResponse,
    TopicModelingDetachRequest,
    TopicModelingDetachResponse,
)
from .concordance import (
    ConcordanceResultQuery,
    concordance_detach_options,
    concordance_task_dispersion_bins,
    concordance_task_result,
    concordance_task_result_post,
    detach_concordance,
    detach_concordance_dispersion,
    materialize_concordance,
)
from .quotation import (
    detach_quotation,
    materialize_quotation,
    quotation_detach_options,
    quotation_task_result,
    update_quotation_task_result,
)
from .sequential_analysis import (
    SequentialAnalysisDetachRequest,
    detach_sequential_analysis_task,
    sequential_analysis_task_result,
    update_sequential_analysis_task_result,
)
from .token_frequencies import (
    token_frequencies_task_result,
    update_token_frequencies_task_result,
)
from .topic_modeling import (
    detach_topic_modeling,
    topic_modeling_detach_options,
    topic_modeling_task_result,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}/analysis-tasks",
    tags=["analysis-tasks"],
)


class AnalysisTaskResultQuery(BaseModel):
    """Query parameters accepted by shared task-result reads.

    Used by:
    - ``analysis_task_result`` because the shared route must preserve existing
      lightweight pagination/sorting controls for analyses that support them.

    Flow: FastAPI parses optional query parameters, then the dispatcher maps
        relevant fields into the analysis-specific result query model.
    """

    node_id: Optional[str] = None
    page: Optional[int] = None
    page_number: Optional[int] = None
    page_size: Optional[int] = None
    sort_by: Optional[str] = None
    descending: Optional[bool] = None
    show_metadata: Optional[bool] = None


def _require_analysis_task(
    user_id: str,
    workspace_id: uuid.UUID,
    task_id: str,
) -> AnalysisTask:
    """Return a task that belongs to the explicit workspace path.

    Called by:
    - shared analysis-task read routes because task ids are only valid within
      the workspace selected by the URL.

    Flow: read the task from the per-user manager, raise a not-found error for
        absent ids, reject workspace mismatches, and return the task record for
        request/result serialization.
    """

    task = get_task_manager(user_id).get_task(task_id)
    if task is None:
        raise TaskNotFoundError("Task not found")
    if task.workspace_id != str(workspace_id):
        raise AccessDeniedError("Task does not belong to this workspace")
    return task


@router.get("/{task_id}/request", response_model=Any)
async def analysis_task_request(
    workspace_id: uuid.UUID,
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return the stored request payload for one analysis task.

    Used by:
    - frontend analysis hydration and re-run comparison because every analysis
      tab can recover its last submitted parameters through one task resource.

    Flow: validate the task belongs to the workspace path, then serialize the
        persisted Pydantic request object.
    """

    task = _require_analysis_task(current_user["id"], workspace_id, task_id)
    return task.request.model_dump()


@router.get("/{task_id}/result", response_model=Any)
async def analysis_task_result(
    workspace_id: uuid.UUID,
    task_id: str,
    query: AnalysisTaskResultQuery = Depends(),
    current_user: dict = Depends(get_current_user),
):
    """Return the current or completed result payload for one analysis task.

    Used by:
    - frontend polling and hydration because task status/result reads should
      not require callers to know the analysis namespace.

    Flow: validate task ownership, dispatch by persisted request type, and
        delegate result shaping to the original analysis helper.
    """

    user_id = current_user["id"]
    task = _require_analysis_task(user_id, workspace_id, task_id)
    request = task.request
    if isinstance(request, AnalysisConcordanceRequest):
        return await concordance_task_result(
            str(workspace_id),
            task_id,
            ConcordanceResultQuery(
                node_id=query.node_id,
                page=query.page,
                page_number=query.page_number,
                page_size=query.page_size,
                sort_by=query.sort_by,
                descending=query.descending,
                show_metadata=query.show_metadata,
            ),
            current_user,
        )
    if isinstance(request, AnalysisQuotationRequest):
        return await quotation_task_result(
            str(workspace_id),
            task_id,
            page=query.page,
            page_size=query.page_size,
            sort_by=query.sort_by,
            descending=query.descending,
            current_user=current_user,
        )
    if isinstance(request, AnalysisSequentialAnalysisRequest):
        return await sequential_analysis_task_result(
            str(workspace_id),
            task_id,
            current_user,
        )
    if isinstance(request, AnalysisTokenFrequencyRequest):
        return await token_frequencies_task_result(
            str(workspace_id),
            task_id,
            current_user,
        )
    if isinstance(request, AnalysisTopicModelingRequest):
        return await topic_modeling_task_result(
            str(workspace_id),
            task_id,
            current_user,
        )
    raise InvalidInputError("Unsupported analysis task type")


@router.post("/{task_id}/result-query", response_model=Any)
async def analysis_task_result_query(
    workspace_id: uuid.UUID,
    task_id: str,
    query: ConcordanceResultQuery | QuotationResultQuery,
    current_user: dict = Depends(get_current_user),
):
    """Run a body-based result query for a completed analysis task.

    Used by:
    - concordance result controls because per-node pagination and sorting
      overrides are richer than simple URL query params.

    Flow: validate task ownership, dispatch body queries to analyses that
        support query-time result shaping, and reject unsupported task types
        instead of overloading their preference-update endpoints.
    """

    user_id = current_user["id"]
    task = _require_analysis_task(user_id, workspace_id, task_id)
    payload = query.model_dump(exclude_unset=True)
    if isinstance(task.request, AnalysisConcordanceRequest):
        return await concordance_task_result_post(
            str(workspace_id),
            task_id,
            ConcordanceResultQuery.model_validate(payload),
            current_user,
        )
    if isinstance(task.request, AnalysisQuotationRequest):
        return await update_quotation_task_result(
            str(workspace_id),
            task_id,
            QuotationResultQuery.model_validate({**payload, "update_only": False}),
            current_user,
        )
    raise InvalidInputError("Analysis task does not support result-query")


@router.patch("/{task_id}/preferences", response_model=Any)
async def analysis_task_preferences(
    workspace_id: uuid.UUID,
    task_id: str,
    updates: (
        QuotationResultQuery
        | SequentialAnalysisPreferenceUpdateRequest
        | TokenFrequencyPreferenceUpdateRequest
        | None
    ) = None,
    current_user: dict = Depends(get_current_user),
):
    """Persist display preferences for an existing analysis task.

    Used by:
    - quotation, sequential-analysis, and token-frequency panels because
      presentation settings should not be overloaded onto result reads.

    Flow: validate task ownership, choose the preference model based on the
        persisted request type, then delegate to the owning helper that mutates
        the task record.
    """

    user_id = current_user["id"]
    task = _require_analysis_task(user_id, workspace_id, task_id)
    payload = updates.model_dump(exclude_unset=True) if updates is not None else {}
    if isinstance(task.request, AnalysisQuotationRequest):
        return await update_quotation_task_result(
            str(workspace_id),
            task_id,
            QuotationResultQuery.model_validate({**payload, "update_only": True}),
            current_user,
        )
    if isinstance(task.request, AnalysisSequentialAnalysisRequest):
        return await update_sequential_analysis_task_result(
            str(workspace_id),
            task_id,
            SequentialAnalysisPreferenceUpdateRequest.model_validate(payload),
            current_user,
        )
    if isinstance(task.request, AnalysisTokenFrequencyRequest):
        return await update_token_frequencies_task_result(
            str(workspace_id),
            task_id,
            TokenFrequencyPreferenceUpdateRequest.model_validate(payload),
            current_user,
        )
    raise InvalidInputError("Analysis task does not support preferences")


@router.get(
    "/{task_id}/detach-options",
    response_model=(
        TopicModelingDetachOptionsResponse
        | ConcordanceDetachOptionsResponse
        | QuotationDetachOptionsResponse
    ),
)
async def analysis_task_detach_options(
    workspace_id: uuid.UUID,
    task_id: str,
    node_id: str | None = None,
    column: str | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Return detach option metadata for an analysis task.

    Used by:
    - topic-modeling, concordance, and quotation detach dialogs because
      generated artifact options should stay under the task resource that
      produced the result.

    Flow: validate task ownership, dispatch to analyses that expose task-level
        detach options, require node/column query data for node-scoped
        analyses, and reject unsupported task types.
    """

    user_id = current_user["id"]
    task = _require_analysis_task(user_id, workspace_id, task_id)
    if isinstance(task.request, AnalysisTopicModelingRequest):
        return await topic_modeling_detach_options(workspace_id, task_id, current_user)
    if isinstance(task.request, AnalysisConcordanceRequest):
        if node_id is None or column is None:
            raise InvalidInputError(
                "node_id and column are required for concordance detach-options"
            )
        return await concordance_detach_options(
            workspace_id, node_id, column, current_user
        )
    if isinstance(task.request, AnalysisQuotationRequest):
        if node_id is None or column is None:
            raise InvalidInputError(
                "node_id and column are required for quotation detach-options"
            )
        return await quotation_detach_options(workspace_id, node_id, column, current_user)
    raise InvalidInputError("Analysis task does not support detach-options")


@router.post(
    "/{task_id}/detachments",
    response_model=(
        SequentialAnalysisDetachResponse
        | TopicModelingDetachResponse
        | AnalysisTaskActionResponse
    ),
)
async def create_analysis_task_detachment(
    workspace_id: uuid.UUID,
    task_id: str,
    request: (
        SequentialAnalysisDetachRequest
        | TopicModelingDetachRequest
        | ConcordanceDetachRequest
        | QuotationDetachRequest
    ),
    current_user: dict = Depends(get_current_user),
):
    """Create detached workspace nodes from an analysis task.

    Used by:
    - sequential-analysis, topic-modeling, concordance, and quotation panels
      because detaching generated analysis output is a task lifecycle
      operation.

    Flow: validate task ownership, parse the body with the owning analysis
        request model, and delegate node creation to the analysis helper.
    """

    user_id = current_user["id"]
    task = _require_analysis_task(user_id, workspace_id, task_id)
    payload = request.model_dump(exclude_unset=True)
    if isinstance(task.request, AnalysisSequentialAnalysisRequest):
        return await detach_sequential_analysis_task(
            workspace_id,
            task_id,
            SequentialAnalysisDetachRequest.model_validate(payload),
            current_user,
        )
    if isinstance(task.request, AnalysisTopicModelingRequest):
        return await detach_topic_modeling(
            workspace_id,
            task_id,
            TopicModelingDetachRequest.model_validate(payload),
            current_user,
        )
    if isinstance(task.request, AnalysisConcordanceRequest):
        detach_request = ConcordanceDetachRequest.model_validate(payload)
        return await detach_concordance(
            workspace_id,
            detach_request.node_id,
            detach_request,
            current_user,
            parent_task_id=task_id,
        )
    if isinstance(task.request, AnalysisQuotationRequest):
        detach_request = QuotationDetachRequest.model_validate(payload)
        return await detach_quotation(
            workspace_id,
            detach_request.node_id,
            detach_request,
            current_user,
            parent_task_id=task_id,
        )
    raise InvalidInputError("Analysis task does not support detachments")


@router.post(
    "/{task_id}/dispersion-detachments",
    response_model=AnalysisTaskActionResponse,
)
async def create_analysis_task_dispersion_detachment(
    workspace_id: uuid.UUID,
    task_id: str,
    request: ConcordanceDispersionDetachRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a dispersion-detached workspace node from a concordance task.

    Used by:
    - concordance dispersion views because selected-bin detaches are a
      concordance-specific child action of the parent analysis task.

    Flow: validate task ownership, reject non-concordance tasks, then submit
        the existing dispersion-detach worker with the URL task id as the
        parent task.
    """

    user_id = current_user["id"]
    task = _require_analysis_task(user_id, workspace_id, task_id)
    if not isinstance(task.request, AnalysisConcordanceRequest):
        raise InvalidInputError("Analysis task does not support dispersion-detachments")
    return await detach_concordance_dispersion(
        workspace_id,
        request.node_id,
        request,
        current_user,
        parent_task_id=task_id,
    )


@router.post(
    "/{task_id}/materializations",
    response_model=AnalysisTaskActionResponse,
)
async def create_analysis_task_materialization(
    workspace_id: uuid.UUID,
    task_id: str,
    request: ConcordanceMaterializeRequest | QuotationMaterializeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create or refresh a materialized artifact for an analysis task.

    Used by:
    - concordance and quotation result views because processing all matches is
      a task child action that should update the parent task's cached artifact
      path.

    Flow: validate task ownership, parse the body with the owning analysis
        request model, and submit the existing materialize worker with the URL
        task id as the parent task.
    """

    user_id = current_user["id"]
    task = _require_analysis_task(user_id, workspace_id, task_id)
    payload = request.model_dump(exclude_unset=True)
    if isinstance(task.request, AnalysisConcordanceRequest):
        materialize_request = ConcordanceMaterializeRequest.model_validate(payload)
        return await materialize_concordance(
            workspace_id,
            materialize_request.node_id,
            materialize_request,
            current_user,
            parent_task_id=task_id,
        )
    if isinstance(task.request, AnalysisQuotationRequest):
        materialize_request = QuotationMaterializeRequest.model_validate(payload)
        return await materialize_quotation(
            workspace_id,
            materialize_request.node_id,
            materialize_request,
            current_user,
            parent_task_id=task_id,
        )
    raise InvalidInputError("Analysis task does not support materializations")


@router.get(
    "/{task_id}/dispersion-bins",
    response_model=ConcordanceDispersionBinsResponse,
)
async def analysis_task_dispersion_bins(
    workspace_id: uuid.UUID,
    task_id: str,
    node_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return materialized concordance dispersion bins for one node.

    Used by:
    - concordance dispersion charts because whole-corpus materialized bins are
      a read of the parent concordance task, not a separate analysis namespace.

    Flow: validate task ownership, reject non-concordance tasks, then delegate
        bin loading to the concordance artifact helper.
    """

    user_id = current_user["id"]
    task = _require_analysis_task(user_id, workspace_id, task_id)
    if not isinstance(task.request, AnalysisConcordanceRequest):
        raise InvalidInputError("Analysis task does not support dispersion-bins")
    return await concordance_task_dispersion_bins(
        workspace_id,
        task_id,
        node_id,
        current_user,
    )
