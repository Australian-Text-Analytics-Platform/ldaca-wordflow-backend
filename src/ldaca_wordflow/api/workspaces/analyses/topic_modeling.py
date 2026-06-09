"""Topic Modeling analysis endpoints (background-task based).

Used by:
- FastAPI workspace analysis routers, frontend analysis features, and backend tests because they need this unit's "Topic Modeling analysis endpoints (background-task based)" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers lock per user/workspace, validate node state, and submit topic tasks.
- Helpers read artifacts, reaggregate topics, manage embedding cache state, and detach columns.
- Responses return topic data, task metadata, cache summaries, or saved workspace updates.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import cast
from uuid import uuid4

import polars as pl
from fastapi import APIRouter, Depends, HTTPException

from docworkspace import Node

from ....analysis.implementations.topic_modeling import (
    TopicModelingRequest as AnalysisTopicModelingRequest,
)
from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....analysis.results import GenericAnalysisResult
from ....core.auth import get_current_user
from ....core.exceptions import (
    InternalServiceError,
    InvalidInputError,
    NoActiveWorkspaceError,
    NotFoundError,
    ResourceConflictError,
    TaskNotFoundError,
    WorkspaceNotFoundError,
)
from ....core.utils import get_user_cache_folder
from ....core.worker_tasks_topic_result import reaggregate_exact_topic_modeling_result
from ....core.workspace import workspace_manager
from ....models import (
    AnalysisClearResponse,
    AnalysisTaskMetadata,
    CurrentAnalysisTasksResponse,
    TopicModelingData,
    TopicModelingDetachData,
    TopicModelingDetachedNode,
    TopicModelingDetachNodeOption,
    TopicModelingDetachOptionsResponse,
    TopicModelingDetachRequest,
    TopicModelingDetachResponse,
    TopicModelingEmbeddingCacheClearResponse,
    TopicModelingEmbeddingCacheSizeResponse,
    TopicModelingRequest,
    TopicModelingResponse,
    TopicModelingResultUpdateRequest,
)
from ..utils import ensure_task_synced, update_workspace
from .cleanup import clear_previous_completed_analysis_task
from .current_tasks import get_current_task_ids_for_analysis
from .generated_columns import (
    TOPIC_COLUMN,
    TOPIC_MEANING_COLUMN,
    is_tokenization_column_name,
)

router = APIRouter(prefix="/workspaces", tags=["topic-modeling"])
logger = logging.getLogger(__name__)

_TOPIC_SUBMISSION_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


def _topic_submission_lock(user_id: str, workspace_id: str) -> asyncio.Lock:
    """Support topic-modeling routes with a topic submission lock helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support topic-modeling routes with a topic submission lock helper" behavior.
    """

    key = (user_id, workspace_id)
    lock = _TOPIC_SUBMISSION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _TOPIC_SUBMISSION_LOCKS[key] = lock
    return lock


def _task_metadata(task_id: object | None) -> AnalysisTaskMetadata:
    """Support topic-modeling routes with a task metadata helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support topic-modeling routes with a task metadata helper" behavior.
    """

    return AnalysisTaskMetadata(task_id=str(task_id) if task_id is not None else None)


def _prepare_topic_artifact_target(user_id: str, workspace_id: str) -> tuple[Path, str]:
    """Prepare topic artifact target data consumed by topic-modeling routes.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Prepare topic artifact target data consumed by topic-modeling routes" behavior.
    """

    workspace_artifacts_dir = workspace_manager.ensure_workspace_artifacts_dir(
        user_id, workspace_id
    )
    if workspace_artifacts_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    artifact_prefix = f"topic_modeling_{uuid4()}"
    return workspace_artifacts_dir, artifact_prefix


def _task_result_payload(task: AnalysisTask) -> dict:
    """Support topic-modeling routes with a task result payload helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support topic-modeling routes with a task result payload helper" behavior.
    """

    if task.result is None:
        return {}
    payload = task.result.to_json()
    if not isinstance(payload, dict):
        return {}
    return payload


def _topic_artifacts_from_task(task: AnalysisTask) -> dict:
    """Run the topic artifacts from task background job submitted by API routes.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Run the topic artifacts from task background job submitted by API routes" behavior.
    """

    payload = _task_result_payload(task)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise NotFoundError(
            "Topic modeling artifacts are not available for this task",
        )
    node_artifacts = artifacts.get("nodes")
    meanings_path = artifacts.get("topic_meanings_parquet_path")
    if not isinstance(node_artifacts, list) or not isinstance(meanings_path, str):
        raise InternalServiceError(
            "Topic modeling artifact manifest is invalid",
        )
    return artifacts


def _task_request_payload(task: AnalysisTask) -> dict[str, object]:
    """Support topic-modeling routes with a task request payload helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support topic-modeling routes with a task request payload helper" behavior.
    """

    request = task.request
    if request is None:
        return {}
    if hasattr(request, "model_dump"):
        payload = request.model_dump()
    elif hasattr(request, "dict"):
        payload = request.dict()
    elif isinstance(request, dict):
        payload = request
    else:
        return {}
    return payload if isinstance(payload, dict) else {}


def _format_sampling_scalar(value: float | int) -> str:
    """Format sampling scalar values for topic-modeling routes responses.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Format sampling scalar values for topic-modeling routes responses" behavior.
    """

    return str(value).replace(".", "_")


def _build_sampling_auto_node_name(
    *,
    base_name: str,
    sample_fraction: float,
    random_seed: int | None,
) -> str:
    """Build sampling auto node name values used by topic-modeling routes.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Build sampling auto node name values used by topic-modeling routes" behavior.
    """

    sample_token = f"fr_{_format_sampling_scalar(sample_fraction)}"
    seed_token = (
        f"_rs_{random_seed}"
        if isinstance(random_seed, int) and random_seed >= 0
        else ""
    )
    return f"{base_name}_sampled_{sample_token}{seed_token}"


def _topic_sampling_details_for_node(
    task: AnalysisTask,
    node_id: str,
) -> tuple[float | None, int]:
    """Support topic-modeling routes with a topic sampling details for node helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support topic-modeling routes with a topic sampling details for node helper" behavior.
    """

    request_payload = _task_request_payload(task)
    random_seed = request_payload.get("random_seed")
    if isinstance(random_seed, bool):
        seed = int(random_seed)
    elif isinstance(random_seed, int | float | str):
        try:
            seed = int(random_seed)
        except ValueError:
            seed = 42
    else:
        seed = 42

    node_ids = request_payload.get("node_ids")
    sample_fractions = request_payload.get("sample_fractions")
    if not isinstance(node_ids, list) or not isinstance(sample_fractions, list):
        return None, seed

    node_index = next(
        (index for index, value in enumerate(node_ids) if str(value) == node_id),
        None,
    )
    if node_index is None or node_index >= len(sample_fractions):
        return None, seed

    sample_fraction = sample_fractions[node_index]
    if sample_fraction is None:
        return None, seed
    if isinstance(sample_fraction, bool):
        fraction_value = float(sample_fraction)
    elif isinstance(sample_fraction, int | float | str):
        try:
            fraction_value = float(sample_fraction)
        except ValueError:
            return None, seed
    else:
        return None, seed

    if not (0.0 < fraction_value < 1.0):
        return None, seed
    return fraction_value, seed


def _default_topic_detach_node_name(
    task: AnalysisTask,
    artifact_payload: dict,
    node_id: str,
) -> str:
    """Support topic-modeling routes with a default topic detach node name helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support topic-modeling routes with a default topic detach node name helper" behavior.
    """

    raw_base_name = str(artifact_payload.get("node_name") or node_id).strip() or "node"
    base_name = f"{raw_base_name}_topic"
    sample_fraction, seed = _topic_sampling_details_for_node(task, node_id)
    if sample_fraction is None:
        return base_name
    return _build_sampling_auto_node_name(
        base_name=base_name,
        sample_fraction=sample_fraction,
        random_seed=seed,
    )


async def _require_completed_topic_task(
    user_id: str,
    workspace_id: str,
    task_id: str,
) -> AnalysisTask:
    """Run the require completed topic task background job submitted by API routes.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Run the require completed topic task background job submitted by API routes" behavior.
    """

    task = await ensure_task_synced(
        user_id, workspace_id, task_id, get_task_manager(user_id)
    )
    if not task:
        raise TaskNotFoundError("Task not found")
    if task.status != AnalysisStatus.COMPLETED:
        raise ResourceConflictError(
            "Topic modeling task is not completed",
        )
    return task


@router.delete("/topic-modeling", response_model=AnalysisClearResponse)
async def clear_topic_modeling_results(
    current_user: dict = Depends(get_current_user),
):
    """Clear stored topic-modeling task state for a workspace.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend clear action: `DELETE /workspaces/{id}/topic-modeling` because they need this unit's "Clear stored topic-modeling task state for a workspace" behavior.

    Why:
    - Removes stale result/task pointers before reruns and keeps UI state
        aligned with backend task registries.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    task_manager = get_task_manager(user_id)
    current_id = task_manager.get_current_task_ids("topic_modeling")
    if current_id:
        task_manager.clear_task(current_id[0])

    worker_tm = workspace_manager.get_task_manager(user_id)
    if current_id:
        await worker_tm.clear_task(current_id[0])

    return {
        "state": "successful",
        "message": "Topic modeling analysis results have been cleared.",
    }


def _embedding_cache_dirs(user_id: str) -> list[Path]:
    """Return the embedding-cache directory if it currently exists.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Return the embedding-cache directory if it currently exists" behavior.
    """
    cache_dir = get_user_cache_folder(user_id) / "embeddings"
    return [cache_dir] if cache_dir.exists() and cache_dir.is_dir() else []


def _measure_embedding_cache(user_id: str) -> dict:
    """Compute total size and file count across embedding-cache files.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Compute total size and file count across embedding-cache files" behavior.
    """
    bytes_total = 0
    file_count = 0
    for cache_dir in _embedding_cache_dirs(user_id):
        for entry in cache_dir.glob("*.duckdb"):
            try:
                bytes_total += entry.stat().st_size
                file_count += 1
            except OSError:
                continue
    return {"bytes": bytes_total, "files": file_count}


@router.get(
    "/topic-modeling/embedding-cache/size",
    response_model=TopicModelingEmbeddingCacheSizeResponse,
)
async def get_topic_modeling_embedding_cache_size(
    current_user: dict = Depends(get_current_user),
):
    """Report current embedding-cache size and file count for the user.

    Used by the frontend Clear-cache confirmation dialog so the user can see
    "X MB will be freed" before they confirm.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /topic-modeling/embedding-cache/size route because they need this unit's "Report current embedding-cache size and file count for the user" behavior.
    """
    return {
        "state": "successful",
        "data": _measure_embedding_cache(current_user["id"]),
    }


@router.delete(
    "/topic-modeling/embedding-cache",
    response_model=TopicModelingEmbeddingCacheClearResponse,
)
async def clear_topic_modeling_embedding_cache(
    current_user: dict = Depends(get_current_user),
):
    """Delete every DuckDB entry in the user's embedding cache.

    Sweeps the canonical `user_cache/embeddings/` directory.
    Returns total bytes and file count freed so the UI can confirm the
    reclaim. Clearing forces the next topic-modelling run to re-encode all
    documents from scratch.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI DELETE /topic-modeling/embedding-cache route because they need this unit's "Delete every DuckDB entry in the user's embedding cache" behavior.
    """
    user_id = current_user["id"]
    measured = _measure_embedding_cache(user_id)
    bytes_freed = 0
    files_removed = 0
    for cache_dir in _embedding_cache_dirs(user_id):
        for entry in cache_dir.glob("*.duckdb"):
            try:
                size = entry.stat().st_size
                entry.unlink()
            except OSError as exc:
                logger.debug("Failed to remove %s: %s", entry, exc)
                continue
            bytes_freed += size
            files_removed += 1
    return {
        "state": "successful",
        "message": "Embedding cache cleared.",
        "data": {
            "bytes_freed": bytes_freed,
            "files_removed": files_removed,
            "measured_before": measured,
        },
    }


@router.post("/topic-modeling", response_model=TopicModelingResponse)
async def run_topic_modeling(
    request: TopicModelingRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit topic-modeling analysis as a worker-backed background task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend run route: `POST /workspaces/{id}/topic-modeling` because they need this unit's "Submit topic-modeling analysis as a worker-backed background task" behavior.

    Why:
    - Offloads heavy modeling work to worker processes and returns `task_id`
        for progress/result polling.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    if not request.node_ids:
        raise InvalidInputError("At least one node ID must be provided")
    node_infos: list[dict[str, object]] = []
    for node_id in request.node_ids:
        node = ws.nodes[node_id]
        node_data = node.data

        column_name = request.node_columns[node_id]
        available_columns = list(node_data.collect_schema().names())

        node_infos.append(
            {
                "node_id": node_id,
                "node_name": getattr(node, "name", None) or node_id,
                "text_column": column_name,
                "original_columns": available_columns,
            }
        )

    tm = workspace_manager.get_task_manager(user_id)
    submission_lock = _topic_submission_lock(user_id, workspace_id)
    async with submission_lock:
        # Match token-frequencies behavior: short-circuit when topic modeling is
        # already running for this workspace/user, with lock to avoid duplicate
        # concurrent submissions.
        try:
            if await tm.any_running(
                task_type="topic_modeling", user_id=user_id, workspace_id=workspace_id
            ):
                latest = await tm.latest_by_type(
                    "topic_modeling", user_id=user_id, workspace_id=workspace_id
                )
                return TopicModelingResponse(
                    state="running",
                    message="Topic Modeling analysis already running",
                    data=None,
                    metadata=_task_metadata(latest.id if latest else None),
                )
        except Exception:
            # Non-fatal: proceed to submit a new task.
            pass

        # Drop any prior completed/failed topic-modeling task before submitting
        # a new one. Prevents unbounded accumulation of in-memory task records
        # and on-disk parquet artifacts as the user iterates on parameters.
        await clear_previous_completed_analysis_task(
            user_id, workspace_id, ["topic_modeling", "topic-modeling"]
        )

        workspace_dir = update_workspace(user_id, workspace_id, ws)
        if workspace_dir is None:
            raise InternalServiceError(
                "Failed to persist workspace before topic modeling",
            )
        artifact_dir, artifact_prefix = _prepare_topic_artifact_target(
            user_id, workspace_id
        )
        worker_task = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type="topic_modeling",
            task_args={
                "workspace_dir": str(workspace_dir),
                "node_infos": node_infos,
                "artifact_dir": str(artifact_dir),
                "artifact_prefix": artifact_prefix,
                "min_topic_size": request.min_topic_size,
                "random_seed": request.random_seed,
                "representative_words_count": request.representative_words_count,
                "embedding_cache_dir": str(
                    get_user_cache_folder(user_id) / "embeddings"
                ),
                "sample_fractions": request.sample_fractions,
                "topic_size_mode": request.topic_size_mode,
                "topic_size_value": request.topic_size_value,
            },
            task_name="Topic Modeling",
        )

    analysis_tm = get_task_manager(user_id)
    min_topic_size = (
        request.min_topic_size if request.min_topic_size is not None else 10
    )
    random_seed = request.random_seed if request.random_seed is not None else 42
    representative_words_count = (
        request.representative_words_count
        if request.representative_words_count is not None
        else 5
    )

    analysis_request = AnalysisTopicModelingRequest(
        node_ids=request.node_ids,
        node_columns=request.node_columns,
        min_topic_size=min_topic_size,
        random_seed=random_seed,
        representative_words_count=representative_words_count,
        sample_fractions=request.sample_fractions,
        topic_size_mode=request.topic_size_mode,
        topic_size_value=request.topic_size_value,
    )
    analysis_tm.save_task(
        AnalysisTask(
            task_id=worker_task.id,
            user_id=user_id,
            workspace_id=workspace_id,
            request=analysis_request,
            status=AnalysisStatus.RUNNING,
        )
    )
    analysis_tm.set_current_task("topic_modeling", worker_task.id)
    return TopicModelingResponse(
        state="running",
        message="Topic Modeling analysis started",
        data=None,
        metadata=_task_metadata(worker_task.id),
    )


@router.get(
    "/topic-modeling/tasks/current", response_model=CurrentAnalysisTasksResponse
)
async def topic_modeling_current_tasks(
    current_user: dict = Depends(get_current_user),
):
    """Return current task IDs for topic-modeling analysis.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /topic-modeling/tasks/current route because they need this unit's "Return current task IDs for topic-modeling analysis" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    return await get_current_task_ids_for_analysis(
        user_id, ["topic_modeling", "topic-modeling"]
    )


@router.get(
    "/topic-modeling/tasks/{task_id}/request",
    response_model=AnalysisTopicModelingRequest,
)
async def topic_modeling_task_request(
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return stored request payload for a topic-modeling task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /topic-modeling/tasks/{task_id}/request route because they need this unit's "Return stored request payload for a topic-modeling task" behavior.
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
    "/topic-modeling/tasks/{task_id}/result",
    response_model=TopicModelingResponse,
)
async def topic_modeling_task_result(
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return current status or final payload for a topic-modeling task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend polling route: because they need this unit's "Return current status or final payload for a topic-modeling task" behavior.
        `GET /workspaces/{id}/topic-modeling/tasks/{task_id}/result`

    Why:
    - Normalizes task lifecycle states into one response contract for UI polling.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    task = await ensure_task_synced(
        user_id, workspace_id, task_id, get_task_manager(user_id)
    )

    if not task:
        raise TaskNotFoundError("Task not found")
    if task.status == AnalysisStatus.RUNNING:
        return TopicModelingResponse(
            state="running",
            message="Topic Modeling analysis is running",
            data=None,
            metadata=_task_metadata(task_id),
        )

    if task.status == AnalysisStatus.FAILED:
        return TopicModelingResponse(
            state="failed",
            message=(task.error or "Topic Modeling analysis failed"),
            data=None,
            metadata=_task_metadata(task_id),
        )

    if task.status == AnalysisStatus.COMPLETED and task.result:
        payload = task.result.to_json()
        if not isinstance(payload, dict):
            payload = {}
        result_data = TopicModelingData.model_validate(payload)
        return TopicModelingResponse(
            state="successful",
            message="Topic Modeling analysis complete",
            data=result_data,
            metadata=_task_metadata(task_id),
        )

    return TopicModelingResponse(
        state="failed",
        message="Topic Modeling analysis failed",
        data=None,
        metadata=_task_metadata(task_id),
    )


@router.post(
    "/topic-modeling/tasks/{task_id}/result",
    response_model=TopicModelingResponse,
)
async def update_topic_modeling_task_result(
    task_id: str,
    updates: TopicModelingResultUpdateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Re-aggregate a completed exact topic-modeling task without refitting.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /topic-modeling/tasks/{task_id}/result route because they need this unit's "Re-aggregate a completed exact topic-modeling task without refitting" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    task = await _require_completed_topic_task(user_id, workspace_id, task_id)
    task_manager = get_task_manager(user_id)
    if not task.result:
        raise ResourceConflictError(
            "Topic modeling task is not completed",
        )
    request_payload = task.request.model_dump()
    if request_payload.get("topic_size_mode") != "exact":
        raise ResourceConflictError(
            "Only exact topic modeling results can be re-aggregated",
        )
    requested_topic_count = updates.topic_size_value
    if isinstance(requested_topic_count, bool) or not isinstance(
        requested_topic_count, int
    ):
        raise InvalidInputError(
            "topic_size_value must be an integer",
        )
    payload = _task_result_payload(task)
    artifacts = _topic_artifacts_from_task(task)
    exact_artifact_path = artifacts.get("exact_reduction_artifact_path")
    if not isinstance(exact_artifact_path, str) or not exact_artifact_path:
        raise ResourceConflictError(
            "This exact topic-modeling result cannot be re-aggregated",
        )
    node_artifacts = artifacts.get("nodes") or []
    node_infos = [
        {
            "node_id": node_payload.get("node_id"),
            "node_name": node_payload.get("node_name"),
            "text_column": node_payload.get("text_column"),
            "original_columns": node_payload.get("original_columns") or [],
        }
        for node_payload in node_artifacts
        if isinstance(node_payload, dict)
    ]
    if len(node_infos) != len(node_artifacts):
        raise InternalServiceError(
            "Topic modeling artifact manifest is invalid",
        )
    try:
        updated_payload = reaggregate_exact_topic_modeling_result(
            artifact_path=exact_artifact_path,
            existing_artifacts=artifacts,
            node_infos=node_infos,
            topic_size_value=requested_topic_count,
            representative_words_count=int(
                request_payload.get("representative_words_count") or 5
            ),
            random_seed=int(request_payload.get("random_seed") or 42),
        )
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise InternalServiceError(
            f"Failed to re-aggregate exact topic model: {exc}",
        ) from exc
    existing_meta = cast(
        dict[str, object],
        payload.get("meta") if isinstance(payload.get("meta"), dict) else {},
    )
    updated_meta = cast(
        dict[str, object],
        updated_payload.get("meta")
        if isinstance(updated_payload.get("meta"), dict)
        else {},
    )
    payload.update(
        {
            "topics": updated_payload.get("topics", []),
            "corpus_sizes": updated_payload.get("corpus_sizes", []),
            "per_corpus_topic_counts": updated_payload.get(
                "per_corpus_topic_counts", []
            ),
            "artifacts": updated_payload.get("artifacts", artifacts),
            "meta": {
                **existing_meta,
                **updated_meta,
                "topic_size_mode": "exact",
                "topic_size_value": requested_topic_count,
                "representative_words_count": int(
                    request_payload.get("representative_words_count") or 5
                ),
            },
        }
    )

    # Leave `task.request.topic_size_value` as the original rerun target —
    # the post-fit slider is a display-only re-aggregation, decoupled from
    # the "Target Topic Number" parameter that drives rerun. The new value
    # is recorded in `payload.meta.topic_size_value` (above) so the frontend
    # can restore the slider position on reload.
    task.result = GenericAnalysisResult(payload)
    task_manager.save_task(task)

    return TopicModelingResponse(
        state="successful",
        message="Topic Modeling analysis updated",
        data=TopicModelingData.model_validate(payload),
        metadata=_task_metadata(task_id),
    )


def _resolve_topic_column_name(base_name: str, existing_columns: set[str]) -> str:
    """Return a unique output column name for detached topic data.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `detach_topic_modeling` because they need this unit's "Return a unique output column name for detached topic data" behavior.

    Why:
    - Prevents overwriting source columns when attaching generated topic labels.
    """
    candidate = base_name.strip() or TOPIC_COLUMN
    if candidate not in existing_columns:
        return candidate
    idx = 1
    while f"{candidate}_{idx}" in existing_columns:
        idx += 1
    return f"{candidate}_{idx}"


@router.get(
    "/topic-modeling/tasks/{task_id}/detach-options",
    response_model=TopicModelingDetachOptionsResponse,
)
async def topic_modeling_detach_options(
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """List detachable node/column options for a completed topic task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend detach-options route: because they need this unit's "List detachable node/column options for a completed topic task" behavior.
        `GET /workspaces/{id}/topic-modeling/tasks/{task_id}/detach-options`

    Why:
    - Exposes artifact-backed node metadata so users can choose output columns safely.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    task = await _require_completed_topic_task(user_id, workspace_id, task_id)

    artifacts = _topic_artifacts_from_task(task)
    node_artifacts = artifacts.get("nodes") or []

    nodes: list[TopicModelingDetachNodeOption] = []
    for payload in node_artifacts:
        if not isinstance(payload, dict):
            continue
        node_id = payload.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        source_node = ws.nodes[node_id]
        source_data = source_node.data
        # Temporary dynamic token columns are analysis internals and are
        # excluded from the detach picker if a caller supplied a hydrated plan.
        original_columns = [
            c
            for c in source_data.collect_schema().names()
            if not is_tokenization_column_name(c)
        ]
        topic_column_name = _resolve_topic_column_name(
            TOPIC_COLUMN, set(original_columns)
        )
        nodes.append(
            TopicModelingDetachNodeOption(
                node_id=source_node.id,
                node_name=str(payload.get("node_name") or node_id),
                text_column=str(payload.get("text_column") or ""),
                available_columns=[topic_column_name, *original_columns],
                # The topic column is user-choosable now (default-selected on
                # the client, deselectable) — nothing is force-disabled.
                disabled_columns=[],
            )
        )

    return TopicModelingDetachOptionsResponse(
        state="successful",
        message="Topic detach options loaded",
        data={"nodes": nodes},
        metadata=_task_metadata(task_id),
    )


@router.post(
    "/topic-modeling/tasks/{task_id}/detach",
    response_model=TopicModelingDetachResponse,
)
async def detach_topic_modeling(
    task_id: str,
    request: TopicModelingDetachRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create detached nodes from artifact-backed topic-modeling outputs.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend detach route: because they need this unit's "Create detached nodes from artifact-backed topic-modeling outputs" behavior.
        `POST /workspaces/{id}/topic-modeling/tasks/{task_id}/detach`

        Why:
        - Materializes user-selected columns and topic labels as reusable workspace
            nodes without rerunning the model.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    task = await _require_completed_topic_task(user_id, workspace_id, task_id)

    artifacts = _topic_artifacts_from_task(task)
    node_artifacts = artifacts.get("nodes") or []
    assignments_by_node_id = {
        str(payload.get("node_id")): payload
        for payload in node_artifacts
        if isinstance(payload, dict) and payload.get("node_id")
    }
    meanings_path = Path(str(artifacts.get("topic_meanings_parquet_path")))
    if not meanings_path.exists():
        raise NotFoundError(
            "Topic meanings artifact is missing",
        )
    selected_topic_ids = sorted(
        {int(topic_id) for topic_id in (request.topic_ids or [])}
    )

    # Frontend ships ``topic_meanings_override`` when it wants the detached
    # meanings node to mirror what's currently on screen: post-fit "Words
    # per topic" slice + post-fit stopword filter. The artifact parquet was
    # written at fit time with the user's *original* slice and no filter,
    # so we'd otherwise stamp out a node that doesn't match the visual.
    # We write a fresh parquet beside the artifact so the resulting
    # workspace node has the same on-disk lazy backing the artifact path
    # provides — the workspace's save/reload depends on that. The
    # downstream per-corpus filter still applies, so an override topic
    # that doesn't appear in a given corpus is dropped from that
    # corpus's meanings node (consistent with the multilingual
    # per-corpus semantics).
    if request.topic_meanings_override:
        override_topic_ids = [
            int(item.topic_id) for item in request.topic_meanings_override
        ]
        override_words = [list(item.words) for item in request.topic_meanings_override]
        override_path = (
            meanings_path.parent
            / f"{meanings_path.stem}_override_{uuid4().hex[:8]}{meanings_path.suffix}"
        )
        pl.DataFrame(
            {
                TOPIC_COLUMN: override_topic_ids,
                TOPIC_MEANING_COLUMN: override_words,
            },
            schema={
                TOPIC_COLUMN: pl.Int64,
                TOPIC_MEANING_COLUMN: pl.List(pl.String),
            },
        ).lazy().sink_parquet(override_path)
        meanings_lf = pl.scan_parquet(override_path)
    else:
        meanings_lf = pl.scan_parquet(meanings_path)

    target_node_ids = request.node_ids or list(assignments_by_node_id.keys())
    if not target_node_ids:
        raise InvalidInputError("No node IDs provided for detach")
    detached_nodes: list[TopicModelingDetachedNode] = []
    for node_id in target_node_ids:
        artifact_payload = assignments_by_node_id.get(node_id)
        if not artifact_payload:
            raise InvalidInputError(
                f"Node {node_id} is not available in topic artifact manifest",
            )
        assignments_path = Path(str(artifact_payload.get("assignments_parquet_path")))
        if not assignments_path.exists():
            raise NotFoundError(
                f"Topic assignments artifact missing for node {node_id}",
            )
        assignments_lf = pl.scan_parquet(assignments_path)

        # Filter assignments to selected topics if topic_ids specified
        if selected_topic_ids:
            assignments_lf = assignments_lf.filter(
                pl.col(TOPIC_COLUMN).is_in(selected_topic_ids)
            )

        # Per-corpus meanings must match the topic IDs actually present in
        # this corpus's filtered assignments — with two corpora, the global
        # selected set can include topic IDs absent from one corpus, which
        # previously produced a topic_meanings block that wasn't a subset of
        # the associated detached topics block.
        assignment_topics_df = cast(
            pl.DataFrame,
            assignments_lf.select(pl.col(TOPIC_COLUMN)).unique().collect(),
        )
        corpus_topic_ids = sorted(
            int(value)
            for value in assignment_topics_df.get_column(TOPIC_COLUMN)
            .drop_nulls()
            .to_list()
        )
        filtered_meanings_lf = (
            meanings_lf.filter(pl.col(TOPIC_COLUMN).is_in(corpus_topic_ids))
            if corpus_topic_ids
            else meanings_lf.filter(pl.lit(False))
        ).select(pl.col(TOPIC_COLUMN), pl.col(TOPIC_MEANING_COLUMN))

        source_node = ws.nodes[node_id]
        source_data = source_node.data

        original_columns = list(source_data.collect_schema().names())
        # Resolve the topic column name the same way the detach-options
        # endpoint does (against the source columns only) so it matches the
        # name the client ticked in the dialog.
        topic_column_name = _resolve_topic_column_name(
            request.topic_column_name or TOPIC_COLUMN,
            set(original_columns),
        )
        raw_selected = list((request.selected_columns or {}).get(node_id) or [])
        # The topic column is now user-choosable: include it only when ticked.
        # Everything else must be a real source column.
        include_topic_column = topic_column_name in raw_selected
        source_selected = [col for col in raw_selected if col != topic_column_name]
        if not source_selected and not include_topic_column:
            raise InvalidInputError(
                f"No columns selected for node {node_id}",
            )
        invalid = [col for col in source_selected if col not in original_columns]
        if invalid:
            raise InvalidInputError(
                f"Invalid selected columns for node {node_id}: {invalid}",
            )

        projection = [pl.col(col) for col in source_selected]
        if include_topic_column:
            projection.append(pl.col(TOPIC_COLUMN).alias(topic_column_name))
        output_lf = assignments_lf.join(
            source_data.with_row_index("__row_nr__"),
            on="__row_nr__",
            how="inner",
        ).select(projection)

        parents = [source_node] if source_node else []
        node_name = (
            (request.new_node_names or {}).get(node_id)
            if request.new_node_names
            else None
        ) or _default_topic_detach_node_name(task, artifact_payload, node_id)

        # Materialize the detached outputs into workspace-owned parquet files
        # so the new nodes are self-contained. The originals live under
        # `data/artifacts/` which gets cleaned by the next analysis submit
        # (clear_previous_completed_analysis_task) and by workspace unload —
        # a detached node that still scanned them would silently corrupt on
        # the next run. The top-level workspace data dir is protected by
        # `_garbage_collect_workspace_data` (deletes only unreferenced files).
        workspace_data_dir = Path(ws.ws_root_dir) / "data"
        workspace_data_dir.mkdir(parents=True, exist_ok=True)
        new_node_id = str(uuid4())
        new_node_parquet = workspace_data_dir / f"topic_detach_{new_node_id}.parquet"
        cast(pl.DataFrame, output_lf.collect()).write_parquet(new_node_parquet)
        new_node = Node(
            data=pl.scan_parquet(new_node_parquet),
            id=new_node_id,
            name=node_name,
            workspace=ws,
            operation="topic_modeling_detach",
            parents=parents,
        )
        ws.add_node(new_node)

        text_column = artifact_payload.get("text_column")
        if text_column and text_column in source_selected:
            try:
                new_node.document = text_column
            except Exception as exc:
                logger.debug(
                    "Failed to set detached topic node document column '%s' for node %s: %s",
                    text_column,
                    new_node.id,
                    exc,
                )

        meanings_node_name = f"{node_name}_topic_meanings"
        meanings_node_id = str(uuid4())
        meanings_node_parquet = (
            workspace_data_dir / f"topic_meanings_detach_{meanings_node_id}.parquet"
        )
        cast(pl.DataFrame, filtered_meanings_lf.collect()).write_parquet(
            meanings_node_parquet
        )
        meanings_node = Node(
            data=pl.scan_parquet(meanings_node_parquet),
            id=meanings_node_id,
            name=meanings_node_name,
            workspace=ws,
            operation="topic_modeling_meanings_detach",
            parents=[new_node],
        )
        ws.add_node(meanings_node)

        detached_nodes.append(
            TopicModelingDetachedNode(
                source_node_id=node_id,
                new_node_id=new_node.id,
                topic_meanings_node_id=meanings_node.id,
            )
        )

    update_workspace(user_id, workspace_id, ws)

    return TopicModelingDetachResponse(
        state="successful",
        message="Topic detach completed",
        data=TopicModelingDetachData(detached_nodes=detached_nodes),
        metadata=_task_metadata(task_id),
    )
