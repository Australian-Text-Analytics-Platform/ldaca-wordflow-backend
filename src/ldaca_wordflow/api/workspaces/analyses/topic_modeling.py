"""Topic Modeling analysis endpoints (background-task based).

Used by:
- FastAPI workspace analysis routers, frontend analysis features, and backend tests because they need this unit's "Topic Modeling analysis endpoints (background-task based)" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers lock per user/workspace, validate node state, and submit topic tasks.
- Helpers read artifacts, reaggregate topics, and detach columns.
- Responses return topic data, task metadata, or saved workspace updates.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import polars as pl
from fastapi import APIRouter, Depends

from docworkspace import Node

from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....core.auth import get_current_user
from ....core.exceptions import (
    InternalServiceError,
    InvalidInputError,
    NotFoundError,
    ResourceConflictError,
    TaskNotFoundError,
)
from ....core.workspace import workspace_manager
from ....models.analysis_common import AnalysisClearResponse, AnalysisTaskMetadata, DetachNodeOption
from ....models.topic_modeling import (
    TopicModelingData,
    TopicModelingDetachData,
    TopicModelingDetachedNode,
    TopicModelingDetachOptionsResponse,
    TopicModelingDetachRequest,
    TopicModelingDetachResponse,
    TopicModelingRequest,
    TopicModelingResponse,
)
from ..utils import ensure_task_synced, require_workspace, update_workspace
from .generated_columns import (
    TOPIC_COLUMN,
    TOPIC_DISTRIBUTION_COLUMN,
    TOPIC_DISTRIBUTION_OUTPUT_COLUMN,
    TOPIC_MEANING_COLUMN,
    TOPIC_TOP1_COLUMN,
    is_tokenization_column_name,
)
from .topic_modeling_submission import submit_topic_modeling

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
    tags=["topic-modeling"],
)


logger = logging.getLogger(__name__)


def _task_metadata(task_id: object | None) -> AnalysisTaskMetadata:
    """Support topic-modeling routes with a task metadata helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support topic-modeling routes with a task metadata helper" behavior.
    """

    return AnalysisTaskMetadata(task_id=str(task_id) if task_id is not None else None)


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
    workspace_id: UUID,
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
    - Removes explicit topic-modeling task records for the requested workspace.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    task_manager = get_task_manager(user_id)
    task_ids = [
        task.task_id
        for task in task_manager.get_all_tasks()
        if task.workspace_id == workspace_id_str
        and task.request.__class__.__name__ == "TopicModelingRequest"
    ]
    for task_id in task_ids:
        task_manager.clear_task(task_id)

    worker_tm = workspace_manager.get_task_manager(user_id)
    for task_id in task_ids:
        await worker_tm.clear_task(task_id)

    return {
        "state": "successful",
        "message": "Topic modeling analysis results have been cleared.",
    }


@router.post("/topic-modeling", response_model=TopicModelingResponse)
async def run_topic_modeling(
    workspace_id: UUID,
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
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    return await submit_topic_modeling(
        user_id=user_id,
        workspace_id=workspace_id_str,
        workspace=ws,
        request=request,
    )


async def topic_modeling_task_result(
    workspace_id: str,
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return current status or final payload for a topic-modeling task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - shared analysis-task result route because topic-modeling result reads
      need the existing worker-sync and status normalization.

    Why:
    - Normalizes task lifecycle states into one response contract for UI polling.
    """
    user_id = current_user["id"]
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


def _resolve_topic_output_columns(original_columns: list[str]) -> tuple[str, str]:
    """Resolve the detached ``TOPIC_top1`` / ``TOPIC_distribution`` column names.

    Both generated columns are renamed to avoid clashing with source columns
    (and with each other). Used identically by the detach-options endpoint (to
    advertise the names) and the detach route (to match the names the client
    ticked), so they always agree.

    Used by:
    - ``topic_modeling_detach_options`` and ``detach_topic_modeling``.
    """
    taken = set(original_columns)
    top1_name = _resolve_topic_column_name(TOPIC_TOP1_COLUMN, taken)
    dist_name = _resolve_topic_column_name(
        TOPIC_DISTRIBUTION_OUTPUT_COLUMN, taken | {top1_name}
    )
    return top1_name, dist_name


async def topic_modeling_detach_options(
    workspace_id: UUID,
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """List detachable node/column options for a completed topic task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - shared analysis-task detach-options route because completed
      topic-modeling artifacts should be inspected through the task resource.

    Why:
    - Exposes artifact-backed node metadata so users can choose output columns safely.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    task = await _require_completed_topic_task(user_id, workspace_id_str, task_id)

    artifacts = _topic_artifacts_from_task(task)
    node_artifacts = artifacts.get("nodes") or []

    nodes: list[DetachNodeOption] = []
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
        topic_top1_name, topic_dist_name = _resolve_topic_output_columns(
            original_columns
        )
        nodes.append(
            DetachNodeOption(
                node_id=source_node.id,
                node_name=str(payload.get("node_name") or node_id),
                text_column=str(payload.get("text_column") or ""),
                available_columns=[topic_top1_name, topic_dist_name, *original_columns],
                # The generated topic columns are user-choosable and default-on;
                # source columns start unticked (matching concordance/quotation).
                disabled_columns=[],
                default_selected_columns=[topic_top1_name, topic_dist_name],
            )
        )

    return TopicModelingDetachOptionsResponse(
        state="successful",
        message="Topic detach options loaded",
        data={"nodes": nodes},
        metadata=_task_metadata(task_id),
    )


async def detach_topic_modeling(
    workspace_id: UUID,
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
    - shared analysis-task detachments route because topic output node creation
      is a task lifecycle operation.

        Why:
        - Materializes user-selected columns and topic labels as reusable workspace
            nodes without rerunning the model.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    task = await _require_completed_topic_task(user_id, workspace_id_str, task_id)

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
        # Resolve the two generated output column names the same way the
        # detach-options endpoint does, so they match what the client ticked.
        topic_top1_name, topic_dist_name = _resolve_topic_output_columns(
            original_columns
        )
        raw_selected = list((request.selected_columns or {}).get(node_id) or [])
        include_top1 = topic_top1_name in raw_selected
        include_distribution = topic_dist_name in raw_selected
        # Everything else must be a real source column.
        generated_names = {topic_top1_name, topic_dist_name}
        source_selected = [col for col in raw_selected if col not in generated_names]
        if not source_selected and not include_top1 and not include_distribution:
            raise InvalidInputError(
                f"No columns selected for node {node_id}",
            )
        invalid = [col for col in source_selected if col not in original_columns]
        if invalid:
            raise InvalidInputError(
                f"Invalid selected columns for node {node_id}: {invalid}",
            )

        projection = [pl.col(col) for col in source_selected]
        if include_top1:
            projection.append(pl.col(TOPIC_COLUMN).alias(topic_top1_name))
        if include_distribution:
            # The persisted distribution column is already the canonical TMDist
            # physical dtype (List(Struct{topic_id, proportion})); just rename it.
            projection.append(pl.col(TOPIC_DISTRIBUTION_COLUMN).alias(topic_dist_name))
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
        # `data/artifacts/` which gets cleaned by explicit task cleanup. A
        # detached node that still scanned them would silently corrupt later.
        # The top-level workspace data dir is protected by
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
        # Smart insertion: keep the detached topic node directly below its mother.
        ws.place_node_after_parent(new_node)

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
        # Smart insertion: keep the meanings node directly below the topic node it
        # was derived from.
        ws.place_node_after_parent(meanings_node)

        detached_nodes.append(
            TopicModelingDetachedNode(
                source_node_id=node_id,
                new_node_id=new_node.id,
                topic_meanings_node_id=meanings_node.id,
            )
        )

    update_workspace(user_id, workspace_id_str, ws)

    return TopicModelingDetachResponse(
        state="successful",
        message="Topic detach completed",
        data=TopicModelingDetachData(detached_nodes=detached_nodes),
        metadata=_task_metadata(task_id),
    )
