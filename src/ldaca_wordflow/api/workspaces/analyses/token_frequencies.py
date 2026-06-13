"""Token Frequency analysis endpoints.

Artifact-first implementation:
- API gathers immutable payload (node corpora) in main process.
- Worker computes and writes Parquet artifacts.
- Result endpoint reconstructs response by lazy-scanning artifacts.

Used by:
- FastAPI workspace analysis routers, frontend analysis features, and backend tests because they need this unit's "Token Frequency analysis endpoints" behavior.

Flow:
- FastAPI mounts these routes through the workspace package router.
- Route handlers lock per user/workspace, hydrate token inputs, and submit artifact-first work.
- Result helpers lazy-scan Parquet artifacts, apply requested limits, and synchronize task state.
- Responses return task metadata, frequency tables, preference updates, or clear-task results.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import polars as pl
from fastapi import APIRouter, Depends, HTTPException

from ....analysis.implementations.token_frequency import (
    TokenFrequencyRequest as AnalysisTokenFrequencyRequest,
)
from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....core.analysis_helpers import sanitize_stop_words
from ....core.auth import get_current_user
from ....core.exceptions import (
    InternalServiceError,
    InvalidInputError,
    NoActiveWorkspaceError,
    NotFoundError,
    TaskNotFoundError,
    WorkspaceNotFoundError,
)
from ....core.tokens_cache import hydrate_tokenization_lazyframe
from ....core.workspace import workspace_manager
from ....models import (
    AnalysisClearResponse,
    TokenFrequencyPreferenceUpdateRequest,
    TokenFrequencyRequest,
    TokenFrequencyResponse,
)
from ..utils import ensure_task_synced

router = APIRouter(prefix="/workspaces")
logger = logging.getLogger(__name__)

_TOKEN_FREQ_SUBMISSION_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


def _token_freq_submission_lock(user_id: str, workspace_id: str) -> asyncio.Lock:
    """Support token-frequency routes with a token freq submission lock helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support token-frequency routes with a token freq submission lock helper" behavior.
    """

    key = (user_id, workspace_id)
    lock = _TOKEN_FREQ_SUBMISSION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _TOKEN_FREQ_SUBMISSION_LOCKS[key] = lock
    return lock


DEFAULT_TOKEN_LIMIT = 25
SERVER_LIMIT_MULTIPLIER = 5
MAX_SERVER_TOKEN_LIMIT = 5000


@dataclass(frozen=True)
class TokenNodeArtifact:
    """TokenNodeArtifact supports token-frequency routes by modeling token node artifact.

    Used by:
    - backend API routes because they need this unit's "TokenNodeArtifact supports token-frequency routes by modeling token node artifact" behavior.
    """

    node_id: str
    node_name: str
    token_parquet_path: Path


@dataclass(frozen=True)
class TokenFrequencyArtifacts:
    """TokenFrequencyArtifacts supports token-frequency routes by modeling token frequency artifacts.

    Used by:
    - backend API routes because they need this unit's "TokenFrequencyArtifacts supports token-frequency routes by modeling token frequency artifacts" behavior.
    """

    nodes: tuple[TokenNodeArtifact, ...]
    statistics_parquet_path: Path | None


@router.delete("/token-frequencies", response_model=AnalysisClearResponse)
async def clear_token_frequencies(
    current_user=Depends(get_current_user),
):
    """Clear Token Frequency analysis state for a workspace.

    Legacy broad clear endpoint: removes all token-frequency task records for
    the active workspace. Tabbed clients should normally clear by explicit
    task_id through `/api/tasks/clear` instead.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI DELETE /token-frequencies route because they need this unit's "Clear Token Frequency analysis state for a workspace" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    task_manager = get_task_manager(user_id)
    task_ids = _token_frequency_task_ids(user_id, workspace_id)
    for task_id in task_ids:
        task_manager.clear_task(task_id)

    worker_tm = workspace_manager.get_task_manager(user_id)
    for task_id in task_ids:
        await worker_tm.clear_task(task_id)

    return {
        "state": "successful",
        "message": "Token frequencies cleared successfully.",
    }


def _coerce_limit_value(value: Any) -> int:
    """Coerce token-limit input to a safe positive integer.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Coerce token-limit input to a safe positive integer" behavior.
    """
    try:
        candidate = int(value)
    except TypeError, ValueError:
        return DEFAULT_TOKEN_LIMIT
    return candidate if candidate > 0 else DEFAULT_TOKEN_LIMIT


def _prepare_token_artifact_target(user_id: str, workspace_id: str) -> tuple[Path, str]:
    """Prepare token artifact target data consumed by token-frequency routes.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Prepare token artifact target data consumed by token-frequency routes" behavior.
    """

    workspace_artifacts_dir = workspace_manager.ensure_workspace_artifacts_dir(
        user_id, workspace_id
    )
    if workspace_artifacts_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    artifact_prefix = f"token_frequencies_{uuid4()}"
    return workspace_artifacts_dir, artifact_prefix


def _task_result_payload(task: AnalysisTask) -> dict:
    """Support token-frequency routes with a task result payload helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support token-frequency routes with a task result payload helper" behavior.
    """

    if task.result is None:
        return {}
    payload = task.result.to_json()
    if not isinstance(payload, dict):
        return {}
    return payload


def _invalid_artifact_manifest() -> HTTPException:
    """Support token-frequency routes with an invalid artifact manifest helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support token-frequency routes with an invalid artifact manifest helper" behavior.
    """

    return HTTPException(
        status_code=500,
        detail="Token-frequency artifact manifest is invalid",
    )


def _node_artifact_from_entry(entry: object) -> TokenNodeArtifact:
    """Support token-frequency routes with a node artifact from entry helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support token-frequency routes with a node artifact from entry helper" behavior.
    """

    if not isinstance(entry, dict):
        raise _invalid_artifact_manifest()

    raw_entry = cast(dict[str, object], entry)
    node_id = str(raw_entry.get("node_id") or "")
    token_path = str(raw_entry.get("token_parquet_path") or "")
    if not node_id or not token_path:
        raise _invalid_artifact_manifest()

    return TokenNodeArtifact(
        node_id=node_id,
        node_name=str(raw_entry.get("node_name") or node_id),
        token_parquet_path=Path(token_path),
    )


def _token_artifacts_from_task(
    task: AnalysisTask,
) -> tuple[dict, TokenFrequencyArtifacts]:
    """Run the token artifacts from task background job submitted by API routes.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Run the token artifacts from task background job submitted by API routes" behavior.
    """

    payload = _task_result_payload(task)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise NotFoundError(
            "Token-frequency artifacts are not available for this task",
        )
    node_artifacts = artifacts.get("nodes")
    if not isinstance(node_artifacts, list):
        raise _invalid_artifact_manifest()

    stats_path = artifacts.get("statistics_parquet_path")
    if stats_path is not None and not isinstance(stats_path, str):
        raise _invalid_artifact_manifest()

    return payload, TokenFrequencyArtifacts(
        nodes=tuple(_node_artifact_from_entry(entry) for entry in node_artifacts),
        statistics_parquet_path=Path(stats_path) if stats_path else None,
    )


def _server_limit(token_limit: int) -> int:
    """Support token-frequency routes with a server limit helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support token-frequency routes with a server limit helper" behavior.
    """

    return min(
        max(token_limit * SERVER_LIMIT_MULTIPLIER, DEFAULT_TOKEN_LIMIT),
        MAX_SERVER_TOKEN_LIMIT,
    )


def _is_token_frequency_task(task: AnalysisTask, workspace_id: str) -> bool:
    """Return whether an analysis task belongs to token-frequency for a workspace."""
    if task.workspace_id != workspace_id:
        return False
    if isinstance(task.request, AnalysisTokenFrequencyRequest):
        return True
    payload = task.request.model_dump() if hasattr(task.request, "model_dump") else {}
    return (
        isinstance(payload, dict)
        and "node_ids" in payload
        and "node_columns" in payload
    )


def _token_frequency_task_ids(user_id: str, workspace_id: str) -> list[str]:
    """List token-frequency task ids for broad workspace clear operations."""
    task_manager = get_task_manager(user_id)
    tasks = [
        task
        for task in task_manager.get_all_tasks()
        if _is_token_frequency_task(task, workspace_id)
    ]
    tasks.sort(key=lambda task: task.updated_at or task.created_at, reverse=True)
    return [task.task_id for task in tasks]


def _safe_float(value: Any) -> float | str | None:
    """Create safe float values for token-frequency routes.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Create safe float values for token-frequency routes" behavior.
    """

    if value is None:
        return None
    try:
        numeric = float(value)
    except TypeError, ValueError:
        return None
    if math.isnan(numeric):
        return None
    if numeric == math.inf:
        return "+Inf"
    if numeric == -math.inf:
        return "-Inf"
    return numeric


def _rebuild_token_result(task: AnalysisTask) -> dict:
    """Support token-frequency routes with a rebuild token result helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support token-frequency routes with a rebuild token result helper" behavior.
    """

    payload, artifacts = _token_artifacts_from_task(task)

    request_payload = task.request.model_dump()
    token_limit = _coerce_limit_value(request_payload.get("token_limit"))
    stop_words = sanitize_stop_words(request_payload.get("stop_words"))
    stop_word_set = set(stop_words)

    node_results: dict[str, dict] = {}
    for node_artifact in artifacts.nodes:
        if not node_artifact.token_parquet_path.exists():
            raise NotFoundError(
                f"Token artifact missing for node {node_artifact.node_id}",
            )
        token_df = cast(
            pl.DataFrame, pl.scan_parquet(node_artifact.token_parquet_path).collect()
        )
        rows = token_df.to_dicts()
        total_tokens = len(rows)
        node_results[node_artifact.node_id] = {
            "data": [
                {
                    "token": str(row.get("token") or ""),
                    "frequency": int(row.get("frequency") or 0),
                }
                for row in rows
            ],
            "columns": ["token", "frequency"],
            "metadata": {
                "applied_server_limit": None,
                "total_tokens_before_limit": total_tokens,
                "total_tokens_returned": total_tokens,
                "truncated": False,
                "token_limit": token_limit,
                "node_id": node_artifact.node_id,
                "display_name": node_artifact.node_name,
                "node_name": node_artifact.node_name,
            },
        }

    statistics_payload = None
    if artifacts.statistics_parquet_path is not None:
        if not artifacts.statistics_parquet_path.exists():
            raise NotFoundError("Token statistics artifact is missing")
        stats_df = cast(
            pl.DataFrame, pl.scan_parquet(artifacts.statistics_parquet_path).collect()
        )
        statistics_payload = [
            {
                "token": str(row.get("token") or ""),
                "freq_reference": int(row.get("freq_corpus_0") or 0),
                "freq_study": int(row.get("freq_corpus_1") or 0),
                "expected_reference": _safe_float(row.get("expected_0")),
                "expected_study": _safe_float(row.get("expected_1")),
                "reference_total": int(row.get("corpus_0_total") or 0),
                "study_total": int(row.get("corpus_1_total") or 0),
                "percent_reference": _safe_float(row.get("percent_corpus_0")),
                "percent_study": _safe_float(row.get("percent_corpus_1")),
                "percent_diff": _safe_float(row.get("percent_diff")),
                "log_likelihood_llv": _safe_float(row.get("log_likelihood_llv")),
                "bayes_factor_bic": _safe_float(row.get("bayes_factor_bic")),
                "effect_size_ell": _safe_float(row.get("effect_size_ell")),
                "relative_risk": _safe_float(row.get("relative_risk")),
                "log_ratio": _safe_float(row.get("log_ratio")),
                "odds_ratio": _safe_float(row.get("odds_ratio")),
                "significance": str(row.get("significance") or ""),
            }
            for row in stats_df.to_dicts()
        ]

    server_limit = _server_limit(token_limit)
    analysis_params = {
        "node_ids": list(request_payload.get("node_ids") or []),
        "node_columns": dict(request_payload.get("node_columns") or {}),
        "token_limit": token_limit,
        "server_limit": server_limit,
        "stop_words": stop_words,
    }
    metadata = {
        "token_limit": token_limit,
        "server_limit": server_limit,
        "stop_words": stop_words,
        "node_display_names": {
            node.node_id: node.node_name for node in artifacts.nodes
        },
    }

    return {
        "state": payload.get("state") or "successful",
        "message": payload.get("message")
        or f"Successfully calculated token frequencies for {len(node_results)} node(s)",
        "data": node_results,
        "statistics": statistics_payload,
        "token_limit": token_limit,
        "analysis_params": analysis_params,
        "metadata": metadata,
        "stop_words": stop_words,
    }


@router.get(
    "/token-frequencies/tasks/{task_id}/request",
    response_model=AnalysisTokenFrequencyRequest,
)
async def token_frequencies_task_request(
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return stored request payload for a token-frequencies task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /token-frequencies/tasks/{task_id}/request route because they need this unit's "Return stored request payload for a token-frequencies task" behavior.
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
    "/token-frequencies/tasks/{task_id}/result",
    response_model=TokenFrequencyResponse | None,
)
async def token_frequencies_task_result(
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return normalized token-frequency result payload for one task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI GET /token-frequencies/tasks/{task_id}/result route because they need this unit's "Return normalized token-frequency result payload for one task" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    task_manager = get_task_manager(user_id)

    task = await ensure_task_synced(user_id, workspace_id, task_id, task_manager)
    if not task:
        return None

    if task.status in (AnalysisStatus.PENDING, AnalysisStatus.RUNNING):
        return {
            "state": "running",
            "message": "Token frequency analysis is still running",
            "data": None,
            "metadata": {"task_id": task_id},
        }

    if task.status == AnalysisStatus.FAILED:
        return {
            "state": "failed",
            "message": task.error or "Token frequency analysis failed",
            "data": None,
            "metadata": {"task_id": task_id},
        }

    if not task.result:
        return {
            "state": "running",
            "message": "Token frequency analysis is finalizing",
            "data": None,
            "metadata": {"task_id": task_id},
        }

    return _rebuild_token_result(task)


@router.post(
    "/token-frequencies/tasks/{task_id}/result", response_model=AnalysisClearResponse
)
async def update_token_frequencies_task_result(
    task_id: str,
    updates: TokenFrequencyPreferenceUpdateRequest | None,
    current_user: dict = Depends(get_current_user),
):
    """Persist token-frequency preference overrides on an existing task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /token-frequencies/tasks/{task_id}/result route because they need this unit's "Persist token-frequency preference overrides on an existing task" behavior.
    """
    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    if not workspace_id:
        raise NoActiveWorkspaceError("No active workspace selected")
    task_manager = get_task_manager(user_id)
    task = task_manager.get_task(task_id)
    if not task:
        raise NotFoundError("No token frequency task found")
    request_payload = task.request.model_dump()

    if updates is not None:
        updates_payload = updates.model_dump(exclude_unset=True)
        if "token_limit" in updates_payload:
            request_payload["token_limit"] = _coerce_limit_value(
                updates_payload.get("token_limit")
            )
        if "stop_words" in updates_payload:
            request_payload["stop_words"] = sanitize_stop_words(
                updates_payload.get("stop_words")
            )

    try:
        task.request = AnalysisTokenFrequencyRequest(**request_payload)
        task_manager.save_task(task)
    except Exception as exc:  # pragma: no cover
        raise InternalServiceError(
            f"Failed to persist token frequency preferences: {exc}",
        )
    return {"state": "successful", "message": "saved"}


@router.post(
    "/token-frequencies",
    response_model=TokenFrequencyResponse,
    summary="Calculate token frequencies for selected nodes",
    description="Calculate and compare token frequencies across one or two nodes using polars-text",
)
async def calculate_token_frequencies(
    request: TokenFrequencyRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit token-frequency analysis as a worker-backed artifact-first task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI POST /token-frequencies route because they need this unit's "Submit token-frequency analysis as a worker-backed artifact-first task" behavior.
    """

    user_id = current_user["id"]
    workspace_id = workspace_manager.get_current_workspace_id(user_id)
    ws = workspace_manager.get_current_workspace(user_id)
    if not workspace_id or ws is None:
        raise NoActiveWorkspaceError("No active workspace selected")
    tm = workspace_manager.get_task_manager(user_id)

    if not request.node_ids:
        raise InvalidInputError("At least one node ID must be provided")
    if len(request.node_ids) > 2:
        raise InvalidInputError("Maximum of 2 nodes can be compared")
    requested_token_limit = getattr(request, "token_limit", None)
    effective_limit = (
        requested_token_limit
        if requested_token_limit is not None and requested_token_limit > 0
        else DEFAULT_TOKEN_LIMIT
    )
    if requested_token_limit is not None and requested_token_limit <= 0:
        raise InvalidInputError("token_limit must be a positive integer")
    tokenizer_model = (request.tokenizer_model or "").strip()
    requested_node_tokenizer_models = {
        node_id: model.strip()
        for node_id, model in (request.node_tokenizer_models or {}).items()
        if model and model.strip()
    }

    # Prepare the artifact target early so the tokens-mode spill files
    # share a parent directory with the eventual frequency parquets and
    # get cleaned up together when the workspace's artifact dir is
    # cleared. Computing it here also unifies the resume-from-error path.
    artifact_dir, artifact_prefix = _prepare_token_artifact_target(
        user_id, workspace_id
    )

    node_corpora: dict[str, list[str]] = {}
    node_token_streams: dict[str, str] = {}
    node_tokenizer_models: dict[str, str] = {}
    node_display_names: dict[str, str] = {}
    for node_id in request.node_ids:
        node = ws.nodes[node_id]
        node_data = node.data

        column_name = request.node_columns[node_id]

        tokenization_col = node.find_tokenization_column(column_name)
        if tokenization_col is not None:
            tokenization_registry = getattr(node, "tokenization", {})
            tokenization_meta = (
                tokenization_registry.get(column_name, {})
                if isinstance(tokenization_registry, dict)
                else {}
            )
            model = (
                tokenization_meta.get("model")
                if isinstance(tokenization_meta, dict)
                else None
            )
            if isinstance(model, str) and model.strip():
                node_tokenizer_models[node_id] = model.strip()
            node_data = hydrate_tokenization_lazyframe(
                node=node,
                source_column=column_name,
                user_id=user_id,
            )
            # Spill the explode-flattened tokens to a parquet via streaming
            # sink instead of materialising a
            # ``list[list[str]]`` of Python str objects (which for 10 k
            # CJK docs with ~1 k tokens each was ~500 MB of pure PyObject
            # overhead). The worker scans the parquet and computes
            # frequencies via ``group_by.len()`` in Polars — never
            # touching Python until the small summary frame at the end.
            stream_path = (
                artifact_dir / f"{artifact_prefix}_tokens_stream_{node_id}.parquet"
            )
            (
                node_data.select(
                    pl.col(tokenization_col)
                    .list.eval(pl.element().struct.field("token"))
                    .explode()
                    .alias("token")
                )
                .filter(pl.col("token").is_not_null())
                .sink_parquet(stream_path)
            )
            node_token_streams[node_id] = str(stream_path)
        else:
            docs_df = node_data.select(
                pl.col(column_name).alias("__doc_col__")
            ).collect()
            node_corpora[node_id] = [
                str(v) if v is not None else ""
                for v in docs_df["__doc_col__"].to_list()
            ]
        node_display_names[node_id] = str(getattr(node, "name", None) or node_id)

    node_tokenizer_models.update(
        {
            node_id: requested_node_tokenizer_models.get(node_id) or tokenizer_model
            for node_id in node_corpora
        }
    )
    missing_tokenizer_model_node_ids = [
        node_id for node_id, model in node_tokenizer_models.items() if not model
    ]
    if missing_tokenizer_model_node_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "node_tokenizer_models must include a tokenizer model for raw-text nodes: "
                + ", ".join(missing_tokenizer_model_node_ids)
            ),
        )

    requested_stop_words = sanitize_stop_words(request.stop_words)

    submission_lock = _token_freq_submission_lock(user_id, workspace_id)
    async with submission_lock:
        task_info = await tm.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type="token_frequencies",
            task_args={
                "node_corpora": node_corpora,
                "node_token_streams": node_token_streams,
                "node_display_names": node_display_names,
                "artifact_dir": str(artifact_dir),
                "artifact_prefix": artifact_prefix,
                "token_limit": effective_limit,
                "stop_words": requested_stop_words,
                "tokenizer_model": tokenizer_model,
                "node_tokenizer_models": node_tokenizer_models,
            },
        )

    analysis_request = AnalysisTokenFrequencyRequest(
        node_ids=request.node_ids,
        node_columns=request.node_columns,
        token_limit=effective_limit,
        stop_words=requested_stop_words,
        tokenizer_model=tokenizer_model,
        node_tokenizer_models=node_tokenizer_models,
    )

    task_manager = get_task_manager(user_id)
    task_manager.save_task(
        AnalysisTask(
            task_id=task_info.id,
            user_id=user_id,
            workspace_id=workspace_id,
            request=analysis_request,
            status=AnalysisStatus.RUNNING,
        )
    )

    return {
        "state": "running",
        "message": "Token frequency analysis started",
        "data": None,
        "token_limit": effective_limit,
        "stop_words": requested_stop_words,
        "metadata": {"task_id": task_info.id},
    }
