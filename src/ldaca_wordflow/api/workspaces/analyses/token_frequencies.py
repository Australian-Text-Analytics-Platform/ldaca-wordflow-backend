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

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import polars as pl
from fastapi import APIRouter, Depends

from ....analysis.implementations.token_frequency import (
    TokenFrequencyRequest as AnalysisTokenFrequencyRequest,
)
from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....core.analysis_helpers import sanitize_stop_words
from ....core.auth import get_current_user
from ....core.exceptions import (
    InternalServiceError,
    NotFoundError,
    TaskNotFoundError,
)
from ....core.workspace import workspace_manager
from ....models.analysis_common import AnalysisClearResponse
from ....models.token_frequencies import (
    TokenFrequencyPreferenceUpdateRequest,
    TokenFrequencyRequest,
    TokenFrequencyResponse,
)
from ..utils import ensure_task_synced, require_workspace
from .token_frequency_submission import (
    DEFAULT_TOKEN_LIMIT,
    MAX_SERVER_TOKEN_LIMIT,
    SERVER_LIMIT_MULTIPLIER,
    submit_token_frequency_analysis,
)

router = APIRouter(
    prefix="/workspaces/{workspace_id:uuid}",
)
logger = logging.getLogger(__name__)

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
    workspace_id: UUID,
    current_user=Depends(get_current_user),
):
    """Clear Token Frequency analysis state for a workspace.

    Legacy broad clear endpoint: removes all token-frequency task records for
    the active workspace. Tabbed clients should normally clear by explicit
    task_id through `DELETE /api/tasks/{task_id}` instead.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - Frontend and API clients through the FastAPI DELETE /token-frequencies route because they need this unit's "Clear Token Frequency analysis state for a workspace" behavior.
    """
    user_id = current_user["id"]
    workspace_id_str = str(workspace_id)
    task_manager = get_task_manager(user_id)
    task_ids = _token_frequency_task_ids(user_id, workspace_id_str)
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


def _invalid_artifact_manifest() -> InternalServiceError:
    """Support token-frequency routes with an invalid artifact manifest helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support token-frequency routes with an invalid artifact manifest helper" behavior.
    """

    return InternalServiceError("Token-frequency artifact manifest is invalid")


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


async def token_frequencies_task_result(
    workspace_id: str,
    task_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Return normalized token-frequency result payload for one task.

    Flow:
    - Resolve authentication and request parameters from FastAPI dependencies.
    - Delegate validation, manager calls, artifacts, or state changes to the owning helper.
    - Shape the response payload or raise the HTTP error the client should see.

    Used by:
    - shared analysis-task result route because token-frequency result reads
      need the existing worker-sync and artifact rebuild behavior.
    """
    user_id = current_user["id"]
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


async def update_token_frequencies_task_result(
    workspace_id: str,
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
    - shared analysis-task preferences route because token-limit and stop-word
      changes update task presentation/request preferences.
    """
    user_id = current_user["id"]
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
    workspace_id: UUID,
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
    workspace_id_str = str(workspace_id)
    ws = require_workspace(user_id, workspace_id_str)
    return await submit_token_frequency_analysis(
        user_id=user_id,
        workspace_id=workspace_id_str,
        workspace=ws,
        request=request,
    )
