"""Token-frequency task submission service.

Used by:
- ``token_frequencies.calculate_token_frequencies`` because the route should
  remain an HTTP boundary while this module owns validation, snapshot creation,
  analysis task persistence, and worker-task submission.

Flow:
- Validate node selection, token-limit, and tokenizer model inputs.
- Resolve per-node tokenizer models from request values or registered
  tokenization metadata.
- Create immutable worker input snapshots and persist the analysis task record.
- Submit the worker task under a per-user/workspace lock and return the running
  response payload consumed by generated clients.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from ....analysis.implementations.token_frequency import (
    TokenFrequencyRequest as AnalysisTokenFrequencyRequest,
)
from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....core.analysis_helpers import sanitize_stop_words
from ....core.exceptions import InvalidInputError, NotFoundError, WorkspaceNotFoundError
from ....core.worker_input_snapshots import create_worker_input_snapshot
from ....core.workspace import workspace_manager
from ....models import TokenFrequencyRequest


DEFAULT_TOKEN_LIMIT = 25
SERVER_LIMIT_MULTIPLIER = 5
MAX_SERVER_TOKEN_LIMIT = 5000

_TOKEN_FREQ_SUBMISSION_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


def _submission_lock(user_id: str, workspace_id: str) -> asyncio.Lock:
    """Return the per-user/workspace token-frequency submission lock.

    Used by:
    - ``submit_token_frequency_analysis`` so concurrent submissions for one
      workspace do not race while registering worker tasks.
    """

    key = (user_id, workspace_id)
    lock = _TOKEN_FREQ_SUBMISSION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _TOKEN_FREQ_SUBMISSION_LOCKS[key] = lock
    return lock


def _prepare_artifact_target(user_id: str, workspace_id: str) -> tuple[Path, str]:
    """Create the artifact directory/prefix for one token-frequency task.

    Used by:
    - ``submit_token_frequency_analysis`` before snapshotting inputs and
      submitting the worker task.
    """

    workspace_artifacts_dir = workspace_manager.ensure_workspace_artifacts_dir(
        user_id, workspace_id
    )
    if workspace_artifacts_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    return workspace_artifacts_dir, f"token_frequencies_{uuid4()}"


def _effective_token_limit(request: TokenFrequencyRequest) -> int:
    """Validate and normalize the requested token limit."""

    requested_token_limit = request.token_limit
    if requested_token_limit is not None and requested_token_limit <= 0:
        raise InvalidInputError("token_limit must be a positive integer")
    if requested_token_limit is None:
        return DEFAULT_TOKEN_LIMIT
    return requested_token_limit


def _requested_node_tokenizer_models(request: TokenFrequencyRequest) -> dict[str, str]:
    """Return non-empty per-node tokenizer model overrides from the request."""

    return {
        node_id: model.strip()
        for node_id, model in (request.node_tokenizer_models or {}).items()
        if model and model.strip()
    }


def _tokenization_model_for(node: Any, column_name: str) -> str | None:
    """Return the model registered for an existing tokenization column."""

    tokenization_registry = getattr(node, "tokenization", {})
    tokenization_meta = (
        tokenization_registry.get(column_name, {})
        if isinstance(tokenization_registry, dict)
        else {}
    )
    model = tokenization_meta.get("model") if isinstance(tokenization_meta, dict) else None
    return model.strip() if isinstance(model, str) and model.strip() else None


def _resolve_node_tokenizer_models(
    workspace: Any,
    request: TokenFrequencyRequest,
    tokenizer_model: str,
) -> dict[str, str]:
    """Resolve tokenizer models required by the selected token-frequency nodes.

    Used by:
    - ``submit_token_frequency_analysis`` because worker input should name the
      tokenizer model for every raw-text node, while tokenized nodes should use
      their registered metadata.
    """

    requested_models = _requested_node_tokenizer_models(request)
    resolved_models: dict[str, str] = {}
    raw_nodes_missing_model: list[str] = []

    for node_id in request.node_ids:
        node = workspace.nodes.get(node_id)
        if node is None:
            raise NotFoundError(f"Node {node_id} not found")

        column_name = request.node_columns.get(node_id)
        if not column_name:
            raise InvalidInputError(f"Missing column selection for node {node_id}")

        tokenization_col = node.find_tokenization_column(column_name)
        if tokenization_col is None:
            model = requested_models.get(node_id) or tokenizer_model
            if model:
                resolved_models[node_id] = model
            else:
                raw_nodes_missing_model.append(node_id)
            continue

        registered_model = _tokenization_model_for(node, column_name)
        if registered_model is not None:
            resolved_models[node_id] = registered_model

    if raw_nodes_missing_model:
        raise InvalidInputError(
            "node_tokenizer_models must include a tokenizer model for raw-text nodes: "
            + ", ".join(raw_nodes_missing_model)
        )
    return resolved_models


async def submit_token_frequency_analysis(
    *,
    user_id: str,
    workspace_id: str,
    workspace: Any,
    request: TokenFrequencyRequest,
) -> dict[str, Any]:
    """Submit a token-frequency analysis task and return the running response.

    Used by:
    - ``token_frequencies.calculate_token_frequencies`` after the route has
      resolved authentication and explicit workspace identity.

    Flow:
    - Validate request-level limits and node selections.
    - Prepare durable artifact paths and immutable worker-input snapshots.
    - Persist the analysis-task record before enqueueing the worker task.
    - Return the stable ``running`` response with the task id.
    """

    if not request.node_ids:
        raise InvalidInputError("At least one node ID must be provided")
    if len(request.node_ids) > 2:
        raise InvalidInputError("Maximum of 2 nodes can be compared")

    effective_limit = _effective_token_limit(request)
    tokenizer_model = (request.tokenizer_model or "").strip()
    node_tokenizer_models = _resolve_node_tokenizer_models(
        workspace,
        request,
        tokenizer_model,
    )
    stop_words = sanitize_stop_words(request.stop_words)

    artifact_dir, artifact_prefix = _prepare_artifact_target(user_id, workspace_id)
    task_id = str(uuid4())
    input_snapshot_dir = create_worker_input_snapshot(
        user_id=user_id,
        workspace_id=workspace_id,
        task_id=task_id,
        node_ids=request.node_ids,
        workspace=workspace,
        artifact_dir=artifact_dir,
    )
    analysis_request = AnalysisTokenFrequencyRequest(
        node_ids=request.node_ids,
        node_columns=request.node_columns,
        token_limit=effective_limit,
        stop_words=stop_words,
        tokenizer_model=tokenizer_model,
        node_tokenizer_models=node_tokenizer_models,
    )

    task_manager = get_task_manager(user_id)
    task_manager.save_task(
        AnalysisTask(
            task_id=task_id,
            user_id=user_id,
            workspace_id=workspace_id,
            request=analysis_request,
            status=AnalysisStatus.RUNNING,
        )
    )

    worker_task_manager = workspace_manager.get_task_manager(user_id)
    async with _submission_lock(user_id, workspace_id):
        await worker_task_manager.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type="token_frequencies",
            task_id=task_id,
            task_args={
                "input_snapshot_dir": str(input_snapshot_dir),
                "node_ids": request.node_ids,
                "node_columns": request.node_columns,
                "artifact_dir": str(artifact_dir),
                "artifact_prefix": artifact_prefix,
                "token_limit": effective_limit,
                "stop_words": stop_words,
                "tokenizer_model": tokenizer_model,
                "node_tokenizer_models": node_tokenizer_models,
            },
        )

    return {
        "state": "running",
        "message": "Token frequency analysis started",
        "data": None,
        "token_limit": effective_limit,
        "stop_words": stop_words,
        "metadata": {"task_id": task_id},
    }
