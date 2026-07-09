"""Topic-modeling task submission service.

Used by:
- ``topic_modeling.run_topic_modeling`` because the route should resolve the
  HTTP user/workspace boundary and delegate validation, artifact setup, task
  persistence, and worker submission here.

Flow:
- Validate selected nodes and build worker node descriptors.
- Normalize optional modeling parameters to existing defaults.
- Snapshot input nodes, persist the analysis task, and submit the worker job
  under a per-user/workspace lock.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from ....analysis.implementations.topic_modeling import (
    TopicModelingRequest as AnalysisTopicModelingRequest,
)
from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....core.exceptions import InvalidInputError, NotFoundError, WorkspaceNotFoundError
from ....core.worker_input_snapshots import create_worker_input_snapshot
from ....core.workspace import workspace_manager
from ....models.topic_modeling import TopicModelingRequest


_TOPIC_SUBMISSION_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


def _submission_lock(user_id: str, workspace_id: str) -> asyncio.Lock:
    """Return the per-user/workspace topic-modeling submission lock."""

    key = (user_id, workspace_id)
    lock = _TOPIC_SUBMISSION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _TOPIC_SUBMISSION_LOCKS[key] = lock
    return lock


def _prepare_artifact_target(user_id: str, workspace_id: str) -> tuple[Path, str]:
    """Create the artifact directory/prefix for one topic-modeling task."""

    workspace_artifacts_dir = workspace_manager.ensure_workspace_artifacts_dir(
        user_id, workspace_id
    )
    if workspace_artifacts_dir is None:
        raise WorkspaceNotFoundError("Workspace not found")
    return workspace_artifacts_dir, f"topic_modeling_{uuid4()}"


async def submit_topic_modeling(
    *,
    user_id: str,
    workspace_id: str,
    workspace: Any,
    request: TopicModelingRequest,
) -> dict[str, Any]:
    """Submit one topic-modeling worker task and return a running response."""

    if not request.node_ids:
        raise InvalidInputError("At least one node ID must be provided")

    node_infos: list[dict[str, object]] = []
    for node_id in request.node_ids:
        column_name = request.node_columns[node_id]
        if node_id not in workspace.nodes:
            raise NotFoundError(f"Node {node_id} not found")
        node_infos.append({"node_id": node_id, "text_column": column_name})

    task_id = str(uuid4())
    min_topic_size = request.min_topic_size if request.min_topic_size is not None else 10
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
    )

    async with _submission_lock(user_id, workspace_id):
        artifact_dir, artifact_prefix = _prepare_artifact_target(user_id, workspace_id)
        input_snapshot_dir = create_worker_input_snapshot(
            workspace_id=workspace_id,
            task_id=task_id,
            node_ids=request.node_ids,
            workspace=workspace,
            artifact_dir=artifact_dir,
        )
        analysis_task_manager = get_task_manager(user_id)
        analysis_task_manager.save_task(
            AnalysisTask(
                task_id=task_id,
                user_id=user_id,
                workspace_id=workspace_id,
                request=analysis_request,
                status=AnalysisStatus.RUNNING,
            )
        )
        worker_task_manager = workspace_manager.get_task_manager(user_id)
        await worker_task_manager.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type="topic_modeling",
            task_id=task_id,
            task_args={
                "input_snapshot_dir": str(input_snapshot_dir),
                "node_infos": node_infos,
                "artifact_dir": str(artifact_dir),
                "artifact_prefix": artifact_prefix,
                "min_topic_size": min_topic_size,
                "random_seed": random_seed,
                "representative_words_count": representative_words_count,
                "sample_fractions": request.sample_fractions,
            },
            task_name="Topic Modeling",
        )

    return {
        "state": "running",
        "message": "Topic Modeling analysis started",
        "data": None,
        "metadata": {"task_id": task_id},
    }
