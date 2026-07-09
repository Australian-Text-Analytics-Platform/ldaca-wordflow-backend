"""Concordance task submission service."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ....analysis.implementations.concordance import (
    ConcordanceRequest as AnalysisConcordanceRequest,
)
from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....core.exceptions import (
    InternalServiceError,
    InvalidInputError,
    NotFoundError,
    WorkspaceNotFoundError,
)
from ....core.worker_input_snapshots import create_worker_input_snapshot
from ....core.workspace import workspace_manager
from ....models import ConcordanceAnalysisRequest
from .concordance_core import DEFAULT_CONCORDANCE_PAGE, normalize_saved_request


async def submit_concordance_analysis(
    *,
    user_id: str,
    workspace_id: str,
    workspace: Any,
    request: ConcordanceAnalysisRequest,
) -> dict[str, Any]:
    """Submit one concordance worker task and return the running response."""

    task_manager = get_task_manager(user_id)
    if not request.node_ids:
        raise InvalidInputError("At least one node ID must be provided")
    try:
        for node_id in request.node_ids:
            if node_id not in workspace.nodes:
                raise NotFoundError(f"Node {node_id} not found")
            if not request.node_columns.get(node_id):
                raise InvalidInputError(f"Missing text column for node {node_id}")

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
        workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
        if workspace_dir is None:
            raise WorkspaceNotFoundError("Workspace not found")
        input_snapshot_dir = create_worker_input_snapshot(
            workspace_id=workspace_id,
            task_id=task_id,
            node_ids=request.node_ids,
            workspace=workspace,
            artifact_dir=workspace_dir / "data" / "artifacts",
        )
        task_manager.save_task(
            AnalysisTask(
                task_id=task_id,
                user_id=user_id,
                workspace_id=workspace_id,
                request=analysis_request,
                status=AnalysisStatus.RUNNING,
            )
        )
        normalized_request = normalize_saved_request(analysis_request.model_dump()) or {}
        normalized_request.setdefault("page", DEFAULT_CONCORDANCE_PAGE)
        if request.sort_by:
            normalized_request["sort_by"] = request.sort_by
        normalized_request["descending"] = request.descending

        worker_task_manager = workspace_manager.get_task_manager(user_id)
        await worker_task_manager.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type="concordance",
            task_id=task_id,
            task_args={
                "input_snapshot_dir": str(input_snapshot_dir),
                "request_payload": normalized_request,
            },
            task_name="Concordance",
        )

        return {
            "state": "running",
            "message": "Concordance analysis started",
            "data": {},
            "analysis_params": normalized_request,
            "metadata": {"task_id": task_id},
        }
    except Exception as exc:
        raise InternalServiceError(f"Failed to run concordance: {exc}") from exc
