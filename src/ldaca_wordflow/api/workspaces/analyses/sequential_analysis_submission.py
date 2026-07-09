"""Sequential-analysis task submission service.

Used by:
- ``sequential_analysis.run_sequential_analysis`` because the route should only
  resolve HTTP identity while this module validates request semantics, snapshots
  the selected node, persists the analysis task, and submits the worker job.

Flow:
- Validate source node, group-by limit, and datetime frequency values.
- Build the analysis request payload with the path node id included.
- Snapshot the input node and register the task as running.
- Submit the worker task and return the running response payload.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from ....analysis.implementations.sequential_analysis import (
    SequentialAnalysisRequest as AnalysisSequentialAnalysisRequest,
)
from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....core.exceptions import InternalServiceError, InvalidInputError, NotFoundError
from ....core.worker_input_snapshots import create_worker_input_snapshot
from ....core.workspace import workspace_manager
from ....models import SequentialAnalysisRequest


logger = logging.getLogger(__name__)
SEQUENTIAL_TASK = "sequential_analysis"
VALID_DATETIME_FREQUENCIES = [
    "hourly",
    "daily",
    "weekly",
    "monthly",
    "quarterly",
    "yearly",
    "custom",
]


async def submit_sequential_analysis(
    *,
    user_id: str,
    workspace_id: str,
    workspace: Any,
    node_id: str,
    request: SequentialAnalysisRequest,
) -> dict[str, Any]:
    """Submit one sequential-analysis worker task.

    Used by:
    - ``sequential_analysis.run_sequential_analysis`` after the route resolves
      the authenticated user and explicit workspace path id.
    """

    try:
        if node_id not in workspace.nodes:
            raise NotFoundError(f"Node {node_id} not found")

        if request.group_by_columns and len(request.group_by_columns) > 3:
            raise InvalidInputError("Maximum 3 group by columns allowed")

        if (
            request.column_type == "datetime"
            and request.frequency not in VALID_DATETIME_FREQUENCIES
        ):
            raise InvalidInputError(
                f"Invalid frequency '{request.frequency}'. "
                f"Valid options: {VALID_DATETIME_FREQUENCIES}",
            )

        request_payload = request.model_dump()
        request_payload["node_id"] = node_id
        analysis_request = AnalysisSequentialAnalysisRequest(**request_payload)
        task_id = str(uuid4())
        artifact_dir = workspace_manager.ensure_workspace_artifacts_dir(
            user_id, workspace_id
        )
        if artifact_dir is None:
            raise InternalServiceError("Workspace artifacts directory is unavailable")
        input_snapshot_dir = create_worker_input_snapshot(
            workspace_id=workspace_id,
            task_id=task_id,
            node_ids=[node_id],
            workspace=workspace,
            artifact_dir=artifact_dir,
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
        await worker_task_manager.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type=SEQUENTIAL_TASK,
            task_id=task_id,
            task_args={
                "input_snapshot_dir": str(input_snapshot_dir),
                "node_id": node_id,
                "request_payload": request_payload,
            },
            task_name="Sequential Analysis",
        )
    except (InvalidInputError, NotFoundError):
        raise
    except Exception as exc:  # pragma: no cover
        logger.error("Unexpected sequential analysis error: %s", exc, exc_info=True)
        raise InternalServiceError(f"Internal server error: {exc}") from exc

    return {
        "state": "running",
        "data": None,
        "columns": None,
        "total_records": None,
        "chart_type": None,
        "metadata": {"task_id": task_id},
    }
