"""Quotation task submission service."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ....analysis.implementations.quotation import (
    QuotationRequest as AnalysisQuotationRequest,
)
from ....analysis.manager import get_task_manager
from ....analysis.models import AnalysisStatus, AnalysisTask
from ....core.exceptions import (
    AppError,
    BadGatewayError,
    InternalServiceError,
    InvalidInputError,
    NotFoundError,
    WorkspaceNotFoundError,
)
from ....core.services.quotation_client import QuotationServiceError
from ....core.worker_input_snapshots import create_worker_input_snapshot
from ....core.workspace import workspace_manager
from ....models.quotation import QuotationEngineConfig, QuotationRequest
from .quotation_core import DEFAULT_CONTEXT_LENGTH


async def submit_quotation_analysis(
    *,
    user_id: str,
    workspace_id: str,
    node_id: str,
    request: QuotationRequest,
    workspace: Any,
) -> dict[str, Any]:
    """Submit one quotation worker task and return the running response."""

    task_manager = get_task_manager(user_id)
    try:
        if node_id not in workspace.nodes:
            raise NotFoundError(f"Node {node_id} not found")

        engine = request.engine or QuotationEngineConfig()
        page = max(1, int(request.page)) if request.page else 1
        analysis_request = AnalysisQuotationRequest(
            node_id=node_id,
            column=request.column,
            engine=engine.model_dump(mode="json"),
            page=page,
            page_size=request.page_size,
            sort_by=request.sort_by or None,
            descending=request.descending,
            context_length=DEFAULT_CONTEXT_LENGTH,
        )

        workspace_dir = workspace_manager.get_workspace_dir(user_id, workspace_id)
        if workspace_dir is None:
            raise WorkspaceNotFoundError("Workspace not found")
        task_id = str(uuid4())
        input_snapshot_dir = create_worker_input_snapshot(
            workspace_id=workspace_id,
            task_id=task_id,
            node_ids=[node_id],
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

        worker_task_manager = workspace_manager.get_task_manager(user_id)
        await worker_task_manager.submit_task(
            user_id=user_id,
            workspace_id=workspace_id,
            task_type="quotation",
            task_id=task_id,
            task_args={
                "input_snapshot_dir": str(input_snapshot_dir),
                "node_id": node_id,
                "request_payload": analysis_request.model_dump(mode="json"),
            },
            task_name="Quotation",
        )

        return {
            "state": "running",
            "message": "Quotation analysis started",
            "data": None,
            "metadata": {"task_id": task_id},
        }
    except AppError:
        raise
    except QuotationServiceError as exc:
        raise BadGatewayError(str(exc)) from exc
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise InternalServiceError(f"Unexpected quotation error: {exc}") from exc
