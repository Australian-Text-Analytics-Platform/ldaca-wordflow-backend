"""Explicit Workspace close and referential deletion coordination."""

from __future__ import annotations

from .analysis_execution import AnalysisExecutionRuntime
from .workspace import WorkspaceRecord, WorkspaceService


class WorkspaceLifecycleService:
    """Coordinate explicit Workspace lifecycle with private Analysis execution."""

    def __init__(
        self,
        workspaces: WorkspaceService,
        analyses: AnalysisExecutionRuntime,
    ) -> None:
        self._workspaces = workspaces
        self._analyses = analyses

    async def request_close(
        self,
        user_id: str,
        workspace_id: str,
    ) -> WorkspaceRecord | None:
        """Return a closing resource, or ``None`` after immediate closure."""

        return await self._workspaces.request_close(
            user_id,
            workspace_id,
            self._analyses.has_workspace_work,
        )

    async def delete(
        self,
        user_id: str,
        workspace_id: str,
    ) -> None:
        """Stop Workspace-owned execution and atomically remove the Workspace."""

        async with self._workspaces.deletion_context(user_id, workspace_id):
            await self._analyses.cancel_workspace(user_id, workspace_id)


__all__ = ["WorkspaceLifecycleService"]
