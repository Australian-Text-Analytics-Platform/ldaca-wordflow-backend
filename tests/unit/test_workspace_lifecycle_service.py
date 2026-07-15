"""Workspace close and deletion coordination with private Analysis execution."""

from pathlib import Path
from typing import Any, cast

import anyio
import pytest

from ldaca_wordflow.infrastructure.storage.workspace_store import WorkspaceStore
from ldaca_wordflow.services.events import EventHub
from ldaca_wordflow.services.workspace import WorkspaceService
from ldaca_wordflow.services.workspace_lifecycle import WorkspaceLifecycleService
from ldaca_wordflow.settings import Settings
from ldaca_wordflow.shared.errors import WorkspaceNotFoundError

from ._storage import unlimited_storage_admission


class _Analyses:
    def __init__(self) -> None:
        self.active = False
        self.cancelled: list[tuple[str, str]] = []

    async def has_workspace_work(self, user_id: str, workspace_id: str) -> bool:
        del user_id, workspace_id
        return self.active

    async def cancel_workspace(self, user_id: str, workspace_id: str) -> None:
        self.cancelled.append((user_id, workspace_id))


def _workspace_service(tmp_path: Path) -> WorkspaceService:
    settings = Settings(data_root=tmp_path, multi_user=False)
    limiter = anyio.CapacityLimiter(2)
    return WorkspaceService(
        settings,
        store=WorkspaceStore(
            max_nodes=settings.max_workspace_nodes,
            max_snapshot_bytes=settings.max_workspace_snapshot_bytes,
        ),
        storage_admission=unlimited_storage_admission(tmp_path, limiter=limiter),
        events=EventHub(),
        io_limiter=limiter,
    )


async def test_close_defers_only_while_analysis_execution_is_active(
    tmp_path: Path,
) -> None:
    workspaces = _workspace_service(tmp_path)
    analyses = _Analyses()
    lifecycle = WorkspaceLifecycleService(workspaces, cast(Any, analyses))
    workspace = await workspaces.create_workspace("owner", "Close safely")
    await workspaces.open_workspace("owner", workspace.id)

    analyses.active = True
    closing = await lifecycle.request_close("owner", workspace.id)

    assert closing is not None
    assert closing.runtime_state == "closing"
    analyses.active = False
    await workspaces.finalize_close_if_idle(
        "owner",
        workspace.id,
        analyses.has_workspace_work,
    )
    assert (await workspaces.get_workspace("owner", workspace.id)).runtime_state == (
        "closed"
    )


async def test_delete_signals_execution_then_atomically_removes_workspace(
    tmp_path: Path,
) -> None:
    workspaces = _workspace_service(tmp_path)
    analyses = _Analyses()
    lifecycle = WorkspaceLifecycleService(workspaces, cast(Any, analyses))
    workspace = await workspaces.create_workspace("owner", "Delete safely")
    await workspaces.open_workspace("owner", workspace.id)

    await lifecycle.delete("owner", workspace.id)

    assert analyses.cancelled == [("owner", workspace.id)]
    with pytest.raises(WorkspaceNotFoundError):
        await workspaces.get_workspace("owner", workspace.id)
