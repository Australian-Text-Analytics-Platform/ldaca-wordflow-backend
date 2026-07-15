"""Transport requests for strict Workspace-owned Tab resources."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..domain.workspace import AnalysisKind, TabName


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TabCreate(_StrictModel):
    kind: AnalysisKind
    name: TabName


class TabRename(_StrictModel):
    name: TabName


__all__ = ["TabCreate", "TabRename"]
