"""Transport models for Workspace-owned Analysis collections."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..domain.workspace import Analysis, CorruptAnalysis


class AnalysisPage(BaseModel):
    """One-based stable page of live valid and corrupt Analyses."""

    model_config = ConfigDict(extra="forbid")

    items: list[Analysis | CorruptAnalysis]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)


__all__ = ["AnalysisPage"]
