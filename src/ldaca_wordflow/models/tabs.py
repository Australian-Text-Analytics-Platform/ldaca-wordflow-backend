"""Transport requests for strict Workspace-owned Tab resources."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, model_validator

from ..domain.workspace import AnalysisKind, TabName


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TabCreate(_StrictModel):
    kind: AnalysisKind
    name: TabName


class TabUpdate(_StrictModel):
    name: TabName | None = None
    annotation_correction_columns: dict[uuid.UUID, TabName] | None = None

    @model_validator(mode="after")
    def require_update(self) -> "TabUpdate":
        if not self.model_fields_set:
            raise ValueError("At least one Tab field must be provided")
        return self


__all__ = ["TabCreate", "TabUpdate"]
