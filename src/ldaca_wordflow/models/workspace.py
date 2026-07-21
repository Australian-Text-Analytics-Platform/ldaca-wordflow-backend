"""Strict canonical workspace, graph, and node resources."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .tokenization import TokenizationMetadata
from .names import NodeName
from ..domain.workspace import NodeProvenance, Tab


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DtypeNormalizationChange(_StrictModel):
    """One source-file dtype normalization applied during node creation."""

    column: str
    from_dtype: str
    to_dtype: str
    reason: str


class WorkspaceNodeInfo(_StrictModel):
    """Complete addressable node metadata returned by node routes."""

    id: uuid.UUID
    name: NodeName
    provenance: NodeProvenance
    derivation_description: str
    parent_ids: list[uuid.UUID] = Field(default_factory=list)
    child_ids: list[uuid.UUID] = Field(default_factory=list)
    document: str | None = None
    color: str | None = None
    shape: tuple[int | None, int | None] = (None, None)
    dtype_normalization: list[DtypeNormalizationChange] | None = None
    tokenizer_models: dict[str, str] = Field(default_factory=dict)
    can_undo: bool
    can_redo: bool


class WorkspaceResource(_StrictModel):
    """Lightweight Workspace metadata plus process-local runtime state."""

    id: uuid.UUID
    name: str
    description: str
    created_at: AwareDatetime
    modified_at: AwareDatetime
    total_nodes: int = Field(ge=0)
    root_nodes: int = Field(ge=0)
    leaf_nodes: int = Field(ge=0)
    revision: int = Field(ge=0)
    runtime_state: Literal["closed", "open", "closing"]


class WorkspaceNodeReorderRequest(_StrictModel):
    """Complete desired workspace node order."""

    ordered_ids: list[uuid.UUID]


class WorkspaceCreateRequest(_StrictModel):
    """Create one workspace resource."""

    name: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=10_000)


class WorkspaceUpdateRequest(_StrictModel):
    """Partial workspace metadata update."""

    name: str | None = Field(default=None, min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=10_000)

    @model_validator(mode="after")
    def validate_patch(self) -> "WorkspaceUpdateRequest":
        if not self.model_fields_set:
            raise ValueError("Workspace patch must contain at least one field")
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("Workspace name cannot be null")
        return self


class WorkspaceArchiveMetadata(_StrictModel):
    """Safe portable workspace metadata stored in archive manifest version 3."""

    id: uuid.UUID
    name: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=10_000)
    created_at: AwareDatetime | None = None
    modified_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> "WorkspaceArchiveMetadata":
        if (
            self.created_at is not None
            and self.modified_at is not None
            and self.modified_at < self.created_at
        ):
            raise ValueError("Workspace modified_at cannot precede created_at")
        return self


class WorkspaceArchiveNode(_StrictModel):
    """Declarative materialized node entry with no executable plan payload."""

    id: uuid.UUID
    name: NodeName
    provenance: NodeProvenance
    document: str | None = None
    color: str | None = None
    tokenization: dict[
        Annotated[str, Field(min_length=1, max_length=500)],
        TokenizationMetadata,
    ] = Field(default_factory=dict, max_length=500)
    data_file: str = Field(min_length=1)


class WorkspaceArchiveManifest(_StrictModel):
    """Only accepted client workspace archive manifest."""

    format: Literal["wordflow-materialized-workspace"]
    version: Literal[3]
    workspace: WorkspaceArchiveMetadata
    nodes: list[WorkspaceArchiveNode]
    tabs: list[Tab]


__all__ = [
    "WorkspaceCreateRequest",
    "WorkspaceArchiveManifest",
    "WorkspaceNodeInfo",
    "WorkspaceNodeReorderRequest",
    "WorkspaceResource",
    "WorkspaceUpdateRequest",
]
