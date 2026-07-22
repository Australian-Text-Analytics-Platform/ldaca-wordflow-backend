"""Strict canonical workspace, graph, and node resources."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .tokenization import TokenizationMetadata
from .names import NodeName
from ..domain.workspace import (
    AnalysisRecord,
    AnalysisState,
    NodeProvenance,
    Tab,
    analysis_input_ids,
)


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
    """Safe portable workspace metadata stored in archive manifest version 4."""

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


class WorkspaceArchiveAnalysisInput(_StrictModel):
    """One materialized immutable run input used to rebuild a query snapshot."""

    id: uuid.UUID
    name: NodeName
    document: str | None = None
    color: str | None = None
    tokenization: dict[
        Annotated[str, Field(min_length=1, max_length=500)],
        TokenizationMetadata,
    ] = Field(default_factory=dict, max_length=500)
    data_file: str = Field(min_length=1)


class WorkspaceArchiveAnalysis(_StrictModel):
    """One terminal Analysis plus safe materialized query inputs, when needed."""

    record: AnalysisRecord
    query_inputs: list[WorkspaceArchiveAnalysisInput] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_query_inputs(self) -> "WorkspaceArchiveAnalysis":
        expected_ids = list(analysis_input_ids(self.record.request))
        actual_ids = [item.id for item in self.query_inputs]
        if self.record.query_snapshot is None:
            if actual_ids:
                raise ValueError("Only a queryable Analysis has archived query inputs")
            return self
        if actual_ids != expected_ids:
            raise ValueError("Archived query inputs must match the Analysis request")
        analysis_id = str(self.record.id)
        for item in self.query_inputs:
            expected_file = f"analyses/{analysis_id}/query-data/{item.id}.parquet"
            if item.data_file != expected_file:
                raise ValueError("Archived query input path is invalid")
        return self


class WorkspaceArchiveManifest(_StrictModel):
    """Only accepted client workspace archive manifest."""

    format: Literal["wordflow-materialized-workspace"]
    version: Literal[4]
    workspace: WorkspaceArchiveMetadata
    nodes: list[WorkspaceArchiveNode]
    tabs: list[Tab]
    analyses: list[WorkspaceArchiveAnalysis]

    @model_validator(mode="after")
    def validate_analysis_ownership(self) -> "WorkspaceArchiveManifest":
        analysis_ids = [str(item.record.id) for item in self.analyses]
        if len(analysis_ids) != len(set(analysis_ids)):
            raise ValueError("Workspace archive has duplicate Analysis IDs")
        by_id = {str(item.record.id): item.record for item in self.analyses}
        if any(
            record.state
            not in {
                AnalysisState.SUCCEEDED,
                AnalysisState.FAILED,
                AnalysisState.CANCELLED,
            }
            for record in by_id.values()
        ):
            raise ValueError("Workspace archives contain only terminal Analyses")
        root_ids = {
            analysis_id
            for analysis_id, record in by_id.items()
            if record.parent_analysis_id is None
        }
        tab_analysis_ids = [
            str(tab.analysis_id) for tab in self.tabs if tab.analysis_id is not None
        ]
        if len(tab_analysis_ids) != len(set(tab_analysis_ids)):
            raise ValueError("A root Analysis may belong to only one archived Tab")
        if set(tab_analysis_ids) != root_ids:
            raise ValueError("Archived root Analyses must belong to exactly one Tab")
        for tab in self.tabs:
            if tab.analysis_id is None:
                continue
            record = by_id[str(tab.analysis_id)]
            if record.parent_analysis_id is not None or record.request.kind != tab.kind:
                raise ValueError("Archived Tab and Analysis ownership is invalid")
        for record in by_id.values():
            if record.parent_analysis_id is None:
                continue
            parent = by_id.get(str(record.parent_analysis_id))
            if parent is None or parent.parent_analysis_id is not None:
                raise ValueError("Archived child Analysis parent is invalid")
        return self


__all__ = [
    "WorkspaceCreateRequest",
    "WorkspaceArchiveAnalysis",
    "WorkspaceArchiveAnalysisInput",
    "WorkspaceArchiveManifest",
    "WorkspaceNodeInfo",
    "WorkspaceNodeReorderRequest",
    "WorkspaceResource",
    "WorkspaceUpdateRequest",
]
