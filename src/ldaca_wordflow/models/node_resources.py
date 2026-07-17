"""Canonical immutable node-creation, metadata, and row-query resources.

Used by the node router and ``NodeService``. Every data-changing operation is a
discriminated node creation request: source nodes come from user files and all
transformations create a derived child. This avoids hidden in-place plan
mutation and gives every successful operation one addressable resource.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain.workspace.provenance import (
    CastDerivation,
    CloneDerivation,
    ConcatDerivation,
    ExpressionDerivation,
    FilterDerivation,
    JoinDerivation,
    ReplaceDerivation,
    SliceDerivation,
)
from .names import NodeName


class _StrictRequest(BaseModel):
    """Reject misspelled or obsolete fields at the public API boundary."""

    model_config = ConfigDict(extra="forbid")


class FileNodeCreateRequest(_StrictRequest):
    """Create one source node from a safe user-file path."""

    kind: Literal["file"] = "file"
    file_path: str = Field(min_length=1)
    sheet_name: str | None = None
    name: NodeName | None = None


class CloneNodeCreateRequest(CloneDerivation):
    """Create an independent lazy-plan child from one source node."""

    source_node_id: uuid.UUID
    name: NodeName | None = None


class SliceNodeCreateRequest(SliceDerivation):
    """Create a slice, random sample, or shuffled child node."""

    source_node_id: uuid.UUID
    name: NodeName | None = None


class FilterNodeCreateRequest(FilterDerivation):
    """Create a child node whose rows satisfy typed filter predicates."""

    source_node_id: uuid.UUID
    name: NodeName | None = None


class ReplaceNodeCreateRequest(ReplaceDerivation):
    """Create a child with one regex-replaced or extracted text column."""

    source_node_id: uuid.UUID
    name: NodeName | None = None


class ExpressionNodeCreateRequest(ExpressionDerivation):
    """Create a child from a typed expression tree compiled by the server."""

    source_node_id: uuid.UUID
    name: NodeName | None = None


class ConcatNodeCreateRequest(ConcatDerivation):
    """Create a vertically concatenated child from schema-compatible nodes."""

    source_node_ids: list[uuid.UUID] = Field(min_length=2)
    name: NodeName | None = None


class JoinNodeCreateRequest(JoinDerivation):
    """Create a relational join child from two source nodes."""

    left_node_id: uuid.UUID
    right_node_id: uuid.UUID
    name: NodeName | None = None


class CastNodeCreateRequest(CastDerivation):
    """Create a child with one column cast to a supported logical type."""

    source_node_id: uuid.UUID
    name: NodeName | None = None


NodeCreateRequest = Annotated[
    FileNodeCreateRequest
    | CloneNodeCreateRequest
    | SliceNodeCreateRequest
    | FilterNodeCreateRequest
    | ReplaceNodeCreateRequest
    | ExpressionNodeCreateRequest
    | ConcatNodeCreateRequest
    | JoinNodeCreateRequest
    | CastNodeCreateRequest,
    Field(discriminator="kind"),
]

NodeDerivationRequest = Annotated[
    CloneNodeCreateRequest
    | SliceNodeCreateRequest
    | FilterNodeCreateRequest
    | ReplaceNodeCreateRequest
    | ExpressionNodeCreateRequest
    | ConcatNodeCreateRequest
    | JoinNodeCreateRequest
    | CastNodeCreateRequest,
    Field(discriminator="kind"),
]


class NodeUpdateRequest(_StrictRequest):
    """Partial public metadata update for one existing node."""

    name: NodeName | None = None
    document: str | None = None
    color: str | None = None

    @model_validator(mode="after")
    def validate_patch(self) -> "NodeUpdateRequest":
        if not self.model_fields_set:
            raise ValueError("Node patch must contain at least one field")
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("Node name cannot be null")
        return self
