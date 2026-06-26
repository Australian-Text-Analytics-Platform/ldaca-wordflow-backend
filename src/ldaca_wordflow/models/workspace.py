"""Workspace, node, and tokenizer models.

Split from models/__init__.py.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .files import FilesTaskMetadataResponse


class WorkspaceInfo(BaseModel):
    """Metadata schema used by API responses to describe workspace info.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    id: str
    name: str
    description: str = ""
    created_at: str | None = None
    modified_at: str | None = None
    total_nodes: int
    root_nodes: int = 0
    leaf_nodes: int = 0


class WorkspaceSummary(BaseModel):
    """Summary metadata for a workspace row in list responses.

    Used by:
    - backend API routes, backend request/response models, core workspace and worker
      services because they need a stable JSON contract shared by route handlers, generated
      clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    id: str
    name: str
    description: str = ""
    created_at: str = ""
    modified_at: str = ""
    total_nodes: int = 0
    root_nodes: int = 0
    leaf_nodes: int = 0
    workspace_size_Byte: int = 0
    folder_name: str | None = None


class CurrentWorkspaceResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for current workspace response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    id: str | None = None


class SetCurrentWorkspaceResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for set current workspace response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    state: Literal["successful"]
    id: str | None = None


class DtypeNormalizationChange(BaseModel):
    """API schema used by routes and generated clients for dtype normalization change.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column: str
    from_dtype: str
    to_dtype: str
    reason: str


class WorkspaceNodeInfo(BaseModel):
    """Metadata schema used by API responses to describe workspace node info.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str
    name: str
    operation: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)
    document: str | None = None
    color: str | None = None
    shape: tuple[int | None, int | None] = (None, None)
    column_schema: dict[str, str] = Field(default_factory=dict, alias="schema")
    columns: list[str] = Field(default_factory=list)
    can_undo: bool | None = None
    can_redo: bool | None = None
    dtype_normalization: list[DtypeNormalizationChange] | None = None
    tokenizer_models: dict[str, str] = Field(default_factory=dict)


class NodeDocumentColumnUpdateRequest(BaseModel):
    """Request schema used by API routes and generated clients for node document column update request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    document_column: str | None = None


class NodeColorUpdateRequest(BaseModel):
    """Request schema used by API routes and generated clients for node colour updates.

    Used by:
    - backend API routes and frontend node selectors because source-node visualisation
      colours are durable workspace-node metadata.

    Flow: validate one CSS hex colour string before the node route normalizes and
    persists it on the selected workspace node.
    """

    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")


class NodeTokenizationPreferenceRequest(BaseModel):
    """Request schema used by API routes and generated clients for node tokenization preference request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: resolve tokenization preferences, hydrate or create token columns, aggregate
        frequencies, and persist derived artifacts for result queries.
    """

    source_column: str
    model: str | None = None
    language: str | None = None


class TokenizerModelInfo(BaseModel):
    """Metadata schema used by API responses to describe tokenizer model info.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    model: str
    label: str
    languages: list[str]


class TokenizerModelsResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for tokenizer models response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    models: list[TokenizerModelInfo]


class WorkspaceGraphEdge(BaseModel):
    """API schema used by routes and generated clients for workspace graph edge.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    model_config = ConfigDict(extra="allow")

    source: str
    target: str
    label: str | None = None


class WorkspaceGraphResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for workspace graph response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    model_config = ConfigDict(extra="allow")

    nodes: list[WorkspaceNodeInfo]
    edges: list[WorkspaceGraphEdge]


class WorkspaceNodeReorderRequest(BaseModel):
    """Request body for persisting a new workspace node order.

    Used by:
    - backend ``reorder_workspace_nodes`` route and the generated client because the
      list-view drag-to-reorder gesture commits the full node id sequence.

    Flow: the route validates the active workspace, applies ``ordered_ids`` via
        ``Workspace.reorder_nodes``, persists, and returns the rebuilt graph.
    """

    ordered_ids: list[str]


class WorkspaceNodesResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for workspace nodes response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    nodes: list[WorkspaceNodeInfo]


class WorkspaceCreateRequest(BaseModel):
    """Request schema used by API routes and generated clients for workspace create request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    name: str
    description: str | None = None


class WorkspaceSaveRequest(BaseModel):
    """Request schema used by API routes and generated clients for workspace save request.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    workspace_id: str
    name: str | None = None
    description: str | None = None


class WorkspaceActionResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for workspace action response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    state: Literal["successful"]
    message: str
    id: str | None = None


class WorkspaceTaskStartResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for workspace task start response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    state: Literal["running"]
    message: str
    metadata: FilesTaskMetadataResponse


class WorkspaceUploadResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for workspace upload response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    state: Literal["successful"]
    workspace: WorkspaceSummary


# =============================================================================
# DATAFRAME MODELS
# =============================================================================
