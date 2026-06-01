"""Pydantic request/response models for backend API contracts.

Used by:
- API routers and worker result serialization boundaries because they need a backend
  boundary that validates inputs before delegating to workspace or worker state.
Why:
- Centralizes schema contracts shared between frontend and backend endpoints.

Refactor note:
- This module is large and mixes domains (auth/files/workspaces/analysis); future
    split by domain files could improve maintainability and import clarity.

Flow: validate incoming API fields, apply defaults or validators, and serialize route
    responses in the shape expected by frontend clients and tests.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator

from ..analysis.models import BaseAnalysisRequest
from .analysis_common import DetachNodeOption

# =============================================================================
# AUTHENTICATION MODELS
# =============================================================================


class User(BaseModel):
    """API schema used by routes and generated clients for user.

    Used by:
    - backend API routes, backend package imports, backend request/response models, backend
      tests because they need a stable JSON contract shared by route handlers, generated
      clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    id: str
    email: str
    name: str
    picture: Optional[str] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None
    created_at: Optional[str] = None
    last_login: Optional[str] = None


class AuthMethod(BaseModel):
    """API schema used by routes and generated clients for auth method.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    name: str  # "google", "github", etc. (changed from 'type' to match frontend)
    display_name: str
    enabled: bool


class AuthInfoResponse(BaseModel):
    """Main auth info response - tells frontend everything it needs to know

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    authenticated: bool
    user: Optional[User] = None
    available_auth_methods: List[AuthMethod] = []
    requires_authentication: bool
    data_folder: Optional[str] = None


class GoogleIn(BaseModel):
    """API schema used by routes and generated clients for google in.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    id_token: str


class GoogleOut(BaseModel):
    """API schema used by routes and generated clients for google out.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    access_token: str
    refresh_token: str
    expires_in: int
    scope: str
    token_type: str
    user: User  # Updated to use User instead of UserInfo


class UserResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for user response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    id: str  # UUID string, not integer
    email: str
    name: str
    picture: Optional[str] = None  # Made optional
    is_active: bool
    is_verified: bool
    created_at: str  # Will be converted from datetime
    last_login: str  # Will be converted from datetime


# =============================================================================
# USER MANAGEMENT MODELS
# =============================================================================


# =============================================================================
# FILE MANAGEMENT MODELS
# =============================================================================


class FileUploadResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for file upload response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    filename: str
    size: int
    upload_time: str
    file_type: str
    preview_available: bool


class ImportSampleDataResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for import sample data response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    status: str
    removed_existing: bool
    file_count: int
    bytes_copied: int
    message: str
    sample_dir: Optional[str] = None
    remote_download_started: bool = False


class ImportSampleDataRequest(BaseModel):
    """Request schema used by API routes and generated clients for import sample data request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    collection_ids: List[str] = Field(default_factory=list)


class SampleDataFileEntry(BaseModel):
    """API schema used by routes and generated clients for sample data file entry.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    path: str
    size: int
    sha256: str


class SampleDataCollection(BaseModel):
    """API schema used by routes and generated clients for sample data collection.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    id: str
    name: str
    description: str
    language: str
    bundled: bool
    total_size_bytes: int
    recommended_for: List[str]
    files: List[SampleDataFileEntry]
    status: Literal["bundled", "downloaded", "partial", "not_downloaded"]


class SampleDataCatalogueResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for sample data catalogue response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    schema_version: int
    collections: List[SampleDataCollection]


class DataFileInfo(BaseModel):
    """Metadata schema used by API responses to describe data file info.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    filename: str
    size: int
    created_at: str
    file_type: str


class LDaCAImportRequest(BaseModel):
    """Request schema used by API routes and generated clients for l da c a import request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    url: str
    filename: Optional[str] = None


class OniSearchRequest(BaseModel):
    """Request schema used by API routes and generated clients for oni search request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    method: Literal[
        "keyword",
        "identifier",
        "id",
        "string",
        "collection",
        "file_format",
        "all",
    ] = Field(default="keyword")
    query: str = Field(default="")
    limit: int = Field(default=25, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class OniSearchResult(BaseModel):
    """API schema used by routes and generated clients for oni search result.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    id: str
    crate_id: str | None = None
    title: str
    description: str | None = None
    types: list[str]
    license: str | None = None
    importable: bool
    access: dict[str, Any] | None = None
    collections: list[str]
    file_formats: list[str]
    stats: dict[str, Any]


class OniSearchResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for oni search response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    data: list[OniSearchResult]
    message: str


class FileTreeNodeResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for file tree node response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    name: str
    path: str
    type: Literal["file", "directory"]
    size: Optional[int] = None
    children: Optional[List["FileTreeNodeResponse"]] = None

    @model_validator(mode="after")
    def normalize_directory_children(self) -> "FileTreeNodeResponse":
        """Support API schema contracts with a normalize directory children helper.

        Called by:
        - `FileTreeNodeResponse` instances owned by backend services, routes, and tests because
          they need a backend boundary that validates inputs before delegating to workspace or
          worker state.

        Flow: validate incoming API fields, apply defaults or validators, and serialize route
            responses in the shape expected by frontend clients and tests.
        """

        if self.type == "directory" and self.children is None:
            self.children = []
        return self


FileTreeNodeResponse.model_rebuild()


class CreateFolderRequest(BaseModel):
    """Request schema used by API routes and generated clients for create folder request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    name: str
    parent_path: str = ""


class CreateFolderResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for create folder response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    message: str
    path: str


class MoveFileRequest(BaseModel):
    """Request schema used by API routes and generated clients for move file request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    source_path: str
    target_directory_path: str


class MessageResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for message response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    message: str


class RenameColumnRequest(BaseModel):
    """Request schema used by API routes and generated clients for rename column request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    new_name: str


class FilesTaskMetadataResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for files task metadata response.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    task_id: str


class FilesImportTaskStartResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for files import task start response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["running"]
    message: str
    metadata: FilesTaskMetadataResponse


class TaskListResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for task list response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    data: list[dict[str, Any]]
    message: str


class TaskClearActionDataResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for task clear action data response.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    cleared_worker: bool
    cleared_analysis: bool
    cleared_worker_ids: list[str]
    cleared_analysis_ids: list[str]
    cleared_task_ids: list[str]


class TaskClearActionResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for task clear action response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    data: TaskClearActionDataResponse
    message: str


class TaskCancelActionDataResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for task cancel action data response.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    stopped: bool


class TaskCancelActionResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for task cancel action response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    data: TaskCancelActionDataResponse
    message: str


class FilesTasksListResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for files tasks list response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: str
    data: List[Dict[str, Any]]
    message: str


class FilesTaskActionDataResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for files task action data response.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    cancelled: Optional[bool] = None
    cancelled_count: Optional[int] = None
    cleared_count: Optional[int] = None


class FilesTaskActionResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for files task action response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: str
    data: FilesTaskActionDataResponse
    message: str


class FileInfoResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for file info response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    filename: str
    size_Byte: int
    created_at: float
    modified_at: float
    file_type: str


# =============================================================================
# WORKSPACE MODELS
# =============================================================================


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
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
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
    folder_name: Optional[str] = None


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
    description: Optional[str] = None


class WorkspaceSaveRequest(BaseModel):
    """Request schema used by API routes and generated clients for workspace save request.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: resolve the user workspace directory, refresh cached path indexes, coordinate task
        cleanup, and return stable workspace metadata to callers.
    """

    workspace_id: str
    name: Optional[str] = None
    description: Optional[str] = None


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


class DataFrameNode(BaseModel):
    """API schema used by routes and generated clients for data frame node.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_id: str
    name: str
    parent_id: Optional[str] = None
    parent_ids: Optional[List[str]] = None  # Enhanced: support multiple parents
    child_ids: Optional[List[str]] = None  # Enhanced: support multiple children
    operation: str
    shape: tuple
    columns: List[str]
    created_at: str
    preview: List[Dict[str, Any]]
    document: Optional[str] = None  # Enhanced: active document column for text data
    column_schema: Optional[Dict[str, str]] = (
        None  # Enhanced: column schema information
    )


class DataFrameInfo(BaseModel):
    """Metadata schema used by API responses to describe data frame info.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_id: str
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    memory_usage: str
    is_text_data: bool  # Whether it's a text-oriented node
    document: Optional[str] = None  # Enhanced: document column for text data
    column_schema: Optional[Dict[str, str]] = (
        None  # Enhanced: column schema information
    )
    operation: Optional[str] = None  # Enhanced: operation that created this node
    parent_ids: Optional[List[str]] = None  # Enhanced: parent node IDs
    child_ids: Optional[List[str]] = None  # Enhanced: child node IDs


# =============================================================================
# DATA OPERATION MODELS
# =============================================================================


class DataOperation(BaseModel):
    """API schema used by routes and generated clients for data operation.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    operation_type: str  # 'filter', 'slice', 'transform', 'aggregate'
    parameters: Dict[str, Any]
    target_columns: Optional[List[str]] = None


class FilterOperation(BaseModel):
    """API schema used by routes and generated clients for filter operation.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column: str
    operator: str  # 'eq', 'gt', 'lt', 'contains', 'regex'
    value: Any


class SliceOperation(BaseModel):
    """API schema used by routes and generated clients for slice operation.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    start_row: Optional[int] = None
    end_row: Optional[int] = None
    columns: Optional[List[str]] = None


class TransformOperation(BaseModel):
    """API schema used by routes and generated clients for transform operation.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    operation: str  # 'rename', 'add_column', 'drop_column', 'convert_type'
    parameters: Dict[str, Any]


class AggregateOperation(BaseModel):
    """API schema used by routes and generated clients for aggregate operation.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    group_by: Optional[List[str]] = None
    aggregations: Dict[str, str]  # column -> function


class ReplaceRequest(BaseModel):
    """Request schema used by API routes and generated clients for replace request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    source_column: str = Field(..., min_length=1, max_length=200)
    pattern: str = Field(..., min_length=1)
    replacement: str = Field(default="")
    output_column_name: Optional[str] = Field(default=None, max_length=200)
    preview_limit: Optional[int] = Field(default=50, ge=1, le=500)
    mode: Literal["replace", "extract"] = Field(default="replace")
    count: Literal["all", "first"] = Field(default="all")
    n: Optional[int] = Field(default=None, ge=1)
    connector: str = Field(default=" ")


class ReplacePreviewResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for replace preview response.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    columns: List[str]
    dtypes: Dict[str, str]
    data: List[Dict[str, Any]]


class ReplaceApplyResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for replace apply response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    node_id: str
    column_name: str
    dtype: Optional[str] = None
    message: str


class JoinRequest(BaseModel):
    """Request schema used by API routes and generated clients for join request.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    right_node_id: str
    join_type: str  # 'inner', 'left', 'right', 'outer'
    left_on: List[str]
    right_on: List[str]
    suffix: str = "_right"


class ConcatPreviewRequest(BaseModel):
    """Request schema used by API routes and generated clients for concat preview request.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_ids: List[str] = Field(..., min_length=2)
    deduplicate: bool = True


class ConcatRequest(ConcatPreviewRequest):
    """Request schema used by API routes and generated clients for concat request.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    new_node_name: Optional[str] = None


class NodeOperationResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for node operation response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_name: str
    node_id: str


class NodeActionResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for node action response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    message: str


class CastNodeRequest(BaseModel):
    """Request schema used by API routes and generated clients for cast node request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column: str
    target_type: str
    format: str | None = None
    strict: bool | None = None


class CastNodeInfo(BaseModel):
    """Metadata schema used by API responses to describe cast node info.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column: str
    original_type: str
    new_type: str
    target_type: str
    format_used: str | None = None
    strict_used: bool | None = None


class CastNodeResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for cast node response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    node_id: str
    cast_info: CastNodeInfo
    message: str


class DataFrameOperationRequest(BaseModel):
    """Request schema used by API routes and generated clients for data frame operation request.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    workspace_id: str
    parent_node_id: str
    operation: DataOperation
    result_name: Optional[str] = None


class ConcordanceAnalysisRequest(BaseModel):
    """Request schema used by API routes and generated clients for concordance analysis request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_ids: List[str]  # Support up to 2 nodes (1 = single node mode)
    node_columns: Dict[str, str]  # node_id -> column_name mapping
    search_word: str
    num_left_tokens: int = 10
    num_right_tokens: int = 10
    regex: bool = False
    whole_word: bool = False
    case_sensitive: bool = False
    combined: bool = False  # if true, backend builds a combined view across nodes
    # "regex" (default) uses the polars-text concordance engine on raw text,
    # preserving partial-word patterns like ``equ\w*`` for English users.
    # "tokens" looks up a tokenization column and walks it for exact-token
    # matches with N-actual-token left/right context — the
    # word-aware semantics CJK users want once Tokenise has been run.
    # Falls back to regex behaviour if no tokenization column exists.
    search_mode: Literal["regex", "tokens"] = "regex"
    # Lets the frontend tell the backend what language to assume.
    # ``None`` defers to the active node's tokenization metadata then ``"en"``.
    language: Optional[str] = None
    # Sorting parameters
    sort_by: Optional[str] = None  # column name to sort by
    descending: bool = True

    model_config = ConfigDict(extra="forbid")


class ConcordanceDetachRequest(BaseModel):
    """Request schema used by API routes and generated clients for concordance detach request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_id: str
    column: str
    search_word: str
    num_left_tokens: int = 10
    num_right_tokens: int = 10
    regex: bool = False
    whole_word: bool = False
    case_sensitive: bool = False
    new_node_name: Optional[str] = None  # If not provided, will be auto-generated
    selected_columns: Optional[list[str]] = None
    materialized_path: Optional[str] = None  # Reuse existing flattened parquet


class ConcordanceDispersionDetachRequest(BaseModel):
    """Detach a per-document aggregation of concordance hits.

    Unlike `ConcordanceDetachRequest` (one row per hit), this produces one row
    per source document with the hits collected into `List<T>` columns and the
    raw match-window text rendered as a multi-line `CONC_extraction` string.

    `selected_bins` + `total_bins` optionally restrict the aggregation to hits
    whose `start_idx / doc_length` falls inside one of the selected bins (the
    chart's "in-range hits only" semantic).

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column: str
    search_word: str
    num_left_tokens: int = 10
    num_right_tokens: int = 10
    regex: bool = False
    whole_word: bool = False
    case_sensitive: bool = False
    new_node_name: Optional[str] = None
    selected_columns: Optional[list[str]] = None
    materialized_path: Optional[str] = None
    # When the slow path runs (no materialized_path), the worker also writes
    # the materialised flat parquet so the user doesn't have to click "Process
    # All" separately before iterating on bin selections. The parent analysis
    # task id + this node's id are needed to publish the standard
    # `analysis_materialized` event back to the frontend.
    parent_task_id: Optional[str] = None
    selected_bins: Optional[list[int]] = None
    total_bins: Optional[int] = None
    # When the chart legend is filtered, the detach should aggregate only over
    # hits whose `CONC_matched_text` lands in this set. `None` means "all".
    # `match_case_insensitive` mirrors the chart's `lowercaseMatches` toggle:
    # when true, both the column and the candidate set are lowercased before
    # comparison so the filter agrees with the legend grouping.
    selected_matched_texts: Optional[list[str]] = None
    match_case_insensitive: bool = False


class ConcordanceMaterializeRequest(BaseModel):
    """Request schema used by API routes and generated clients for concordance materialize request.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column: str
    search_word: str
    num_left_tokens: int = 10
    num_right_tokens: int = 10
    regex: bool = False
    whole_word: bool = False
    case_sensitive: bool = False
    # Mirror the live ``/concordance`` request — materialize must honour the
    # engine the user actually searched with. Defaults to ``"regex"`` so
    # existing English flows are byte-identical.
    search_mode: Literal["regex", "tokens"] = "regex"
    language: Optional[str] = None
    parent_task_id: str


ConcordanceDetachNodeOption = DetachNodeOption  # shared base, kept for backwards compat


class ConcordanceDetachOptionsResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for concordance detach options
    response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    message: str
    data: Dict[str, List[ConcordanceDetachNodeOption]] | None = None
    metadata: AnalysisTaskMetadata | None = None


# Quotation requests (mirror concordance shape but without search parameters)
class QuotationEngineType(str, Enum):
    """Enum used by API schema contracts to constrain quotation engine type values.

    Used by:
    - backend API routes, backend request/response models, backend tests, core workspace and
      worker services because they need a stable JSON contract shared by route handlers,
      generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    LOCAL = "local"
    REMOTE = "remote"


class QuotationEngineConfig(BaseModel):
    """API schema used by routes and generated clients for quotation engine config.

    Used by:
    - analysis task helpers, backend API routes, backend request/response models, backend
      tests, core workspace and worker services because they need a stable JSON contract
      shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    type: QuotationEngineType = QuotationEngineType.LOCAL
    url: Optional[AnyHttpUrl] = None

    @model_validator(mode="after")
    def _validate_remote(self) -> "QuotationEngineConfig":
        """Validate remote inputs before API schema contracts proceeds.

        Called by:
        - `QuotationEngineConfig` instances owned by backend services, routes, and tests because
          they need a backend boundary that validates inputs before delegating to workspace or
          worker state.

        Flow: validate incoming API fields, apply defaults or validators, and serialize route
            responses in the shape expected by frontend clients and tests.
        """

        if self.type is QuotationEngineType.LOCAL:
            # Normalise to ensure we never persist stale URLs for local mode
            self.url = None
        elif self.url is None:
            raise ValueError("Remote quotation engines require a URL")
        return self

    model_config = ConfigDict(extra="forbid")


class QuotationRequest(BaseModel):
    """Request schema used by API routes and generated clients for quotation request.

    Used by:
    - analysis task helpers, backend API routes, backend request/response models, backend
      tests because they need a stable JSON contract shared by route handlers, generated
      clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column: str
    # Pagination parameters
    page: int = 1
    page_size: Optional[int] = None
    # Sorting parameters
    sort_by: Optional[str] = None  # column name to sort by
    descending: bool = True
    engine: Optional[QuotationEngineConfig] = None
    # Quotation is English-only. The route resolves an effective language and
    # rejects non-EN with a typed UnsupportedLanguageError so users see a clear
    # "English-only" message rather than garbage output.
    # ``None`` falls back to the node's tokenization metadata (if it's been
    # tokenised) and then ``"en"``.
    language: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class QuotationDetachRequest(BaseModel):
    """Request schema used by API routes and generated clients for quotation detach request.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_id: str
    column: str
    new_node_name: Optional[str] = None  # If not provided, will be auto-generated
    engine: Optional[QuotationEngineConfig] = None
    selected_columns: Optional[list[str]] = None
    materialized_path: Optional[str] = None  # Reuse existing flattened parquet
    language: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class QuotationMaterializeRequest(BaseModel):
    """Request schema used by API routes and generated clients for quotation materialize request.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column: str
    engine: Optional[QuotationEngineConfig] = None
    parent_task_id: str
    language: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


QuotationDetachNodeOption = DetachNodeOption  # shared base, kept for backwards compat


class QuotationDetachOptionsResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for quotation detach options response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    message: str
    data: Dict[str, List[QuotationDetachNodeOption]] | None = None
    metadata: AnalysisTaskMetadata | None = None


class QuotationMetadata(BaseModel):
    """API schema used by routes and generated clients for quotation metadata.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    quotation_columns: list[str]
    metadata_columns: list[str]
    all_columns: list[str]


class QuotationAnalysisResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for quotation analysis response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    data: list[list[dict[str, Any]]]
    columns: list[str]
    metadata: QuotationMetadata
    pagination: SourceRowPagination
    sorting: AnalysisSorting
    preferences: dict[str, Any] | None = None
    task_id: str | None = None


class QuotationPreferenceUpdateData(BaseModel):
    """Data payload schema embedded in API responses for quotation preference update data.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    context_length: int | None = None


class QuotationPreferenceUpdateResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for quotation preference update
    response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    message: str
    data: QuotationPreferenceUpdateData | None = None


class AnalysisTaskActionResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for analysis task action response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    message: str
    data: None = None
    metadata: AnalysisTaskMetadata | None = None


class AnalysisClearResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for analysis clear response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    message: str


class CurrentAnalysisTasksResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for current analysis tasks response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    task_ids: list[str]


class QuotationResultQuery(BaseModel):
    """API schema used by routes and generated clients for quotation result query.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    page: Optional[int] = None
    page_size: Optional[int] = None
    sort_by: Optional[str] = None
    descending: Optional[bool] = None
    context_length: Optional[int] = None
    update_only: bool = False

    model_config = ConfigDict(extra="forbid")


class SequentialAnalysisRequest(BaseModel):
    """Request schema used by API routes and generated clients for sequential analysis request.

    Used by:
    - analysis task helpers, backend API routes, backend request/response models because
      they need a stable JSON contract shared by route handlers, generated clients, and
      tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    time_column: str
    group_by_columns: Optional[List[str]] = None
    # ``second`` and ``minute`` remain valid API frequencies even though
    # the live preset dropdown keeps the common, coarser choices front-and-center.
    frequency: Literal[
        "second",
        "minute",
        "hourly",
        "daily",
        "weekly",
        "monthly",
        "quarterly",
        "yearly",
        "custom",
    ] = "monthly"
    sort_by_time: bool = True
    column_type: Literal["datetime", "numeric"] = "datetime"
    numeric_origin: Optional[float] = None
    numeric_interval: Optional[float] = None
    custom_interval_value: Optional[int] = None
    custom_interval_unit: Optional[
        Literal["seconds", "minutes", "hours", "days", "weeks"]
    ] = None
    case_sensitive: bool = True

    @model_validator(mode="after")
    def validate_numeric_params(self) -> "SequentialAnalysisRequest":
        """Validate numeric params inputs before API schema contracts proceeds.

        Called by:
        - `SequentialAnalysisRequest` instances owned by backend services, routes, and tests
          because they need a backend boundary that validates inputs before delegating to
          workspace or worker state.

        Flow: validate incoming API fields, apply defaults or validators, and serialize route
            responses in the shape expected by frontend clients and tests.
        """

        if self.column_type == "numeric":
            if self.numeric_interval is None or self.numeric_interval <= 0:
                raise ValueError(
                    "numeric_interval must be a positive number when column_type='numeric'"
                )
        if self.column_type == "datetime" and self.frequency == "custom":
            if self.custom_interval_value is None or self.custom_interval_value <= 0:
                raise ValueError(
                    "custom_interval_value must be a positive integer when frequency='custom'"
                )
            if self.custom_interval_unit is None:
                raise ValueError(
                    "custom_interval_unit is required when frequency='custom'"
                )
        return self

    # Pydantic v2 model config
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "time_column": "created_at",
                "group_by_columns": ["party", "electorate"],
                "frequency": "monthly",
                "sort_by_time": True,
                "column_type": "datetime",
                "numeric_origin": None,
                "numeric_interval": None,
                "custom_interval_value": None,
                "custom_interval_unit": None,
            }
        }
    )


class SequentialAnalysisResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for sequential analysis response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    data: list[dict[str, Any]] | None = None
    columns: list[str] | None = None
    total_records: int | None = None
    chart_type: Literal["line", "bar", "area"] | None = None
    metadata: AnalysisTaskMetadata | None = None


class SequentialAnalysisPreviewResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for sequential analysis preview
    response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    total_records: int
    columns: list[str]
    data: list[dict[str, Any]] | None = None
    analysis_params: dict[str, Any] | None = None


class SequentialAnalysisPreferenceUpdateData(BaseModel):
    """Data payload schema embedded in API responses for sequential analysis preference update data.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    chart_type: Literal["line", "bar", "area"]


class SequentialAnalysisPreferenceUpdateRequest(BaseModel):
    """Request schema used by API routes and generated clients for sequential analysis preference update request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    chart_type: str | None = None


class SequentialAnalysisPreferenceUpdateResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for sequential analysis preference
    update response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    message: str
    data: SequentialAnalysisPreferenceUpdateData


class SequentialAnalysisDetachResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for sequential analysis detach
    response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    new_node_id: str
    new_node_name: str


class TextAnalysisInfo(BaseModel):
    """Metadata schema used by API responses to describe text analysis info.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    document: Optional[str]
    avg_document_length: Optional[float]
    total_documents: int
    vocabulary_size: Optional[int]
    is_text_ready: bool


class AiAnnotationDetachData(BaseModel):
    """Data payload schema embedded in API responses for ai annotation detach data.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    new_node_name: str
    record_count: int


class AiAnnotationDetachResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for ai annotation detach response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    message: str
    data: AiAnnotationDetachData


class AiAnnotationSaveData(BaseModel):
    """Data payload schema embedded in API responses for ai annotation save data.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    annotation_column: str
    edits_applied: int


class AiAnnotationSaveResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for ai annotation save response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    message: str
    data: AiAnnotationSaveData


class TopicModelingEmbeddingCacheMeasurement(BaseModel):
    """API schema used by routes and generated clients for topic modeling embedding cache measurement.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    bytes: int
    files: int


class TopicModelingEmbeddingCacheSizeResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for topic modeling embedding cache size
    response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    data: TopicModelingEmbeddingCacheMeasurement


class TopicModelingEmbeddingCacheClearData(BaseModel):
    """Data payload schema embedded in API responses for topic modeling embedding cache clear data.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    bytes_freed: int
    files_removed: int
    measured_before: TopicModelingEmbeddingCacheMeasurement


class TopicModelingEmbeddingCacheClearResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for topic modeling embedding cache
    clear response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful"]
    message: str
    data: TopicModelingEmbeddingCacheClearData


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class APIResponse(BaseModel):
    """Generic API response wrapper

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseModel):
    """Generic paginated response

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    data: List[Dict[str, Any]]
    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_more: bool


# =============================================================================
# FILE PREVIEW MODELS (Unified endpoint)
# =============================================================================


class FilePreviewRequest(BaseModel):
    """Request schema used by API routes and generated clients for file preview request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    filename: str
    page: int = 0
    page_size: int = 20
    payload: Optional[Dict[str, Any]] = None  # e.g., {"sheet_name": "Sheet1"}


class FilePreviewResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for file preview response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    filename: str
    file_type: str
    supported_types: List[str]  # ["LazyFrame", "DataFrame"]
    columns: List[str]
    preview: List[Dict[str, Any]]
    total_rows: int
    sheet_names: Optional[List[str]] = None
    selected_sheet: Optional[str] = None


class FilterCondition(BaseModel):
    """API schema used by routes and generated clients for filter condition.

    Used by:
    - backend request/response models, backend tests because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column: str
    operator: str  # Allow any string to support new operators like 'between'
    value: Any
    id: Optional[str] = None  # Frontend includes this for tracking
    dataType: Optional[str] = None  # Frontend includes this for UI
    # New flags from frontend Filter UI
    negate: Optional[bool] = False
    regex: Optional[bool] = False
    case_sensitive: Optional[bool] = False


class FilterRequest(BaseModel):
    """Request schema used by API routes and generated clients for filter request.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    conditions: List[FilterCondition]
    logic: Optional[str] = "and"
    new_node_name: Optional[str] = None


class SliceRequest(BaseModel):
    """Request schema used by API routes and generated clients for slice request.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    mode: Literal["slice", "random_sample", "shuffle"] = "slice"
    offset: int = Field(default=0, ge=0)
    length: Optional[int] = Field(default=None, ge=0)
    sample_size: Optional[float] = Field(default=None, gt=0)
    random_seed: Optional[int] = Field(default=None, ge=0)
    new_node_name: Optional[str] = None

    @model_validator(mode="after")
    def validate_sampling_mode(self) -> "SliceRequest":
        """Validate sampling mode inputs before API schema contracts proceeds.

        Called by:
        - `SliceRequest` instances owned by backend services, routes, and tests because they
          need a backend boundary that validates inputs before delegating to workspace or worker
          state.

        Flow: validate incoming API fields, apply defaults or validators, and serialize route
            responses in the shape expected by frontend clients and tests.
        """

        if self.mode == "random_sample":
            if self.sample_size is None:
                raise ValueError("sample_size is required when mode is 'random_sample'")
            if self.sample_size >= 1 and self.sample_size != int(self.sample_size):
                raise ValueError(
                    "sample_size >= 1 must be an integer (absolute row count)"
                )
        return self


class PaginationInfo(BaseModel):
    """Metadata schema used by API responses to describe pagination info.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    page: int
    page_size: int
    total_rows: int
    total_pages: int
    has_next: bool
    has_prev: bool


class FilterPreviewResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for filter preview response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    data: List[Dict[str, Any]]
    columns: List[str]
    dtypes: Dict[str, str]
    pagination: PaginationInfo


ColumnScalarValue = str | int | float | bool
AnalysisTaskState = Literal["pending", "running", "successful", "failed", "cancelled"]


class AnalysisTaskMetadata(BaseModel):
    """API schema used by routes and generated clients for analysis task metadata.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    model_config = ConfigDict(extra="allow")

    task_id: str | None = None


class SourceRowPagination(BaseModel):
    """API schema used by routes and generated clients for source row pagination.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    page: int
    page_size: int
    total_source_rows: int
    total_source_pages: int
    result_count: int
    has_next: bool
    has_prev: bool


class AnalysisSorting(BaseModel):
    """API schema used by routes and generated clients for analysis sorting.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    sort_by: str | None = None
    descending: bool


NodeDataSorting = AnalysisSorting  # structurally identical, kept for backwards compat


class NodeDataFiltering(BaseModel):
    """API schema used by routes and generated clients for node data filtering.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column: str | None = None
    value: str | None = None
    op: str


class NodeDataResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for node data response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    data: list[dict[str, Any]]
    pagination: PaginationInfo
    columns: list[str]
    dtypes: dict[str, str]
    sorting: NodeDataSorting
    filtering: NodeDataFiltering


class NodeQueryPlanResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for node query plan response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    plan: str


class NodeShapeResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for node shape response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    shape: tuple[int | None, int | None]


class ColumnUniqueValuesResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for column unique values response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column_name: str
    unique_count: int
    unique_values: list[ColumnScalarValue]
    has_null: bool


class ColumnOperationInfo(BaseModel):
    """Metadata schema used by API responses to describe column operation info.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    method: str
    label: str


class ColumnOperationsResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for column operations response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    operations: dict[str, list[ColumnOperationInfo]]


# =============================================================================
# POLARS EXPRESSION MODELS
# =============================================================================


class PolarsExpressionContext(str, Enum):
    """Enum used by API schema contracts to constrain polars expression context values.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    filter = "filter"
    with_columns = "with_columns"
    select = "select"
    sort = "sort"
    group_by_agg = "group_by_agg"


class PolarsExpressionItem(BaseModel):
    """A single polars expression supplied as a Python code string, e.g. ``pl.col('x') > 0``.

    Used by:
    - backend request/response models, backend tests because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    code: str  # Python expression string evaluated with pl available
    descending: Optional[bool] = None  # used only in sort context


class PolarsExpressionRequest(BaseModel):
    """Request schema used by API routes and generated clients for polars expression request.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    context: PolarsExpressionContext
    expressions: List[PolarsExpressionItem]
    # For group_by_agg: these are the grouping key expressions
    group_by_keys: Optional[List[PolarsExpressionItem]] = None
    new_node_name: Optional[str] = None


class PolarsExpressionApplyResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for polars expression apply response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_id: str
    node_name: str


# =============================================================================
# TOKEN FREQUENCY MODELS
# =============================================================================


class StopWordsPayload(BaseModel):
    """API schema used by routes and generated clients for stop words payload.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    stop_words: List[str]


class TokenFrequencyRequest(BaseModel):
    """Request schema used by API routes and generated clients for token frequency request.

    Used by:
    - analysis task helpers, backend API routes, backend request/response models, backend
      tests because they need a stable JSON contract shared by route handlers, generated
      clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_ids: List[str]  # 1 or 2 node IDs
    node_columns: Dict[str, str]  # Maps node_id -> column_name
    stop_words: Optional[List[str]] = None
    token_limit: Optional[int] = None
    tokenizer_model: Optional[str] = None
    node_tokenizer_models: Optional[Dict[str, str]] = None
    # Pydantic v2 model config
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "node_ids": ["node1", "node2"],
                "node_columns": {"node1": "text_column", "node2": "content_column"},
                "stop_words": ["the", "and", "or"],
                "token_limit": 50,
                "node_tokenizer_models": {
                    "node1": "native:plain_words_en",
                    "node2": "huggingface:bert-base-uncased",
                },
            }
        },
    )


class TokenFrequencyPreferenceUpdateRequest(BaseModel):
    """Request schema used by API routes and generated clients for token frequency preference update request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    token_limit: int | None = None
    stop_words: list[str] | None = None


class TokenFrequencyData(BaseModel):
    """Data payload schema embedded in API responses for token frequency data.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    token: str
    frequency: int


class TokenStatisticsData(BaseModel):
    """Token-level comparative statistics.

    Numeric statistics use a JSON-safe union to preserve semantic distinctions:
    - finite number -> float
    - positive infinity -> "+Inf"
    - negative infinity -> "-Inf"
    - missing/undefined -> None

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    token: str
    freq_reference: int  # OR - observed frequency in reference corpus
    freq_study: int  # OS - observed frequency in study corpus
    expected_reference: float | str | None  # Expected frequency in reference corpus
    expected_study: float | str | None  # Expected frequency in study corpus
    reference_total: int  # Total tokens in reference corpus
    study_total: int  # Total tokens in study corpus
    percent_reference: float | str | None  # %R - percentage in reference corpus
    percent_study: float | str | None  # %S - percentage in study corpus
    percent_diff: float | str | None  # %DIFF - percentage difference
    log_likelihood_llv: float | str | None  # LL - log likelihood G2 statistic
    bayes_factor_bic: float | str | None  # Bayes - Bayes factor (BIC)
    effect_size_ell: float | str | None  # ELL - effect size for log likelihood
    relative_risk: float | str | None = None  # RRisk - relative risk ratio
    log_ratio: float | str | None = None  # LogRatio - log of relative frequencies
    odds_ratio: float | str | None = None  # OddsRatio - odds ratio
    significance: str  # Significance level indicator


class TokenFrequencyNodeResult(BaseModel):
    """API schema used by routes and generated clients for token frequency node result.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    data: List[TokenFrequencyData]
    columns: List[str] = ["token", "frequency"]
    # Optional metadata (e.g., server-side truncation info)
    metadata: AnalysisTaskMetadata | None = None


class TokenFrequencyResponse(BaseModel):
    """Unified response model for token frequency analysis.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState | None = None
    message: str | None = None
    data: Optional[Dict[str, TokenFrequencyNodeResult]] = (
        None  # Maps node_name -> { data: [...], columns: [...] }
    )
    statistics: Optional[List[TokenStatisticsData]] = (
        None  # Statistical measures (only when comparing 2 nodes)
    )
    token_limit: Optional[int] = None
    analysis_params: Optional[Dict[str, Any]] = None
    metadata: AnalysisTaskMetadata | None = None
    stop_words: Optional[List[str]] = None


# =============================================================================
# AI ANNOTATION MODELS
# =============================================================================


class AiAnnotationClassDef(BaseModel):
    """API schema used by routes and generated clients for ai annotation class def.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    name: str
    description: str


class AiAnnotationExample(BaseModel):
    """API schema used by routes and generated clients for ai annotation example.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    query: str
    classification: str


class AiAnnotationModelsRequest(BaseModel):
    """Request schema used by API routes and generated clients for ai annotation models request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    base_url: Optional[str] = None
    api_key: Optional[str] = None


class AiAnnotationRequest(BaseAnalysisRequest):
    """Request schema used by API routes and generated clients for ai annotation request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_ids: List[str]
    node_columns: Dict[str, str]
    annotation_column: Optional[str] = None

    classes: List[AiAnnotationClassDef] = Field(min_length=1)
    examples: List[AiAnnotationExample] = Field(default_factory=list)

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1.0)
    seed: Optional[int] = 42
    batch_size: int = Field(default=100, ge=1)
    # When set, the classification system prompt gains a line like
    # "Texts are in Chinese." so the LLM doesn't mistake CJK for noise.
    # ``None`` falls back to ``effective_language(None, node)`` per node,
    # which keeps existing English flows unchanged (default = "en").
    language: Optional[str] = None

    page: int = 1
    page_size: int = 20
    sort_by: Optional[str] = None
    descending: bool = True

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "node_ids": ["node1"],
                "node_columns": {"node1": "document"},
                "classes": [
                    {"name": "support", "description": "Supportive tone"},
                    {"name": "critical", "description": "Critical tone"},
                ],
                "examples": [
                    {
                        "query": "This policy is fantastic and fair.",
                        "classification": "support",
                    }
                ],
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "top_p": 0.9,
            }
        }
    )


class AiAnnotationDetachRequest(BaseModel):
    """Request schema used by API routes and generated clients for ai annotation detach request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column: str
    new_node_name: Optional[str] = None
    annotation_column: Optional[str] = None

    classes: List[AiAnnotationClassDef] = Field(min_length=1)
    examples: List[AiAnnotationExample] = Field(default_factory=list)

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1.0)
    seed: Optional[int] = 42
    batch_size: int = Field(default=100, ge=1)
    # Optional language hint surfaced to the LLM prompt; falls back to the
    # node's tokenization metadata then to ``"en"``.
    language: Optional[str] = None


class AiAnnotationEdit(BaseModel):
    """API schema used by routes and generated clients for ai annotation edit.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    row_index: int = Field(ge=0)
    provider: str = Field(min_length=1)
    annotation: str = ""


class AiAnnotationSaveRequest(BaseModel):
    """Request schema used by API routes and generated clients for ai annotation save request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    annotation_column: Optional[str] = None
    edits: List[AiAnnotationEdit] = Field(default_factory=list)


class AiAnnotationNodeResult(BaseModel):
    """API schema used by routes and generated clients for ai annotation node result.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    data: List[Dict[str, Any]]
    columns: List[str]
    metadata: AnalysisTaskMetadata | None = None
    pagination: SourceRowPagination | None = None
    sorting: AnalysisSorting | None = None


class AiAnnotationResultQuery(BaseModel):
    """API schema used by routes and generated clients for ai annotation result query.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    page: Optional[int] = None
    page_size: Optional[int] = None
    sort_by: Optional[str] = None
    descending: Optional[bool] = None

    model_config = ConfigDict(extra="forbid")


class AiAnnotationResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for ai annotation response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    message: str
    data: Optional[Dict[str, AiAnnotationNodeResult]] = None
    analysis_params: Optional[Dict[str, Any]] = None
    combinable: Optional[bool] = None
    metadata: AnalysisTaskMetadata | None = None


class AiAnnotationModelInfo(BaseModel):
    """Metadata schema used by API responses to describe ai annotation model info.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    id: str
    name: str


class AiAnnotationModelsData(BaseModel):
    """Data payload schema embedded in API responses for ai annotation models data.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    models: list[AiAnnotationModelInfo]


class AiAnnotationModelsResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for ai annotation models response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful", "failed"]
    message: str
    data: AiAnnotationModelsData
    metadata: AnalysisTaskMetadata | None = None


class AiAnnotationProvidersData(BaseModel):
    """Data payload schema embedded in API responses for ai annotation providers data.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    providers: list[str]


class AiAnnotationProvidersResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for ai annotation providers response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful", "failed"]
    message: str
    data: AiAnnotationProvidersData
    metadata: AnalysisTaskMetadata | None = None


class AiAnnotationCategoriesData(BaseModel):
    """Data payload schema embedded in API responses for ai annotation categories data.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    categories: list[str]


class AiAnnotationCategoriesResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for ai annotation categories response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: Literal["successful", "failed"]
    message: str
    data: AiAnnotationCategoriesData
    metadata: AnalysisTaskMetadata | None = None


# =============================================================================
# TOPIC MODELING MODELS
# =============================================================================


class TopicModelingRequest(BaseModel):
    """Request schema used by API routes and generated clients for topic modeling request.

    Used by:
    - analysis task helpers, backend API routes, backend request/response models, backend
      tests because they need a stable JSON contract shared by route handlers, generated
      clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_ids: List[str]  # 1 or 2 node IDs
    node_columns: Dict[str, str]  # Maps node_id -> column_name
    min_topic_size: Optional[int] = (
        10  # kept for backwards compat; ignored when topic_size_mode != "min"
    )
    random_seed: Optional[int] = 42
    representative_words_count: Optional[int] = 5
    # Sampling: one entry per corpus in node_ids order. None = no sampling for that corpus.
    sample_fractions: Optional[List[Optional[float]]] = None
    # Topic size mode: controls how min_topic_size is derived
    topic_size_mode: Optional[Literal["target", "min", "exact"]] = "target"
    topic_size_value: Optional[int] = 25
    # Controls the per-topic LABEL stage's CountVectorizer stopword filter (not
    # the clustering stage). Default ``None`` falls back to
    # ``effective_language(...)`` per node. English uses sklearn's "english"
    # list; other languages get ``None`` so Chinese function words aren't
    # English-filtered (and so don't dominate every topic label).
    language: Optional[str] = None

    # Pydantic v2 model config
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "node_ids": ["node1", "node2"],
                "node_columns": {"node1": "text", "node2": "content"},
                "random_seed": 42,
                "representative_words_count": 5,
                "sample_fractions": [0.2, 0.5],
                "topic_size_mode": "target",
                "topic_size_value": 25,
            }
        }
    )


class TopicModelingTopic(BaseModel):
    """API schema used by routes and generated clients for topic modeling topic.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    id: int
    label: str
    representative_words: List[str] = Field(default_factory=list)
    size: List[int]  # per-corpus sizes aligned to request.node_ids order
    total_size: int
    x: float
    y: float


class TopicModelingData(BaseModel):
    """Data payload schema embedded in API responses for topic modeling data.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    topics: List[TopicModelingTopic]
    corpus_sizes: List[int]
    per_corpus_topic_counts: Optional[List[Dict[int, int]]] = None
    meta: AnalysisTaskMetadata | None = None


class TopicModelingResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for topic modeling response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    message: str
    data: Optional[TopicModelingData] = None
    metadata: AnalysisTaskMetadata | None = None


class TopicModelingResultUpdateRequest(BaseModel):
    """Request schema used by API routes and generated clients for topic modeling result update request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    topic_size_value: int


class TopicMeaningOverrideItem(BaseModel):
    """One topic's representative-words override for detach.

    Lets the frontend ship exactly what the user sees — post-fit
    "Words per topic" slice, post-fit stopword filter — instead of
    forcing the meanings parquet (written at fit time) into the
    detached node.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    topic_id: int
    words: List[str]


class TopicModelingDetachRequest(BaseModel):
    """Request payload for detaching topic assignments from cached topic-modeling output.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_ids: Optional[List[str]] = None
    selected_columns: Dict[str, List[str]] = Field(default_factory=dict)
    new_node_names: Optional[Dict[str, str]] = None
    topic_column_name: Optional[str] = "TOPIC_topic"
    topic_ids: Optional[List[int]] = None
    topic_meanings_override: Optional[List[TopicMeaningOverrideItem]] = None


TopicModelingDetachNodeOption = DetachNodeOption  # shared base, kept for backwards compat


class TopicModelingDetachOptionsResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for topic modeling detach options
    response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    message: str
    data: Dict[str, List[TopicModelingDetachNodeOption]] | None = None
    metadata: AnalysisTaskMetadata | None = None


class TopicModelingDetachedNode(BaseModel):
    """API schema used by routes and generated clients for topic modeling detached node.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    source_node_id: str
    new_node_id: str
    topic_meanings_node_id: str | None = None


class TopicModelingDetachData(BaseModel):
    """Data payload schema embedded in API responses for topic modeling detach data.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    detached_nodes: list[TopicModelingDetachedNode] = Field(default_factory=list)


class TopicModelingDetachResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for topic modeling detach response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    message: str
    data: TopicModelingDetachData | None = None
    metadata: AnalysisTaskMetadata | None = None


# Concordance response models
class ConcordanceMetadata(BaseModel):
    """Metadata about concordance columns to help frontend display logic

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    concordance_columns: List[
        str
    ]  # Core concordance columns (CONC_left_context, CONC_matched_text, CONC_right_context, etc.)
    metadata_columns: List[str]  # Original document metadata columns
    all_columns: List[str]  # All available columns


class ConcordanceNodeResult(BaseModel):
    """Per-node concordance payload returned to the frontend.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    data: List[List[Dict[str, Any]]]
    columns: List[str]
    metadata: ConcordanceMetadata
    total_matches: int | None = None
    pagination: SourceRowPagination
    sorting: AnalysisSorting
    materialized: bool | None = None


class ConcordanceAnalysisResponse(BaseModel):
    """Unified concordance response for single or multi-node requests.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    state: AnalysisTaskState
    message: str
    data: Dict[str, ConcordanceNodeResult]
    analysis_params: Optional[Dict[str, Any]] = None
    combinable: bool | None = None
    preferences: dict[str, Any] | None = None
    metadata: AnalysisTaskMetadata | None = None


class ConcordanceDispersionBinRow(BaseModel):
    """API schema used by routes and generated clients for concordance dispersion bin row.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    matched_text: str | None = None
    bin_idx: int | None = None
    count: int | None = None


class ConcordanceDispersionBinsResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for concordance dispersion bins
    response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_id: str
    total_hits: int
    document_column: str | None = None
    bin_count: int
    rows: list[ConcordanceDispersionBinRow]


# =============================================================================
# COLUMN DESCRIBE MODELS
# =============================================================================


class ColumnDescribeResponse(BaseModel):
    """Response model for column describe statistics.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column_name: str
    count: Optional[int] = None
    null_count: Optional[int] = None
    mean: ColumnScalarValue | None = None
    std: ColumnScalarValue | None = None
    min: ColumnScalarValue | None = None
    percentile_25: ColumnScalarValue | None = None
    median: ColumnScalarValue | None = None
    percentile_75: ColumnScalarValue | None = None
    max: ColumnScalarValue | None = None
