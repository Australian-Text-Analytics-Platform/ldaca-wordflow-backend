"""File management, import, and data catalogue models.

Split from models/__init__.py.
"""

from __future__ import annotations
from typing import Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, model_validator


class TaskEntry(TypedDict, total=False):
    """Dict shape of a serialized task from ``WorkerTaskManager._serialize_task``.

    Keys match ``core/worker_task_manager.py:_serialize_task``.
    """

    task_id: str
    task_type: str
    name: str
    user_id: str
    workspace_id: str
    state: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    progress: float
    progress_message: str | None
    parent_task_id: str | None

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
    sample_dir: str | None = None
    remote_download_started: bool = False



class ImportSampleDataRequest(BaseModel):
    """Request schema used by API routes and generated clients for import sample data request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    collection_ids: list[str] = Field(default_factory=list)



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
    recommended_for: list[str]
    files: list[SampleDataFileEntry]
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
    collections: list[SampleDataCollection]


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
    filename: str | None = None



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
    size: int | None = None
    children: list[FileTreeNodeResponse] | None = None

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
    data: list[TaskEntry]
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
    data: list[dict[str, Any]]
    message: str



class FilesTaskActionDataResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for files task action data response.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    cancelled: bool | None = None
    cancelled_count: int | None = None
    cleared_count: int | None = None



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
