"""Shared analysis models (metadata, pagination, sorting, state).

Split from models/__init__.py.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TextSetupRequest(BaseModel):
    """Request schema used by API routes and generated clients for text setup request.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    document_column: str
    content_column: str | None = None
    auto_detect: bool = True


class DTMRequest(BaseModel):
    """Request schema used by API routes and generated clients for d t m request.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    max_features: int | None = 1000
    min_df: float = 0.01
    max_df: float = 0.95
    ngram_range: tuple = (1, 2)
    use_tfidf: bool = False


class KeywordExtractionRequest(BaseModel):
    """Request schema used by API routes and generated clients for keyword extraction request.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    method: str  # 'tfidf', 'count', 'custom'
    top_k: int = 20
    by_document: bool = False


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


class TextAnalysisInfo(BaseModel):
    """Metadata schema used by API responses to describe text analysis info.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    document: str | None = None
    avg_document_length: float | None = None
    total_documents: int
    vocabulary_size: int | None = None
    is_text_ready: bool


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


class DetachNodeOption(BaseModel):
    """Shared base for detach-node-option responses across analysis tools.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_id: str
    node_name: str
    text_column: str | None = None
    available_columns: list[str]
    disabled_columns: list[str] = Field(default_factory=list)
    # Columns to tick by default when the detach dialog opens. When None the
    # client falls back to its per-tool default (e.g. select-all). Topic
    # modeling sets this to its generated columns so source columns start
    # unticked, matching concordance/quotation.
    default_selected_columns: list[str] | None = None


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


ColumnScalarValue = str | int | float | bool
AnalysisTaskState = Literal["pending", "running", "successful", "failed", "cancelled"]
