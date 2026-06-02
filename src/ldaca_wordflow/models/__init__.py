"""Pydantic request/response models for backend API contracts.

Used by:
- API routers and worker result serialization boundaries because they need a backend
  boundary that validates inputs before delegating to workspace or worker state.
Why:
- Centralizes schema contracts shared between frontend and backend endpoints.

Note:
- Most domain models now live in separate files under models/ (auth.py, files.py,
  workspace.py, nodes.py, concordance.py, etc.). This __init__.py re-exports them
  for backward compatibility so that ``from ..models import X`` still works.

Flow: validate incoming API fields, apply defaults or validators, and serialize route
    responses in the shape expected by frontend clients and tests.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# =============================================================================
# RE-EXPORTS FROM DOMAIN MODEL FILES
#
# These classes and type aliases were split out of __init__.py into dedicated
# domain files. They are re-exported here for backward compatibility.
# =============================================================================

from .ai_annotation import AiAnnotationCategoriesResponse, AiAnnotationDetachRequest
from .ai_annotation import AiAnnotationDetachResponse, AiAnnotationModelsRequest
from .ai_annotation import AiAnnotationModelsResponse, AiAnnotationProvidersResponse
from .ai_annotation import AiAnnotationRequest, AiAnnotationResponse, AiAnnotationResultQuery
from .ai_annotation import AiAnnotationSaveRequest, AiAnnotationSaveResponse
from .analysis_common import AnalysisClearResponse, AnalysisSorting, AnalysisTaskActionResponse
from .analysis_common import AnalysisTaskMetadata, CurrentAnalysisTasksResponse, DetachNodeOption
from .analysis_common import PaginationInfo
from .auth import AuthInfoResponse, GoogleIn, GoogleOut, User, UserResponse
from .concordance import ConcordanceAnalysisRequest, ConcordanceAnalysisResponse
from .concordance import ConcordanceDetachNodeOption, ConcordanceDetachOptionsResponse
from .concordance import ConcordanceDetachRequest, ConcordanceDispersionBinsResponse
from .concordance import ConcordanceDispersionDetachRequest, ConcordanceMaterializeRequest
from .files import CreateFolderRequest, CreateFolderResponse, FileInfoResponse
from .files import FileTreeNodeResponse, FileUploadResponse, FilesImportTaskStartResponse
from .files import FilesTaskActionResponse, FilesTasksListResponse, ImportSampleDataRequest
from .files import ImportSampleDataResponse, LDaCAImportRequest, MessageResponse, MoveFileRequest
from .files import OniSearchRequest, OniSearchResponse, OniSearchResult, RenameColumnRequest
from .files import SampleDataCatalogueResponse, SampleDataCollection, SampleDataFileEntry
from .files import TaskCancelActionResponse, TaskClearActionResponse, TaskListResponse
from .nodes import ColumnDescribeResponse, ColumnOperationsResponse, ColumnUniqueValuesResponse
from .nodes import FilterCondition, FilterPreviewResponse, FilterRequest, NodeDataResponse
from .nodes import NodeQueryPlanResponse, NodeShapeResponse, SliceRequest
from .polars_expression import PolarsExpressionApplyResponse, PolarsExpressionContext
from .polars_expression import PolarsExpressionItem, PolarsExpressionRequest
from .preferences import QuotationPreferences
from .quotation import QuotationAnalysisResponse, QuotationDetachNodeOption
from .quotation import QuotationDetachOptionsResponse, QuotationDetachRequest
from .quotation import QuotationEngineConfig, QuotationEngineType
from .quotation import QuotationMaterializeRequest, QuotationPreferenceUpdateResponse
from .quotation import QuotationRequest, QuotationResultQuery
from .sequential_analysis import SequentialAnalysisDetachResponse
from .sequential_analysis import SequentialAnalysisPreferenceUpdateRequest
from .sequential_analysis import SequentialAnalysisPreferenceUpdateResponse
from .sequential_analysis import SequentialAnalysisPreviewResponse, SequentialAnalysisRequest
from .sequential_analysis import SequentialAnalysisResponse
from .token_frequencies import TokenFrequencyPreferenceUpdateRequest, TokenFrequencyRequest
from .token_frequencies import TokenFrequencyResponse
from .topic_modeling import TopicModelingData, TopicModelingDetachData
from .topic_modeling import TopicModelingDetachNodeOption, TopicModelingDetachOptionsResponse
from .topic_modeling import TopicModelingDetachRequest, TopicModelingDetachResponse
from .topic_modeling import TopicModelingDetachedNode
from .topic_modeling import TopicModelingEmbeddingCacheClearResponse
from .topic_modeling import TopicModelingEmbeddingCacheSizeResponse, TopicModelingRequest
from .topic_modeling import TopicModelingResponse, TopicModelingResultUpdateRequest
from .workspace import CurrentWorkspaceResponse, NodeDocumentColumnUpdateRequest
from .workspace import NodeTokenizationPreferenceRequest, SetCurrentWorkspaceResponse
from .workspace import TokenizerModelsResponse, WorkspaceActionResponse, WorkspaceCreateRequest
from .workspace import WorkspaceGraphResponse, WorkspaceInfo, WorkspaceNodeInfo
from .workspace import WorkspaceNodesResponse, WorkspaceSummary, WorkspaceTaskStartResponse
from .workspace import WorkspaceUploadResponse

# =============================================================================
# CLASSES UNIQUE TO THIS MODULE (no domain-file equivalent)
# =============================================================================

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
