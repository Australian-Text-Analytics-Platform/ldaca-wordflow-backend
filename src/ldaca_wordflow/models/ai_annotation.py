"""AI annotation and LLM-prompting models.

Split from models/__init__.py.
"""

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field
from ..analysis.models import BaseAnalysisRequest
from .analysis_common import AnalysisSorting, AnalysisTaskMetadata, AnalysisTaskState, SourceRowPagination

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

    base_url: str | None = None
    api_key: str | None = None



class AiAnnotationRequest(BaseAnalysisRequest):
    """Request schema used by API routes and generated clients for ai annotation request.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    node_ids: list[str]
    node_columns: dict[str, str]
    annotation_column: str | None = None

    classes: list[AiAnnotationClassDef] = Field(min_length=1)
    examples: list[AiAnnotationExample] = Field(default_factory=list)

    model: str
    api_key: str | None = None
    base_url: str | None = None

    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1.0)
    seed: int | None = 42
    batch_size: int = Field(default=100, ge=1)

    page: int = 1
    page_size: int = 20
    sort_by: str | None = None
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
    new_node_name: str | None = None
    annotation_column: str | None = None

    classes: list[AiAnnotationClassDef] = Field(min_length=1)
    examples: list[AiAnnotationExample] = Field(default_factory=list)

    model: str
    api_key: str | None = None
    base_url: str | None = None

    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1.0)
    seed: int | None = 42
    batch_size: int = Field(default=100, ge=1)



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

    annotation_column: str | None = None
    edits: list[AiAnnotationEdit] = Field(default_factory=list)



class AiAnnotationNodeResult(BaseModel):
    """API schema used by routes and generated clients for ai annotation node result.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    data: list[dict[str, Any]]
    columns: list[str]
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

    page: int | None = None
    page_size: int | None = None
    sort_by: str | None = None
    descending: bool | None = None

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
    data: dict[str, AiAnnotationNodeResult] | None = None
    analysis_params: dict[str, Any] | None = None
    combinable: bool | None = None
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



