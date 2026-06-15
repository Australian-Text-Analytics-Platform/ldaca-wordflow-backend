"""Quotation analysis request and response models.

Split from models/__init__.py.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator
from typing_extensions import TypedDict

from .analysis_common import (
    AnalysisSorting,
    AnalysisTaskMetadata,
    AnalysisTaskState,
    DetachNodeOption,
    PaginationInfo,
    SourceRowPagination,
)


class QuotationHitEntry(TypedDict, total=False):
    """Fixed keys on a quotation hit row from ``_project_quotation_hit``.

    Keys match the ``QUOTE_*`` constants in ``api/workspaces/analyses/generated_columns.py``.
    Metadata columns from the source node vary per workspace and are *not* listed here.
    """

    QUOTE_speaker: str | None
    QUOTE_speaker_start_idx: int | None
    QUOTE_speaker_end_idx: int | None
    QUOTE_quote: str | None
    QUOTE_quote_start_idx: int | None
    QUOTE_quote_end_idx: int | None
    QUOTE_verb: str | None
    QUOTE_verb_start_idx: int | None
    QUOTE_verb_end_idx: int | None
    QUOTE_quote_type: str | None
    QUOTE_quote_token_count: int | None
    QUOTE_is_floating_quote: bool | None
    QUOTE_quote_row_idx: int | None


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

    model_config = ConfigDict(extra="forbid")


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
    data: Dict[str, List[DetachNodeOption]] | None = None
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

    data: list[
        list[dict[str, Any]]
    ]  # inner dicts are QuotationHitEntry + dynamic metadata columns
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
