"""Node data, filter, slice, and column-describe models.

Split from models/__init__.py.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .analysis_common import (
    AnalysisSorting,
    AnalysisTaskState,
    NodeDataFiltering,
    PaginationInfo,
)


ColumnScalarValue = str | int | float | bool | None
FilterConditionValue = ColumnScalarValue | list[ColumnScalarValue] | dict[str, Any]


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
    value: FilterConditionValue
    id: str | None = None  # Frontend includes this for tracking
    dataType: str | None = None  # Frontend includes this for UI
    # New flags from frontend Filter UI
    negate: bool | None = False
    regex: bool | None = False
    case_sensitive: bool | None = False


class FilterRequest(BaseModel):
    """Request schema used by API routes and generated clients for filter request.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    conditions: list[FilterCondition]
    logic: str | None = "and"
    new_node_name: str | None = None


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
    length: int | None = Field(default=None, ge=0)
    sample_size: float | None = Field(default=None, gt=0)
    random_seed: int | None = Field(default=None, ge=0)
    new_node_name: str | None = None

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


class FilterPreviewResponse(BaseModel):
    """Response schema returned by API routes and consumed by generated clients for filter preview response.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    data: list[dict[str, Any]]
    columns: list[str]
    dtypes: dict[str, str]
    pagination: PaginationInfo


class ReplaceRequest(BaseModel):
    """Request schema used by API routes and generated clients for replace request.

    Used by:
    - replace node routes because replace/extract operations need one validated
      request body shared by preview and apply endpoints.

    Flow: validate the source/pattern/replacement fields and expose defaults
        used by the replace route helpers before they build Polars expressions.
    """

    source_column: str = Field(..., min_length=1, max_length=200)
    pattern: str = Field(..., min_length=1)
    replacement: str = Field(default="")
    output_column_name: str | None = Field(default=None, max_length=200)
    preview_limit: int | None = Field(default=50, ge=1, le=500)
    mode: Literal["replace", "extract"] = Field(default="replace")
    count: Literal["all", "first"] = Field(default="all")
    n: int | None = Field(default=None, ge=1)
    connector: str = Field(default=" ")


class ReplaceApplyResponse(BaseModel):
    """Response returned when a replace operation materializes a node column.

    Used by:
    - replace apply route because generated clients need the new node id,
      output column name, dtype, and user-facing completion message.
    """

    state: Literal["successful"]
    node_id: str
    column_name: str
    dtype: str | None = None
    message: str


class ConcatPreviewRequest(BaseModel):
    """Request schema used by concat preview endpoints.

    Used by:
    - concat preview and apply routes because both need the same node id list
      and deduplication flag before building aligned LazyFrames.
    """

    node_ids: list[str] = Field(..., min_length=2)
    deduplicate: bool = True


class ConcatRequest(ConcatPreviewRequest):
    """Request schema used when concatenation creates a persisted node.

    Used by:
    - concat apply route because it extends the preview contract with an
      optional output node name.
    """

    new_node_name: str | None = None


class NodeOperationResponse(BaseModel):
    """Response shared by node operations that create one child node.

    Used by:
    - filter and slice apply routes because both return the child node id and
      display name after persisting a derived node.
    """

    node_name: str
    node_id: str


class NodeActionResponse(BaseModel):
    """Response shared by node actions that do not need a full node payload.

    Used by:
    - node delete route because the frontend only needs success state and a
      message after the workspace graph is invalidated.
    """

    state: Literal["successful"]
    message: str


class CastNodeRequest(BaseModel):
    """Request body for casting one node column.

    Used by:
    - cast node route because it validates the column, target type, optional
      datetime format, and strict-mode flag before delegating to casting logic.
    """

    column: str
    target_type: str
    format: str | None = None
    strict: bool | None = None


class CastNodeInfo(BaseModel):
    """Metadata returned for a completed cast operation.

    Used by:
    - cast node response because the UI needs the original/resolved dtypes and
      effective format/strict options for feedback.
    """

    column: str
    original_type: str
    new_type: str
    target_type: str
    format_used: str | None = None
    strict_used: bool | None = None


class CastNodeResponse(BaseModel):
    """Response returned after casting one node column.

    Used by:
    - cast node route and generated clients because the route returns an action
      status plus typed cast metadata instead of a full node snapshot.
    """

    state: Literal["successful"]
    node_id: str
    cast_info: CastNodeInfo
    message: str


ColumnScalarValue = str | int | float | bool
AnalysisTaskState = Literal["pending", "running", "successful", "failed", "cancelled"]


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
    sorting: AnalysisSorting
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


class ColumnDescribeResponse(BaseModel):
    """Response model for column describe statistics.

    Used by:
    - backend API routes, backend request/response models because they need a stable JSON
      contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    column_name: str
    count: int | None = None
    null_count: int | None = None
    mean: ColumnScalarValue | None = None
    std: ColumnScalarValue | None = None
    min: ColumnScalarValue | None = None
    percentile_25: ColumnScalarValue | None = None
    median: ColumnScalarValue | None = None
    percentile_75: ColumnScalarValue | None = None
    max: ColumnScalarValue | None = None
