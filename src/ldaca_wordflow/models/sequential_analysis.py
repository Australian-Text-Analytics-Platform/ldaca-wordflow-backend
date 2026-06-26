"""Sequential/chart analysis request and response models.

Split from models/__init__.py.
"""

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, model_validator
from .analysis_common import AnalysisSorting, AnalysisTaskMetadata, AnalysisTaskState

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
    group_by_columns: list[str] | None = None
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
    numeric_origin: float | None = None
    numeric_interval: float | None = None
    custom_interval_value: int | None = None
    custom_interval_unit: Literal["seconds", "minutes", "hours", "days", "weeks"] | None = None
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



