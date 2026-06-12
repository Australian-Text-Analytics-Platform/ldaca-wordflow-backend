"""FastAPI integration models for DocWorkspace.

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: normalize inputs, delegate to the owning backend state or service boundary, and
    return serialized values or existing domain errors to callers.
"""

from typing import Any

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    """Schema information for a single column.

    Used by:
    - core workspace and worker services because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """

    name: str
    dtype: str
    js_type: str = Field(
        ...,
        description=(
            "JavaScript-compatible type "
            "(string, categorical, integer, float, boolean, datetime, "
            "list[string], annotation, tmdist, unknown)"
        ),
    )


class NodeSummary(BaseModel):
    """Summary information about a Node for API responses.

    Used by:
    - Backend services, routes, and tests that import this symbol because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """

    id: str
    name: str
    operation: str | None = None
    shape: tuple[int, int] = (0, 0)
    columns: list[str] = Field(default_factory=list)
    node_schema: list[ColumnSchema] = Field(default_factory=list, alias="schema")
    document: str | None = None
    parent_ids: list[str] = Field(default_factory=list)
    child_ids: list[str] = Field(default_factory=list)


class PaginatedData(BaseModel):
    """Paginated data response for large datasets.

    Used by:
    - Backend services, routes, and tests that import this symbol because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """

    data: list[dict[str, Any]]
    pagination: dict[str, Any] = Field(
        default_factory=lambda: {
            "page": 1,
            "page_size": 100,
            "total_rows": 0,
            "total_pages": 0,
            "has_next": False,
            "has_previous": False,
        }
    )
    columns: list[str] = Field(default_factory=list)
    data_schema: list[ColumnSchema] = Field(default_factory=list, alias="schema")


class ErrorResponse(BaseModel):
    """Standard error response format.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """

    error: str
    message: str
    details: dict[str, Any] | None = None


class OperationResult(BaseModel):
    """Result of a workspace operation.

    Used by:
    - Backend services, routes, and tests that import this symbol because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: normalize inputs, delegate to the owning backend state or service boundary, and
        return serialized values or existing domain errors to callers.
    """

    success: bool
    message: str
    node_id: str | None = None
    data: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
