"""Polars expression evaluation models.

Split from models/__init__.py.
"""

from __future__ import annotations
from enum import Enum
from pydantic import BaseModel

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
    descending: bool | None = None  # used only in sort context



class PolarsExpressionRequest(BaseModel):
    """Request schema used by API routes and generated clients for polars expression request.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    context: PolarsExpressionContext
    expressions: list[PolarsExpressionItem]
    # For group_by_agg: these are the grouping key expressions
    group_by_keys: list[PolarsExpressionItem] | None = None
    new_node_name: str | None = None



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



