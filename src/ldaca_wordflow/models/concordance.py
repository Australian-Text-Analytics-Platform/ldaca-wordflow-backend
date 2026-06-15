"""Concordance analysis request and response models.

Split from models/__init__.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .analysis_common import (
    AnalysisSorting,
    AnalysisTaskMetadata,
    AnalysisTaskState,
    DetachNodeOption,
    PaginationInfo,
    SourceRowPagination,
)


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
    # "regex" (default) uses the polars-text concordance engine on raw text,
    # preserving partial-word patterns like ``equ\w*`` for English users.
    # "tokens" looks up a tokenization column and walks it for exact-token
    # matches with N-actual-token left/right context — the
    # word-aware semantics CJK users want once Tokenise has been run.
    # Falls back to regex behaviour if no tokenization column exists.
    search_mode: Literal["regex", "tokens"] = "regex"
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
    parent_task_id: str


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
    data: Dict[str, List[DetachNodeOption]] | None = None
    metadata: AnalysisTaskMetadata | None = None


# Quotation requests (mirror concordance shape but without search parameters)


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
