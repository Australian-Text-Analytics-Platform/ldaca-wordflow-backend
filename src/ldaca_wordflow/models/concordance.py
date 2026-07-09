"""Concordance analysis request and response models.

Split from models/__init__.py.
"""

from __future__ import annotations

from typing import Any, Literal

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

    node_ids: list[str]  # Support up to 2 nodes (1 = single node mode)
    node_columns: dict[str, str]  # node_id -> column_name mapping
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
    sort_by: str | None = None  # column name to sort by
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
    new_node_name: str | None = None  # If not provided, will be auto-generated
    selected_columns: list[str] = Field(min_length=1)
    materialized_path: str | None = None  # Reuse existing flattened parquet
    model_config = ConfigDict(extra="forbid")


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

    node_id: str
    column: str
    search_word: str
    num_left_tokens: int = 10
    num_right_tokens: int = 10
    regex: bool = False
    whole_word: bool = False
    case_sensitive: bool = False
    new_node_name: str | None = None
    selected_columns: list[str] = Field(min_length=1)
    materialized_path: str | None = None
    # When the slow path runs (no materialized_path), the worker also writes
    # the materialised flat parquet so the user doesn't have to click "Process
    # All" separately before iterating on bin selections. The shared
    # analysis-task route provides the parent task id for the standard
    # `analysis_materialized` event.
    selected_bins: list[int] | None = None
    total_bins: int | None = None
    # When the chart legend is filtered, the detach should aggregate only over
    # hits whose `CONC_matched_text` lands in this set. `None` means "all".
    # `match_case_insensitive` mirrors the chart's `lowercaseMatches` toggle:
    # when true, both the column and the candidate set are lowercased before
    # comparison so the filter agrees with the legend grouping.
    selected_matched_texts: list[str] | None = None
    match_case_insensitive: bool = False
    model_config = ConfigDict(extra="forbid")


class ConcordanceMaterializeRequest(BaseModel):
    """Request schema used by API routes and generated clients for concordance materialize request.

    Used by:
    - backend API routes, backend request/response models, backend tests because they need a
      stable JSON contract shared by route handlers, generated clients, and tests.

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
    # Mirror the live ``/concordance`` request — materialize must honour the
    # engine the user actually searched with. Defaults to ``"regex"`` so
    # existing English flows are byte-identical.
    search_mode: Literal["regex", "tokens"] = "regex"
    model_config = ConfigDict(extra="forbid")


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
    data: dict[str, list[DetachNodeOption]] | None = None
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

    concordance_columns: list[str]  # Core concordance columns (CONC_left_context, CONC_matched_text, CONC_right_context, etc.)
    metadata_columns: list[str]  # Original document metadata columns
    all_columns: list[str]  # All available columns


class ConcordanceNodeResult(BaseModel):
    """Per-node concordance payload returned to the frontend.

    Used by:
    - backend request/response models because they need a stable JSON contract shared by
      route handlers, generated clients, and tests.

    Flow: validate incoming API fields, apply defaults or validators, and serialize route
        responses in the shape expected by frontend clients and tests.
    """

    data: list[list[dict[str, Any]]]
    columns: list[str]
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
    data: dict[str, ConcordanceNodeResult]
    analysis_params: dict[str, Any] | None = None
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
