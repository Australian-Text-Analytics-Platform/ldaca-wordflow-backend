"""Core quotation analysis helpers shared by API routes.

Used by:
- FastAPI workspace analysis routers, frontend analysis features, and backend tests because they need this unit's "Core quotation analysis helpers shared by API routes" behavior.

Flow:
- Quotation routes pass saved requests, extractor rows, and artifact paths into these helpers.
- Helpers normalize pagination/context settings, compute pages, and build extraction columns.
- Response builders serialize grouped quotation rows and generated-column metadata for clients.
"""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import polars as pl
from polars.exceptions import ColumnNotFoundError

logger = logging.getLogger(__name__)

from ....core.services.quotation_client import extract_remote_quotations
from ....core.utils import stringify_unsafe_integers
from ....models.quotation import QuotationEngineConfig, QuotationEngineType
from ....settings import settings
from .generated_columns import (
    QUOTE_COLUMN_NAMES,
    QUOTE_IS_FLOATING_COLUMN,
    QUOTE_QUOTE_COLUMN,
    QUOTE_QUOTE_END_IDX_COLUMN,
    QUOTE_QUOTE_START_IDX_COLUMN,
    QUOTE_ROW_IDX_COLUMN,
    QUOTE_SPEAKER_COLUMN,
    QUOTE_SPEAKER_END_IDX_COLUMN,
    QUOTE_SPEAKER_START_IDX_COLUMN,
    QUOTE_TOKEN_COUNT_COLUMN,
    QUOTE_TYPE_COLUMN,
    QUOTE_VERB_COLUMN,
    QUOTE_VERB_END_IDX_COLUMN,
    QUOTE_VERB_START_IDX_COLUMN,
)

DEFAULT_CONTEXT_LENGTH = 20
MAX_CONTEXT_LENGTH = 2000
DEFAULT_PAGE_SIZE = 100
DEFAULT_DESCENDING = True
QUOTATION_GROUP_COLUMN = "quotation"
CORE_QUOTATION_COLUMNS = QUOTE_COLUMN_NAMES


def normalize_context_length(value: Any) -> int:
    """Normalize quote-context length into bounded integer limits.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - quotation request normalization paths because they need this unit's "Normalize quote-context length into bounded integer limits" behavior.
    - `extract_context_preference` because they need this unit's "Normalize quote-context length into bounded integer limits" behavior.

    Why:
    - Prevents invalid or oversized context values from destabilizing parsing.
    """
    try:
        numeric = int(value)
    except TypeError, ValueError:
        logger.debug("Non-numeric context length %r, using default", value)
        return DEFAULT_CONTEXT_LENGTH
    if numeric < 0:
        return 0
    if numeric > MAX_CONTEXT_LENGTH:
        return MAX_CONTEXT_LENGTH
    return numeric


def normalize_pagination(
    page: Optional[int], page_size: Optional[int]
) -> Tuple[int, int]:
    """Normalize page and page-size inputs for quotation views.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - quotation result endpoints before page computation because they need this unit's "Normalize page and page-size inputs for quotation views" behavior.

    Why:
    - Keeps pagination behavior stable across GET/POST result retrieval flows.
    """
    normalized_page = max(1, int(page)) if isinstance(page, int) else 1
    try:
        normalized_size = int(page_size) if page_size is not None else DEFAULT_PAGE_SIZE
    except TypeError, ValueError:
        logger.debug("Non-numeric page_size %r, using default", page_size)
        normalized_size = DEFAULT_PAGE_SIZE
    if normalized_size <= 0:
        normalized_size = DEFAULT_PAGE_SIZE
    return normalized_page, normalized_size


def extract_context_preference(record_result: Optional[Dict[str, Any]]) -> int:
    """Read preferred context length from a stored quotation payload.

    Used by:
    - quotation result hydration logic because they need this unit's "Read preferred context length from a stored quotation payload" behavior.

    Why:
    - Reuses saved UI preferences when users revisit existing analysis tasks.
    """
    if not record_result:
        return DEFAULT_CONTEXT_LENGTH
    prefs = record_result.get("preferences")
    if isinstance(prefs, dict) and "context_length" in prefs:
        return normalize_context_length(prefs.get("context_length"))
    return DEFAULT_CONTEXT_LENGTH


def to_polars_dataframe(data: Any) -> pl.DataFrame:
    """Convert node data into an eager Polars DataFrame.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `compute_quote_dataframe` because they need this unit's "Convert node data into an eager Polars DataFrame" behavior.

    Why:
    - Enforces strict Polars-only node data contracts for quotation analysis.
    """
    if isinstance(data, pl.LazyFrame):
        return data.collect()

    raise ValueError(
        f"Quotation analysis requires Polars LazyFrame, got {type(data).__name__}"
    )


def prepare_documents_payload(
    base_df: pl.DataFrame, column: str
) -> Dict[str, Dict[str, Any]]:
    """Build remote-extraction payload documents from a source text column.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `compute_quote_dataframe` for remote quotation engines because they need this unit's "Build remote-extraction payload documents from a source text column" behavior.

    Why:
    - Adapts tabular node data into the remote service input contract.
    """
    try:
        series = base_df.get_column(column)
    except ColumnNotFoundError as exc:  # pragma: no cover
        raise ValueError(str(exc)) from exc

    docs: Dict[str, Dict[str, Any]] = {}
    for idx, value in enumerate(series.to_list()):
        if value is None:
            text_value = ""
        elif isinstance(value, str):
            text_value = value
        else:
            text_value = str(value)
        docs[str(idx)] = {"text": text_value}
    return docs


def stable_document_items(
    documents: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return deterministically ordered document items for batching.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `batched_documents` because they need this unit's "Return deterministically ordered document items for batching" behavior.

    Why:
    - Keeps batch ordering reproducible for pagination and debugging.
    """
    items: List[Tuple[str, Dict[str, Any]]] = list(documents.items())

    def _key(pair: Tuple[str, Dict[str, Any]]) -> Tuple[int, Any]:
        """Support quotation computation helpers with a key helper.

        Steps:
        - Normalize caller input into the representation this module expects.
        - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
        - Return the compact value the caller uses for artifacts, validation, or response shaping.

        Called by:
        - The `stable_document_items` local workflow in this module because they need this unit's "Support quotation computation helpers with a key helper" behavior.
        """

        identifier = pair[0]
        try:
            return (0, int(identifier))
        except TypeError, ValueError:
            logger.debug("Non-numeric document key %r, sorting as string", identifier)
            return (1, identifier)

    items.sort(key=_key)
    return items


def batched_documents(
    documents: Dict[str, Dict[str, Any]],
    batch_size: int,
) -> Iterable[Dict[str, Dict[str, Any]]]:
    """Yield deterministic document chunks for remote extraction.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `extract_remote_paginated` because they need this unit's "Yield deterministic document chunks for remote extraction" behavior.

    Why:
    - Splits large requests to honor remote service batch limits.
    """
    if batch_size <= 0:
        batch_size = len(documents) or 1

    ordered_items = stable_document_items(documents)
    for start in range(0, len(ordered_items), batch_size):
        chunk = ordered_items[start : start + batch_size]
        yield {key: value for key, value in chunk}


async def extract_remote_paginated(
    engine: QuotationEngineConfig,
    documents: Dict[str, Dict[str, Any]],
    *,
    batch_size: int,
    timeout: float,
    extract_remote_fn,
) -> Dict[str, Any]:
    """Call remote quotation extraction in batches and merge responses.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `compute_quote_dataframe` because they need this unit's "Call remote quotation extraction in batches and merge responses" behavior.

    Why:
    - Avoids oversized single requests while preserving one combined payload.
    """
    combined_payload: Dict[str, Any] = {"results": []}
    combined_errors: List[Any] = []
    combined_warnings: List[Any] = []
    meta_captured = False

    for chunk in batched_documents(documents, batch_size):
        payload = await extract_remote_fn(
            engine,
            chunk,
            options={"preprocess": True},
            timeout=timeout,
        )

        if not isinstance(payload, dict):
            continue

        results = payload.get("results")
        if isinstance(results, list):
            combined_payload["results"].extend(results)

        errors = payload.get("errors")
        if isinstance(errors, list):
            combined_errors.extend(errors)

        warnings = payload.get("warnings")
        if isinstance(warnings, list):
            combined_warnings.extend(warnings)

        if not meta_captured and "meta" in payload:
            combined_payload["meta"] = payload["meta"]
            meta_captured = True

    if combined_errors:
        combined_payload["errors"] = combined_errors
    if combined_warnings:
        combined_payload["warnings"] = combined_warnings

    return combined_payload


def quotation_groups_via_quote_extractor(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Extract quotations using the vendored QuoteExtractor (replaces polars-text).

    Used by:
    - backend API routes, backend tests, core workspace and worker services because they need this unit's "Extract quotations using the vendored QuoteExtractor (replaces polars-text)" behavior.
    """
    from ....core.quotation_extractor import quotation_groups_for_dataframe

    return quotation_groups_for_dataframe(df, column)


def remote_payload_to_grouped_dataframe(
    base_df: pl.DataFrame,
    payload: Dict[str, Any],
) -> pl.DataFrame:
    """Attach remote quotation lists to their source rows without exploding.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - backend API routes, backend tests because they need this unit's "Attach remote quotation lists to their source rows without exploding" behavior.
    """
    results = payload.get("results", []) if isinstance(payload, dict) else []
    quotes_by_identifier: dict[str, list[dict[str, Any]]] = {}

    for result_index, entry in enumerate(results):
        if not isinstance(entry, dict):
            continue
        identifier_value = entry.get("identifier")
        identifier = (
            str(identifier_value) if identifier_value is not None else str(result_index)
        )
        quotes = entry.get("quotes")
        if not isinstance(quotes, list):
            quotes_by_identifier[identifier] = []
            continue

        normalized_quotes: list[dict[str, Any]] = []
        for quote_idx, quote in enumerate(quotes):
            if not isinstance(quote, dict):
                continue
            quote_record = cast(dict[str, Any], quote)
            normalized_quotes.append(
                {
                    "speaker": quote_record.get("speaker"),
                    "speaker_start_idx": quote_record.get("speaker_start_idx"),
                    "speaker_end_idx": quote_record.get("speaker_end_idx"),
                    "quote": quote_record.get("quote"),
                    "quote_start_idx": quote_record.get("quote_start_idx"),
                    "quote_end_idx": quote_record.get("quote_end_idx"),
                    "verb": quote_record.get("verb"),
                    "verb_start_idx": quote_record.get("verb_start_idx"),
                    "verb_end_idx": quote_record.get("verb_end_idx"),
                    "quote_type": quote_record.get("quote_type"),
                    "quote_token_count": quote_record.get("quote_token_count"),
                    "is_floating_quote": quote_record.get("is_floating_quote"),
                    "quote_row_idx": quote_record.get("quote_row_idx", quote_idx),
                }
            )
        quotes_by_identifier[identifier] = normalized_quotes

    grouped_quotes = [
        quotes_by_identifier.get(str(idx), []) for idx in range(base_df.height)
    ]
    return base_df.with_columns(pl.Series(QUOTATION_GROUP_COLUMN, grouped_quotes))


def _project_quotation_hit(raw_hit: dict[str, Any]) -> dict[str, Any]:
    """Project raw quotation-struct fields into canonical quotation columns.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Project raw quotation-struct fields into canonical quotation columns" behavior.
    """
    return {
        QUOTE_SPEAKER_COLUMN: raw_hit.get("speaker"),
        QUOTE_SPEAKER_START_IDX_COLUMN: raw_hit.get("speaker_start_idx"),
        QUOTE_SPEAKER_END_IDX_COLUMN: raw_hit.get("speaker_end_idx"),
        QUOTE_QUOTE_COLUMN: raw_hit.get("quote"),
        QUOTE_QUOTE_START_IDX_COLUMN: raw_hit.get("quote_start_idx"),
        QUOTE_QUOTE_END_IDX_COLUMN: raw_hit.get("quote_end_idx"),
        QUOTE_VERB_COLUMN: raw_hit.get("verb"),
        QUOTE_VERB_START_IDX_COLUMN: raw_hit.get("verb_start_idx"),
        QUOTE_VERB_END_IDX_COLUMN: raw_hit.get("verb_end_idx"),
        QUOTE_TYPE_COLUMN: raw_hit.get("quote_type"),
        QUOTE_TOKEN_COUNT_COLUMN: raw_hit.get("quote_token_count"),
        QUOTE_IS_FLOATING_COLUMN: raw_hit.get("is_floating_quote"),
        QUOTE_ROW_IDX_COLUMN: raw_hit.get("quote_row_idx"),
    }


def _quotation_hit_has_content(hit: dict[str, Any]) -> bool:
    """Return whether a projected quotation hit contains meaningful content.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Return whether a projected quotation hit contains meaningful content" behavior.
    """
    for key in (QUOTE_QUOTE_COLUMN, QUOTE_SPEAKER_COLUMN, QUOTE_VERB_COLUMN):
        value = hit.get(key)
        if value is None:
            continue
        if str(value).strip():
            return True
    return False


def _quotation_metadata(columns: list[str]) -> dict[str, list[str]]:
    """Support quotation computation helpers with a quotation metadata helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support quotation computation helpers with a quotation metadata helper" behavior.
    """

    return {
        "quotation_columns": [c for c in columns if c in CORE_QUOTATION_COLUMNS],
        "metadata_columns": [c for c in columns if c not in CORE_QUOTATION_COLUMNS],
        "all_columns": columns,
    }


def _empty_flattened_quotation_dataframe(result_df: pl.DataFrame) -> pl.DataFrame:
    """Support quotation computation helpers with an empty flattened quotation dataframe helper.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support quotation computation helpers with an empty flattened quotation dataframe helper" behavior.
    """

    metadata_columns = [
        column for column in result_df.columns if column != QUOTATION_GROUP_COLUMN
    ]
    schema: dict[str, pl.DataType] = {
        **{column: result_df.schema[column] for column in metadata_columns},
        QUOTE_SPEAKER_COLUMN: cast(pl.DataType, pl.Utf8),
        QUOTE_SPEAKER_START_IDX_COLUMN: cast(pl.DataType, pl.Int64),
        QUOTE_SPEAKER_END_IDX_COLUMN: cast(pl.DataType, pl.Int64),
        QUOTE_QUOTE_COLUMN: cast(pl.DataType, pl.Utf8),
        QUOTE_QUOTE_START_IDX_COLUMN: cast(pl.DataType, pl.Int64),
        QUOTE_QUOTE_END_IDX_COLUMN: cast(pl.DataType, pl.Int64),
        QUOTE_VERB_COLUMN: cast(pl.DataType, pl.Utf8),
        QUOTE_VERB_START_IDX_COLUMN: cast(pl.DataType, pl.Int64),
        QUOTE_VERB_END_IDX_COLUMN: cast(pl.DataType, pl.Int64),
        QUOTE_TYPE_COLUMN: cast(pl.DataType, pl.Utf8),
        QUOTE_TOKEN_COUNT_COLUMN: cast(pl.DataType, pl.Int64),
        QUOTE_IS_FLOATING_COLUMN: cast(pl.DataType, pl.Boolean),
        QUOTE_ROW_IDX_COLUMN: cast(pl.DataType, pl.Int64),
    }
    return pl.DataFrame(schema=schema)


def _serialize_grouped_quotation_rows(
    result_df: pl.DataFrame,
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Serialize collected quotation rows into grouped per-document hit lists.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Serialize collected quotation rows into grouped per-document hit lists" behavior.
    """
    if result_df.height == 0:
        return [], []

    metadata_columns = [
        column for column in result_df.columns if column != QUOTATION_GROUP_COLUMN
    ]
    columns = [
        *metadata_columns,
        *CORE_QUOTATION_COLUMNS,
    ]

    grouped_rows: list[list[dict[str, Any]]] = []
    for row in result_df.to_dicts():
        raw_hits = row.get(QUOTATION_GROUP_COLUMN) or []
        if not isinstance(raw_hits, list):
            continue

        base_row = {
            key: value for key, value in row.items() if key != QUOTATION_GROUP_COLUMN
        }
        grouped_hits: list[dict[str, Any]] = []
        for raw_hit in raw_hits:
            if not isinstance(raw_hit, dict):
                continue
            projected_hit = {
                **base_row,
                **_project_quotation_hit(raw_hit),
            }
            if _quotation_hit_has_content(projected_hit):
                grouped_hits.append(projected_hit)

        if grouped_hits:
            grouped_rows.append(grouped_hits)

    return grouped_rows, columns


def flatten_grouped_quotation_dataframe(result_df: pl.DataFrame) -> pl.DataFrame:
    """Flatten grouped quotation rows into a detach/export friendly dataframe.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - backend tests, core workspace and worker services because they need this unit's "Flatten grouped quotation rows into a detach/export friendly dataframe" behavior.
    """
    if result_df.height == 0:
        return _empty_flattened_quotation_dataframe(result_df)

    flattened_rows: list[dict[str, Any]] = []
    for row in result_df.to_dicts():
        raw_hits = row.get(QUOTATION_GROUP_COLUMN) or []
        if not isinstance(raw_hits, list):
            continue

        base_row = {
            key: value for key, value in row.items() if key != QUOTATION_GROUP_COLUMN
        }
        for raw_hit in raw_hits:
            if not isinstance(raw_hit, dict):
                continue
            projected_hit = {
                **base_row,
                **_project_quotation_hit(raw_hit),
            }
            if _quotation_hit_has_content(projected_hit):
                flattened_rows.append(projected_hit)

    if not flattened_rows:
        return _empty_flattened_quotation_dataframe(result_df)

    metadata_columns = [
        column for column in result_df.columns if column != QUOTATION_GROUP_COLUMN
    ]
    ordered_columns = [*metadata_columns, *CORE_QUOTATION_COLUMNS]
    return pl.DataFrame(flattened_rows).select(ordered_columns)


async def compute_quote_dataframe(
    node: Any,
    base_df: pl.DataFrame,
    column: str,
    engine: QuotationEngineConfig,
    *,
    use_base_only: bool = False,
    extract_remote_fn,
    quotation_service_max_batch_size: int,
    quotation_service_timeout: float,
) -> pl.DataFrame:
    """Compute grouped quote rows for one node/column pair.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - quotation API endpoints and on-demand page computation flow because they need this unit's "Compute grouped quote rows for one node/column pair" behavior.

    Why:
    - Abstracts local vs remote extraction behind one shared contract.
    """
    if engine.type is QuotationEngineType.REMOTE:
        documents = prepare_documents_payload(base_df, column)
        if not documents:
            return base_df.with_columns(
                pl.Series(QUOTATION_GROUP_COLUMN, [], dtype=pl.List(pl.Null))
            )
        payload = await extract_remote_paginated(
            engine,
            documents,
            batch_size=max(1, int(quotation_service_max_batch_size or 0)),
            timeout=quotation_service_timeout,
            extract_remote_fn=extract_remote_fn,
        )
        return remote_payload_to_grouped_dataframe(base_df, payload)

    if not use_base_only:
        node_data = node.data
        source_df = to_polars_dataframe(node_data)
        return quotation_groups_via_quote_extractor(source_df, column)

    return quotation_groups_via_quote_extractor(base_df, column)


async def compute_on_demand_page(
    node: Any,
    column: str,
    engine: QuotationEngineConfig,
    *,
    page: int,
    page_size: Optional[int],
    sort_by: Optional[str],
    descending: bool,
    compute_quote_dataframe_fn,
    materialized_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute one on-demand quotation page from source node data.

    - When `materialized_path` is set, paginate the flat parquet directly
      (each row becomes a single-hit group for the quotation result UI).
    - When `page_size` is None, estimate via `page_size_estimation.estimate_page_size`.

    Why:
    - Delays expensive quotation extraction to requested slices for responsive
      UI paging while keeping a dense first page via estimation.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - backend API routes because they need this unit's "Compute one on-demand quotation page from source node data" behavior.
    """
    if materialized_path:
        return await _compute_materialized_quotation_page(
            materialized_path,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            descending=descending,
        )

    lazy_df = node.data
    try:
        schema = lazy_df.collect_schema()
        available_columns = set(schema.keys())
    except Exception:
        logger.debug("Could not collect schema for quotation sort")
        available_columns = set()

    effective_sort_by = sort_by if sort_by and sort_by in available_columns else None

    if effective_sort_by:
        lazy_df = lazy_df.sort(
            pl.col(effective_sort_by),
            descending=descending,
        )

    effective_page_size = await _resolve_quotation_page_size(
        lazy_df, node, column, engine, page_size, compute_quote_dataframe_fn
    )

    total_source_rows = cast(pl.DataFrame, lazy_df.select(pl.len()).collect()).item()
    total_source_pages = (
        0
        if total_source_rows == 0
        else max(1, math.ceil(total_source_rows / effective_page_size))
    )

    start_doc = (page - 1) * effective_page_size
    slice_df = lazy_df.slice(start_doc, effective_page_size).collect()

    quote_df = await compute_quote_dataframe_fn(
        node, slice_df, column, engine, use_base_only=True
    )
    page_rows, columns = _serialize_grouped_quotation_rows(quote_df)
    metadata = _quotation_metadata(columns)

    return {
        "data": stringify_unsafe_integers(page_rows),
        "columns": columns,
        "metadata": metadata,
        "pagination": {
            "page": page,
            "page_size": effective_page_size,
            "total_source_rows": total_source_rows,
            "total_source_pages": total_source_pages,
            "result_count": len(page_rows),
            "has_next": page < total_source_pages,
            "has_prev": page > 1,
        },
        "sorting": {
            "sort_by": effective_sort_by,
            "descending": descending,
        },
    }


async def compute_remote_on_demand_page(
    node: Any,
    column: str,
    engine: QuotationEngineConfig,
    *,
    page: int,
    page_size: Optional[int],
    sort_by: Optional[str],
    descending: bool,
    materialized_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute one quotation page with the configured local/remote extractor.

    Used by:
    - quotation API result routes and quotation worker tasks because both need
      identical pagination, sorting, and service-client settings without
      depending on the router module.

    Flow: bind the configured quotation service client and settings to
        ``compute_quote_dataframe``, then delegate page collection to
        ``compute_on_demand_page``.
    """
    compute_quote_dataframe_fn = partial(
        compute_quote_dataframe,
        extract_remote_fn=extract_remote_quotations,
        quotation_service_max_batch_size=settings.quotation_service_max_batch_size,
        quotation_service_timeout=settings.quotation_service_timeout,
    )
    return await compute_on_demand_page(
        node,
        column,
        engine,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        descending=descending,
        compute_quote_dataframe_fn=compute_quote_dataframe_fn,
        materialized_path=materialized_path,
    )


async def _resolve_quotation_page_size(
    lazy_df: pl.LazyFrame,
    node: Any,
    column: str,
    engine: QuotationEngineConfig,
    requested: Optional[int],
    compute_quote_dataframe_fn,
) -> int:
    """Return an effective page size, estimating from candidate ladder if needed.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Return an effective page size, estimating from candidate ladder if needed" behavior.
    """
    if requested is not None and int(requested) > 0:
        return int(requested)

    from .page_size_estimation import DEFAULT_PAGE_SIZE_CANDIDATES, TARGET_OCCURRENCES

    async def _probe(size: int) -> int:
        """Probe runtime state used by quotation computation helpers.

        Steps:
        - Normalize caller input into the representation this module expects.
        - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
        - Return the compact value the caller uses for artifacts, validation, or response shaping.

        Called by:
        - The `_resolve_quotation_page_size` local workflow in this module because they need this unit's "Probe runtime state used by quotation computation helpers" behavior.
        """

        try:
            slice_df = lazy_df.slice(0, size).collect()
            quote_df = await compute_quote_dataframe_fn(
                node, slice_df, column, engine, use_base_only=True
            )
            if quote_df.height == 0:
                return 0
            if QUOTATION_GROUP_COLUMN not in quote_df.columns:
                return 0
            counts = quote_df.select(
                pl.col(QUOTATION_GROUP_COLUMN).list.len().fill_null(0).sum().alias("t")
            ).item()
            return int(counts or 0)
        except Exception as exc:
            logger.debug("Quotation hit probe failed at size=%d: %s", size, exc)
            return 0

    for candidate in DEFAULT_PAGE_SIZE_CANDIDATES:
        hits = await _probe(candidate)
        if hits >= TARGET_OCCURRENCES:
            return candidate
    return DEFAULT_PAGE_SIZE_CANDIDATES[-1]


async def _compute_materialized_quotation_page(
    materialized_path: str,
    *,
    page: int,
    page_size: Optional[int],
    sort_by: Optional[str],
    descending: bool,
) -> Dict[str, Any]:
    """Paginate a materialized quotation parquet as one-hit-per-group rows.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Paginate a materialized quotation parquet as one-hit-per-group rows" behavior.
    """
    effective_page_size = (
        int(page_size)
        if page_size is not None and int(page_size) > 0
        else DEFAULT_PAGE_SIZE
    )
    lazy = pl.scan_parquet(materialized_path)
    total_rows = int(cast(pl.DataFrame, lazy.select(pl.len()).collect()).item() or 0)

    effective_sort_by: Optional[str] = None
    if sort_by:
        try:
            schema = lazy.collect_schema()
            if sort_by in schema:
                lazy = lazy.sort(pl.col(sort_by), descending=descending)
                effective_sort_by = sort_by
        except Exception as exc:
            logger.debug(
                "Ignoring unsupported sort_by '%s' for materialized quotation page: %s",
                sort_by,
                exc,
            )

    start = max(page - 1, 0) * effective_page_size
    slice_df = cast(pl.DataFrame, lazy.slice(start, effective_page_size).collect())

    columns = list(slice_df.columns)
    grouped_rows: list[list[dict[str, Any]]] = []
    for row in slice_df.to_dicts():
        # The materialized parquet already uses canonical QUOTE_* column names,
        # so skip _project_quotation_hit (which expects unprefixed raw keys).
        if _quotation_hit_has_content(row):
            grouped_rows.append([row])

    total_source_pages = (
        0 if total_rows == 0 else max(1, math.ceil(total_rows / effective_page_size))
    )
    metadata = _quotation_metadata(columns)
    return {
        "data": stringify_unsafe_integers(grouped_rows),
        "columns": columns,
        "metadata": metadata,
        "pagination": {
            "page": page,
            "page_size": effective_page_size,
            "total_source_rows": total_rows,
            "total_source_pages": total_source_pages,
            "result_count": len(grouped_rows),
            "has_next": page < total_source_pages,
            "has_prev": page > 1,
        },
        "sorting": {
            "sort_by": effective_sort_by,
            "descending": descending,
        },
        "materialized": True,
    }


def collect_non_empty_quotation_corpus(
    node_data: pl.LazyFrame,
    document_column: str,
    extra_columns: list[str],
) -> tuple[list[str], dict[str, list], dict[str, Any]]:
    """Filter a lazy corpus to non-empty documents and extract needed columns.

    Used by:
    - quotation detach and materialize routes because they need the filtered document
      corpus and additional metadata columns for worker task payloads.

    Flow:
    - Filter rows where the document column is non-empty, collect the result, and return
      the document texts alongside extra column data and dtypes for worker serialization.
    """
    corpus_df = cast(
        pl.DataFrame,
        node_data.select(
            [pl.col(document_column)] + [pl.col(col) for col in extra_columns]
        )
        .filter(
            pl.col(document_column)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.len_chars()
            .fill_null(0)
            > 0
        )
        .collect(),
    )
    node_corpus = [
        str(value) if value is not None else ""
        for value in corpus_df.get_column(document_column).to_list()
    ]
    extra_columns_data: dict[str, list] = {}
    extra_columns_dtypes: dict[str, Any] = {}
    for col in extra_columns:
        series = corpus_df.get_column(col)
        extra_columns_data[col] = series.to_list()
        extra_columns_dtypes[col] = series.dtype
    return node_corpus, extra_columns_data, extra_columns_dtypes
