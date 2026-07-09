"""Core concordance computation helpers shared by route handlers.

Used by:
- FastAPI workspace analysis routers, frontend analysis features, and backend tests because they need this unit's "Core concordance computation helpers shared by route handlers" behavior.

Flow:
- Route handlers pass saved requests, node data, and artifact paths into these helpers.
- Helpers normalize request payloads, compute regex/token concordance pages, and attach metadata.
- Response builders serialize dense page payloads, dispersion bins, and generated-column artifacts.
"""

from __future__ import annotations

import logging
import math
import re
from functools import partial
from typing import Any, Optional, cast

import polars as pl

from ....core.exceptions import InvalidInputError
from ....core.utils import stringify_unsafe_integers
from .concordance_tokens_mode import (
    compute_tokens_concordance_page,
    find_token_matches,
)
from .generated_columns import (
    CONC_END_IDX_COLUMN,
    CONC_EXTRACTION_COLUMN,
    CONC_L1_COLUMN,
    CONC_LEFT_CONTEXT_COLUMN,
    CONC_MATCHED_TEXT_COLUMN,
    CONC_R1_COLUMN,
    CONC_RIGHT_CONTEXT_COLUMN,
    CONC_START_IDX_COLUMN,
    CORE_CONCORDANCE_COLUMNS,
    MATERIALIZED_CONCORDANCE_COLUMNS,
    compute_concordance_extraction_string,
)
from .page_size_estimation import DEFAULT_PAGE_SIZE_CANDIDATES, estimate_page_size
from ....core.exceptions import NotFoundError, WorkspaceNotFoundError
from ..utils import require_workspace


logger = logging.getLogger(__name__)

DEFAULT_CONCORDANCE_PAGE = 1
DEFAULT_CONCORDANCE_PAGE_SIZE = 20
DEFAULT_CONCORDANCE_DESCENDING = True

_REQUEST_EXCLUDE_KEYS = {
    "page",
    "page_size",
    "sort_by",
    "descending",
    "pagination",
    # ``combined`` is not part of the backend request contract; the comparison
    # view is synthesized client-side from per-node result payloads.
    "combined",
}


def normalize_saved_request(raw_request: Optional[dict]) -> Optional[dict]:
    """Normalize stored concordance request payloads.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `sanitize_request_for_storage` because they need this unit's "Normalize stored concordance request payloads" behavior.
    - concordance result endpoints before response rebuild because they need this unit's "Normalize stored concordance request payloads" behavior.

    Why:
    - Ensures persisted requests omit view-only keys and keep stable defaults.
    """
    if not raw_request:
        return None
    if "node_ids" not in raw_request or "node_columns" not in raw_request:
        return None

    normalized_request = dict(raw_request)
    for field in _REQUEST_EXCLUDE_KEYS:
        normalized_request.pop(field, None)

    normalized_request = {
        key: value for key, value in normalized_request.items() if value is not None
    }
    return normalized_request


def sanitize_request_for_storage(request_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a storage-safe concordance request snapshot.

    Used by:
    - Concordance route handlers when persisting task requests because they need this unit's "Return a storage-safe concordance request snapshot" behavior.

    Why:
    - Prevents transient pagination/sorting fields from polluting saved inputs.
    """
    normalized = normalize_saved_request(request_dict)
    return normalized or {}


def concordance_non_empty_expr() -> pl.Expr:
    """Build an expression that removes empty concordance rows.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `build_concordance_lazyframe` because they need this unit's "Build an expression that removes empty concordance rows" behavior.

    Why:
    - Drops rows with no meaningful matched/context text before pagination.
    """
    return pl.any_horizontal(
        [
            pl.col(CONC_MATCHED_TEXT_COLUMN)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.len_chars()
            .fill_null(0)
            > 0,
            pl.col(CONC_LEFT_CONTEXT_COLUMN)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.len_chars()
            .fill_null(0)
            > 0,
            pl.col(CONC_RIGHT_CONTEXT_COLUMN)
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .str.len_chars()
            .fill_null(0)
            > 0,
        ]
    )


def build_concordance_lazyframe(
    node_data: pl.LazyFrame,
    column: str,
    request: dict[str, Any],
) -> pl.LazyFrame:
    """Create concordance rows from a source LazyFrame and request options.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `compute_concordance_page` because they need this unit's "Create concordance rows from a source LazyFrame and request options" behavior.

    Why:
    - Encapsulates `polars_text.concordance` expansion and filtering in one
      reusable transformation step.
    """
    import polars_text as pt

    search_pattern, use_regex = build_concordance_search_pattern(
        request["search_word"],
        regex=bool(request["regex"]),
        whole_word=bool(request.get("whole_word", False)),
    )

    expr = pt.concordance(
        pl.col(column),
        search_pattern,
        num_left_tokens=request["num_left_tokens"],
        num_right_tokens=request["num_right_tokens"],
        regex=use_regex,
        case_sensitive=request["case_sensitive"],
    )
    return node_data.select([pl.all(), expr.alias("concordance")])


def build_concordance_search_pattern(
    search_word: str,
    *,
    regex: bool,
    whole_word: bool,
) -> tuple[str, bool]:
    """Return the effective concordance pattern and whether regex mode is needed.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - backend API routes, backend tests, core workspace and worker services because they need this unit's "Return the effective concordance pattern and whether regex mode is needed" behavior.
    """
    if not whole_word:
        return search_word, regex

    base_pattern = search_word if regex else re.escape(search_word)
    return rf"\b(?:{base_pattern})\b", True


def _project_concordance_hit(
    raw_hit: dict[str, Any],
    *,
    document_text: Optional[str] = None,
) -> dict[str, Any]:
    """Project one raw concordance struct into canonical response columns.

    When ``document_text`` is provided, ``CONC_extraction`` is computed
    using the same slicing rule as the worker-side materialised parquet.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Project one raw concordance struct into canonical response columns" behavior.
    """
    start_idx = raw_hit.get("start_idx")
    end_idx = raw_hit.get("end_idx")
    projected: dict[str, Any] = {
        CONC_LEFT_CONTEXT_COLUMN: raw_hit.get("left_context"),
        CONC_MATCHED_TEXT_COLUMN: raw_hit.get("matched_text"),
        CONC_RIGHT_CONTEXT_COLUMN: raw_hit.get("right_context"),
        CONC_START_IDX_COLUMN: start_idx,
        CONC_END_IDX_COLUMN: end_idx,
        CONC_L1_COLUMN: raw_hit.get("l1"),
        CONC_R1_COLUMN: raw_hit.get("r1"),
    }
    if document_text is not None and start_idx is not None and end_idx is not None:
        projected[CONC_EXTRACTION_COLUMN] = compute_concordance_extraction_string(
            document_text=document_text,
            left_context=raw_hit.get("left_context"),
            right_context=raw_hit.get("right_context"),
            start_idx=int(start_idx),
            end_idx=int(end_idx),
        )
    return projected


def _concordance_hit_has_content(hit: dict[str, Any]) -> bool:
    """Return whether a projected concordance hit contains meaningful text.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Return whether a projected concordance hit contains meaningful text" behavior.
    """
    for key in (
        CONC_MATCHED_TEXT_COLUMN,
        CONC_LEFT_CONTEXT_COLUMN,
        CONC_RIGHT_CONTEXT_COLUMN,
    ):
        value = hit.get(key)
        if value is None:
            continue
        if str(value).strip():
            return True
    return False


def _column_metadata(
    columns: list[str],
    concordance_columns: tuple[str, ...],
) -> dict[str, list[str]]:
    """Support concordance computation helpers with a column metadata helper.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Support concordance computation helpers with a column metadata helper" behavior.
    """

    return {
        "concordance_columns": [c for c in columns if c in concordance_columns],
        "metadata_columns": [c for c in columns if c not in concordance_columns],
        "all_columns": columns,
    }


def _serialize_grouped_concordance_rows(
    result_df: pl.DataFrame,
    *,
    node_label: Optional[str] = None,
    text_column: Optional[str] = None,
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Serialize collected concordance rows into grouped per-document hit lists.

    When ``text_column`` is given and that column survives on the result frame
    (it normally does — ``build_concordance_lazyframe`` keeps ``pl.all()``),
    each projected hit gets a ``CONC_extraction`` field with the stitched
    raw KWIC window.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Serialize collected concordance rows into grouped per-document hit lists" behavior.
    """
    if result_df.height == 0:
        return [], []

    metadata_columns = [
        column for column in result_df.columns if column != "concordance"
    ]
    has_extraction = bool(text_column) and text_column in metadata_columns
    columns = [
        *metadata_columns,
        *CORE_CONCORDANCE_COLUMNS,
    ]
    if has_extraction:
        columns.append(CONC_EXTRACTION_COLUMN)
    if node_label:
        columns.append("__source_node")

    grouped_rows: list[list[dict[str, Any]]] = []
    for row in result_df.to_dicts():
        raw_hits = row.get("concordance") or []
        if not isinstance(raw_hits, list):
            continue

        base_row = {key: value for key, value in row.items() if key != "concordance"}
        document_text: Optional[str] = None
        if has_extraction:
            raw_doc = base_row.get(text_column)
            document_text = str(raw_doc) if raw_doc is not None else ""
        grouped_hits: list[dict[str, Any]] = []
        for raw_hit in raw_hits:
            if not isinstance(raw_hit, dict):
                continue
            projected_hit = {
                **base_row,
                **_project_concordance_hit(raw_hit, document_text=document_text),
            }
            if node_label:
                projected_hit["__source_node"] = node_label
            if _concordance_hit_has_content(projected_hit):
                grouped_hits.append(projected_hit)

        if grouped_hits:
            grouped_rows.append(grouped_hits)

    return grouped_rows, columns


def resolve_node_sources(
    user_id: str,
    workspace_id: str,
    request: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str], str | None]:
    """Resolve workspace nodes into validated concordance source metadata.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `build_concordance_response` because they need this unit's "Resolve workspace nodes into validated concordance source metadata" behavior.

    Why:
    - Centralizes node lookup, label mapping, and LazyFrame/type validation.
    - Returns a 4-tuple (sources, label_map, labels, error). When error is not None the
      first three values are empty fallbacks.
    """
    node_ids = request.get("node_ids") or []
    node_columns = request.get("node_columns") or {}

    node_sources: dict[str, dict[str, Any]] = {}
    label_to_node_map: dict[str, str] = {}
    node_labels: dict[str, str] = {}
    try:
        workspace = require_workspace(user_id, workspace_id)
    except WorkspaceNotFoundError:
        return {}, {}, {}, "Workspace not found"

    for node_id in node_ids:
        node = workspace.nodes[node_id]
        node_label = getattr(node, "name", None) or node_id
        label_to_node_map[node_label] = node_id
        node_labels[node_id] = node_label
        node_data = node.data
        column = node_columns.get(node_id)
        if not column:
            continue
        tokenization_column: Optional[str] = None
        if hasattr(node, "find_tokenization_column"):
            tokenization_column = node.find_tokenization_column(column)
        node_sources[node_id] = {
            "lf": node_data,
            "column": column,
            "label": node_label,
            "tokenization_column": tokenization_column,
            "node": node,
            "user_id": user_id,
        }

    return node_sources, label_to_node_map, node_labels, None


def compute_concordance_page(
    base_lf: pl.LazyFrame,
    column: str,
    request: dict[str, Any],
    *,
    page: int,
    page_size: Optional[int],
    sort_by: Optional[str],
    descending: bool,
    node_label: Optional[str] = None,
) -> dict[str, Any]:
    """Compute one concordance page for a single node source.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `build_concordance_response` because they need this unit's "Compute one concordance page for a single node source" behavior.
    - `collect_interleaved_combined` because they need this unit's "Compute one concordance page for a single node source" behavior.

    Why:
    - Produces a stable page payload shape shared by single-node and combined
      result views. When `page_size` is None the size is estimated from the
      configured candidate ladder so the first page yields dense results.
    """
    total_rows_df = cast(pl.DataFrame, base_lf.select(pl.len()).collect())
    total_source_rows = total_rows_df.item()

    resolved_page_size = _resolve_page_size(base_lf, column, request, page_size)

    effective_sort_by: Optional[str] = None
    if sort_by:
        try:
            schema = base_lf.collect_schema()
            if sort_by in schema and sort_by not in CORE_CONCORDANCE_COLUMNS:
                base_lf = base_lf.sort(sort_by, descending=descending)
                effective_sort_by = sort_by
        except Exception as exc:
            logger.debug(
                "Ignoring unsupported sort_by '%s' for concordance page: %s",
                sort_by,
                exc,
            )

    start = max(page - 1, 0) * resolved_page_size
    page_lf = base_lf.slice(start, resolved_page_size)

    concordance_lf = build_concordance_lazyframe(page_lf, column, request)
    result_df = cast(pl.DataFrame, concordance_lf.collect())
    page_rows, columns = _serialize_grouped_concordance_rows(
        result_df,
        node_label=node_label,
        text_column=column,
    )

    total_source_pages = max(1, math.ceil(total_source_rows / resolved_page_size))

    # `CONC_extraction` is intentionally classified under `metadata_columns`
    # so the existing metadata-columns picker offers it as an opt-in toggle,
    # matching the rest of the user-controllable column set. The CONC_
    # prefix makes the source obvious; behaviourally it's "an optional column
    # you can show / detach if you want it."
    metadata = _column_metadata(columns, CORE_CONCORDANCE_COLUMNS)

    return {
        "data": stringify_unsafe_integers(page_rows),
        "columns": columns,
        "metadata": metadata,
        "pagination": {
            "page": page,
            "page_size": resolved_page_size,
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


def compute_node_concordance_page(
    src: dict[str, Any],
    request: dict[str, Any],
    *,
    page: int,
    page_size: Optional[int],
    sort_by: Optional[str],
    descending: bool,
) -> dict[str, Any]:
    """Route a node to either regex-mode or tokens-mode page computation.

    Tokens-mode only activates when ``request['search_mode'] == 'tokens'``
    AND the node carries tokenization metadata for the source column.
    Otherwise we fall back to the existing regex/text path so EN goldens
    stay byte-identical.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - backend API routes, backend tests because they need this unit's "Route a node to either regex-mode or tokens-mode page computation" behavior.
    """
    base_lf = src["lf"]
    column = src["column"]
    label = src.get("label")
    tokenization_column = src.get("tokenization_column")
    search_mode = str(request.get("search_mode") or "regex")

    node_request: dict[str, Any] = dict(request)

    if search_mode == "tokens" and tokenization_column:
        token_node = src.get("node")
        source_user_id = src.get("user_id")
        if token_node is not None and source_user_id:
            from ....core.tokens_cache import hydrate_tokenization_lazyframe

            base_lf = hydrate_tokenization_lazyframe(
                node=token_node,
                source_column=column,
                user_id=source_user_id,
            )
        effective_page_size = (
            int(page_size)
            if page_size is not None and int(page_size) > 0
            else DEFAULT_CONCORDANCE_PAGE_SIZE
        )
        return compute_tokens_concordance_page(
            base_lf,
            column=column,
            tokenization_column=tokenization_column,
            request=node_request,
            page=page,
            page_size=effective_page_size,
            sort_by=sort_by,
            descending=descending,
            node_label=label,
        )
    return compute_concordance_page(
        base_lf,
        column,
        node_request,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        descending=descending,
        node_label=label,
    )


def _count_concordance_hits(
    base_lf: pl.LazyFrame,
    column: str,
    request: dict[str, Any],
    size: int,
) -> int:
    """Return occurrence count when running concordance on the first `size` rows.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Return occurrence count when running concordance on the first `size` rows" behavior.
    """
    try:
        slice_lf = build_concordance_lazyframe(base_lf.slice(0, size), column, request)
        count_df = cast(
            pl.DataFrame,
            slice_lf.select(
                pl.col("concordance").list.len().fill_null(0).sum().alias("total")
            ).collect(),
        )
        value = count_df.item()
    except Exception as exc:
        logger.debug("Concordance hit probe failed at size=%d: %s", size, exc)
        return 0
    return int(value or 0)


def _count_tokens_concordance_hits(
    base_lf: pl.LazyFrame,
    column: str,
    tokenization_column: str,
    request: dict[str, Any],
    size: int,
) -> int:
    """Tokens-mode equivalent of :func:`_count_concordance_hits`.

    Walks the first ``size`` rows of the tokenization column and counts
    exact-token matches of ``search_word``. Without this, the page-size
    estimator probes via the regex engine, which produces 0 hits for CJK
    queries (``\b``-style whole-word semantics don't apply) and pushes the
    estimator all the way to the largest candidate.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Tokens-mode equivalent of :func:`_count_concordance_hits`" behavior.
    """
    search_word = str(request.get("search_word") or "")
    if not search_word:
        return 0
    case_sensitive = bool(request.get("case_sensitive", False))
    try:
        slice_lf = base_lf.slice(0, size)
        slice_df = cast(pl.DataFrame, slice_lf.select(tokenization_column).collect())
    except Exception as exc:
        logger.debug("Tokens-mode hit probe failed at size=%d: %s", size, exc)
        return 0
    total = 0
    for tokens in slice_df.get_column(tokenization_column).to_list():
        if not isinstance(tokens, list) or not tokens:
            continue
        total += len(
            find_token_matches(tokens, search_word, case_sensitive=case_sensitive)
        )
    return total


def _resolve_page_size(
    base_lf: pl.LazyFrame,
    column: str,
    request: dict[str, Any],
    requested: Optional[int],
    *,
    tokenization_column: Optional[str] = None,
) -> int:
    """Return an effective page size, estimating when the client omitted one.

    For tokens-mode requests with tokenization metadata on the node, the
    probe walks the tokens column directly so CJK searches estimate against
    actual hit density instead of the regex engine's near-zero count.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Return an effective page size, estimating when the client omitted one" behavior.
    """
    if requested is not None and int(requested) > 0:
        return int(requested)
    use_tokens_probe = (
        str(request.get("search_mode") or "regex") == "tokens"
        and tokenization_column is not None
    )
    if use_tokens_probe:
        probe = partial(
            _count_tokens_concordance_hits,
            base_lf,
            column,
            tokenization_column,
            request,
        )
    else:
        probe = partial(_count_concordance_hits, base_lf, column, request)
    return estimate_page_size(probe, candidates=DEFAULT_PAGE_SIZE_CANDIDATES)


def _serialize_materialized_rows(
    df: pl.DataFrame,
    *,
    node_label: Optional[str] = None,
    document_column: str,
    include_document_column: bool = True,
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """Convert a materialised concordance slice into per-document groups.

    Consecutive rows that share the same document value are folded into one
    group so the dispersion view renders one horizontal bar per document with
    every hit marked along it. The materialise worker writes rows in document
    order, so a single linear ``groupby`` is enough; we don't re-sort.

    ``include_document_column`` controls only display. The materialized parquet
    must still contain ``document_column`` so grouping and dispersion binning
    have a stable source of truth.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need this unit's "Convert a materialised concordance slice into per-document groups" behavior.
    """
    if not document_column or document_column not in df.columns:
        raise InvalidInputError(
            "Materialized concordance rows must include the document column",
        )

    if df.height == 0:
        columns = [
            column
            for column in df.columns
            if include_document_column or column != document_column
        ]
        return [], columns

    columns = [
        column
        for column in df.columns
        if include_document_column or column != document_column
    ]
    grouped_rows: list[list[dict[str, Any]]] = []

    from itertools import groupby

    for _, group in groupby(df.to_dicts(), key=lambda r: r.get(document_column)):
        hits: list[dict[str, Any]] = []
        for row in group:
            hit = dict(row)
            if not include_document_column:
                hit.pop(document_column, None)
            if node_label:
                hit["__source_node"] = node_label
            hits.append(hit)
        if hits:
            grouped_rows.append(hits)

    if node_label and "__source_node" not in columns:
        columns.append("__source_node")
    return grouped_rows, columns


def compute_materialized_page(
    materialized_path: str,
    *,
    page: int,
    page_size: Optional[int],
    sort_by: Optional[str],
    descending: bool,
    node_label: Optional[str] = None,
    document_column: str,
    include_document_column: bool = True,
) -> dict[str, Any]:
    """Paginate a materialized concordance parquet as occurrence rows.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - backend API routes because they need this unit's "Paginate a materialized concordance parquet as occurrence rows" behavior.
    """
    effective_page_size = (
        int(page_size)
        if page_size is not None and int(page_size) > 0
        else DEFAULT_CONCORDANCE_PAGE_SIZE
    )
    lazy = pl.scan_parquet(materialized_path)
    total_rows = cast(pl.DataFrame, lazy.select(pl.len()).collect()).item() or 0

    effective_sort_by: Optional[str] = None
    if sort_by:
        try:
            schema = lazy.collect_schema()
            if sort_by in schema:
                lazy = lazy.sort(sort_by, descending=descending)
                effective_sort_by = sort_by
        except Exception as exc:
            logger.debug(
                "Ignoring unsupported sort_by '%s' for materialized page: %s",
                sort_by,
                exc,
            )

    start = max(page - 1, 0) * effective_page_size
    slice_df = cast(pl.DataFrame, lazy.slice(start, effective_page_size).collect())
    rows, columns = _serialize_materialized_rows(
        slice_df,
        node_label=node_label,
        document_column=document_column,
        include_document_column=include_document_column,
    )

    total_source_pages = (
        max(1, math.ceil(total_rows / effective_page_size)) if total_rows else 0
    )

    metadata = _column_metadata(columns, MATERIALIZED_CONCORDANCE_COLUMNS)

    return {
        "data": stringify_unsafe_integers(rows),
        "columns": columns,
        "metadata": metadata,
        "pagination": {
            "page": page,
            "page_size": effective_page_size,
            "total_source_rows": total_rows,
            "total_source_pages": total_source_pages,
            "result_count": len(rows),
            "has_next": page < total_source_pages,
            "has_prev": page > 1,
        },
        "sorting": {
            "sort_by": effective_sort_by,
            "descending": descending,
        },
        "materialized": True,
    }


DISPERSION_BIN_COUNT = 100


def read_dispersion_bins(
    materialized_path: str,
    document_column: Optional[str] = None,
) -> dict[str, Any]:
    """Pre-bin a materialised concordance parquet into 100 fixed buckets.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `concordance.concordance_task_dispersion_bins` because they need this unit's "Pre-bin a materialised concordance parquet into 100 fixed buckets" behavior.

    Why:
    - The dispersion summary plot needs counts of hits across relative positions
      in each document. By aggregating to 100 (1 %-wide) bins server-side we
      ship a payload of size `O(distinct_matched_texts × 100)` instead of one
      row per hit, which scales to materialised parquets with millions of rows.
      The frontend can then re-aggregate to any display bin count whose value
      divides 100 (4, 5, 10, 20, 25, 50, 100) without another round trip.
    """
    lf = pl.scan_parquet(materialized_path)
    schema = lf.collect_schema()

    has_matched_text = CONC_MATCHED_TEXT_COLUMN in schema
    has_start_idx = CONC_START_IDX_COLUMN in schema
    if not document_column or document_column not in schema:
        raise InvalidInputError(
            "Materialized concordance rows must include the document column",
        )
    doc_col = document_column

    if not has_matched_text or not has_start_idx:
        return {
            "total_hits": 0,
            "document_column": document_column,
            "bin_count": DISPERSION_BIN_COUNT,
            "rows": [],
        }

    doc_length_expr = (
        pl.col(doc_col).cast(pl.Utf8, strict=False).str.len_chars().cast(pl.Int64)
    )
    start_idx_expr = pl.col(CONC_START_IDX_COLUMN).cast(pl.Int64, strict=False)
    bin_idx_expr = (
        (
            start_idx_expr.cast(pl.Float64)
            / doc_length_expr.cast(pl.Float64)
            * DISPERSION_BIN_COUNT
        )
        .floor()
        .cast(pl.Int64)
        .clip(0, DISPERSION_BIN_COUNT - 1)
    )

    binned = (
        lf.filter(doc_length_expr > 0)
        .filter(start_idx_expr.is_not_null())
        .with_columns(bin_idx_expr.alias("bin_idx"))
        .group_by([CONC_MATCHED_TEXT_COLUMN, "bin_idx"])
        .len()
        .rename({"len": "count", CONC_MATCHED_TEXT_COLUMN: "matched_text"})
        .sort(["matched_text", "bin_idx"])
    )
    df = cast(pl.DataFrame, binned.collect())

    total_hits = int(df["count"].sum()) if df.height > 0 else 0
    return {
        "total_hits": total_hits,
        "document_column": doc_col,
        "bin_count": DISPERSION_BIN_COUNT,
        "rows": stringify_unsafe_integers(df.to_dicts()),
    }


def build_concordance_response(
    user_id: str,
    workspace_id: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Build the full concordance API response from a normalized request.

    Steps:
    - Normalize caller input into the representation this module expects.
    - Delegate stateful, expensive, or validating work to the owning manager/helper when needed.
    - Return the compact value the caller uses for artifacts, validation, or response shaping.

    Used by:
    - `concordance.run_concordance` because they need this unit's "Build the full concordance API response from a normalized request" behavior.
    - `concordance_task_result` because they need this unit's "Build the full concordance API response from a normalized request" behavior.
    - `concordance_task_result_post` because they need this unit's "Build the full concordance API response from a normalized request" behavior.

    Why:
    - Provides one shared response builder for run and retrieval endpoints,
      avoiding payload drift across routes.
    """
    page = int(request.get("page") or DEFAULT_CONCORDANCE_PAGE)
    raw_page_size = request.get("page_size")
    page_size: Optional[int] = (
        int(raw_page_size)
        if raw_page_size is not None and int(raw_page_size) > 0
        else None
    )
    sort_by = request.get("sort_by")
    descending = bool(request.get("descending", DEFAULT_CONCORDANCE_DESCENDING))
    materialized_paths_raw = request.get("materialized_paths") or {}
    materialized_paths: dict[str, str] = {
        str(node_id): str(path)
        for node_id, path in materialized_paths_raw.items()
        if isinstance(path, str) and path
    }

    node_ids = request.get("node_ids") or []

    node_sources, label_to_node_map, _node_labels, resolve_error = resolve_node_sources(
        user_id, workspace_id, request
    )
    if resolve_error is not None:
        raise NotFoundError(resolve_error)
    data: dict[str, Any] = {}

    # Pre-resolve page_size for non-materialized nodes so each per-node page stays
    # consistent. For materialized nodes we rely on the client's choice and default
    # to DEFAULT_CONCORDANCE_PAGE_SIZE inside compute_materialized_page.
    if page_size is None:
        estimates: list[int] = []
        for node_id in node_ids:
            if node_id in materialized_paths:
                continue
            src = node_sources.get(node_id)
            if not src:
                continue
            estimates.append(
                _resolve_page_size(
                    src["lf"],
                    src["column"],
                    request,
                    None,
                    tokenization_column=src.get("tokenization_column"),
                )
            )
        if estimates:
            page_size = max(estimates)

    # Each node is paged independently and returned under its own node_id key. The
    # combined comparison view is synthesized client-side from these per-node
    # slices, so the backend no longer interleaves rows. Per-node result queries
    # carry ``result_node_id`` so a page or sort change recomputes only the
    # targeted node; other nodes keep their independently-paged client data after
    # the frontend merges this partial response. ``combinable`` advertises whether
    # the frontend may offer the combined toggle and stays derived from the full
    # node set so scoping a single node never hides the combine action.
    result_node_id = request.get("result_node_id")
    scoped_node_ids = (
        [result_node_id]
        if result_node_id and result_node_id in node_ids
        else node_ids
    )
    for node_id in scoped_node_ids:
        src = node_sources.get(node_id)
        if not src:
            continue
        if node_id in materialized_paths:
            document_column = src["column"]
            selected_columns = request.get("selected_columns")
            include_document_column = (
                not isinstance(selected_columns, list)
                or document_column in selected_columns
            )
            data[node_id] = compute_materialized_page(
                materialized_paths[node_id],
                page=page,
                page_size=page_size,
                sort_by=sort_by,
                descending=descending,
                node_label=src.get("label"),
                document_column=document_column,
                include_document_column=include_document_column,
            )
            continue
        data[node_id] = compute_node_concordance_page(
            src,
            request,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            descending=descending,
        )
    combinable = len(node_ids) > 1

    analysis_params = dict(request)
    if label_to_node_map:
        analysis_params["label_to_node_map"] = label_to_node_map

    return {
        "state": "successful",
        "message": "Concordance analysis complete",
        "data": data,
        "analysis_params": analysis_params,
        "combinable": combinable,
    }
