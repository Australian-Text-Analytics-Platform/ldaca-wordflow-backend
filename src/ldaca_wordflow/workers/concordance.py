"""Process-worker implementations for concordance Analysis execution.

Used by:
- canonical Analysis execution and backend tests that exercise concordance
  computation from immutable inputs.

Flow: load text or token inputs, derive concordance rows or dispersion bins, persist
    Parquet Artifacts, and return the owning Analysis result payload.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

from ..analysis.concordance_core import build_concordance_search_pattern
from ..analysis.concordance_tokens import (
    build_token_hit,
    find_token_matches,
)
from ..analysis.generated_columns import (
    CONC_END_IDX_COLUMN,
    CONC_EXTRACTION_COLUMN,
    CONC_L1_COLUMN,
    CONC_L1_FREQ_COLUMN,
    CONC_LEFT_CONTEXT_COLUMN,
    CONC_MATCHED_TEXT_COLUMN,
    CONC_R1_COLUMN,
    CONC_R1_FREQ_COLUMN,
    CONC_RIGHT_CONTEXT_COLUMN,
    CONC_START_IDX_COLUMN,
    CORE_CONCORDANCE_COLUMNS,
    DETACHABLE_CONCORDANCE_COLUMNS,
    concordance_extraction_expr,
    concordance_struct_projection,
)
from ..domain.workspace import (
    ConcordanceAnalysisRequest,
    ConcordanceDetachmentDerivation,
    ConcordanceDispersionDetachmentDerivation,
    DerivationInput,
    DerivationProvenance,
    node_reference,
)
from .utils import process_entrypoint

# The dispersion-detach output reuses `CONC_extraction` as the column name
# for the per-document multi-line joined string. It carries the same KWIC
# windows as the per-hit `CONC_extraction` column, collapsed into one row per
# source document.
DISPERSION_EXTRACTED_CONTENTS_COLUMN = CONC_EXTRACTION_COLUMN

logger = logging.getLogger(__name__)


def _source_text_filter(document_column: str):
    """Return the non-empty document filter used by concordance workers.

    Called by:
    - root and child concordance Analysis workers using immutable LazyFrame-plan
      snapshots instead of pre-collected corpora.
    """

    import polars as pl

    return (
        pl.col(document_column)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.len_chars()
        .fill_null(0)
        > 0
    )


def _collect_source_input_from_snapshot(
    *,
    input_snapshot_dir: str,
    node_id: str,
    document_column: str,
    token_cache_path: str | None,
    extra_column_names: list[str] | None,
    include_all_metadata: bool = False,
    search_mode: str = "regex",
) -> tuple[list[str], dict[str, list] | None, dict[str, Any] | None, list[Any] | None]:
    """Collect concordance source inputs inside the worker process.

    Used by:
    - root and child concordance Analysis workers that receive immutable input
      snapshots.

    Flow:
    1. Load the snapshotted LazyFrame plan for ``node_id``.
    2. Optionally hydrate registered tokenization for tokens-mode searches.
    3. Select/filter the document and requested metadata columns.
    4. Materialize the aligned Python lists only inside the fresh child process.
    """

    import polars as pl

    from .input_snapshots import load_snapshot_node

    snapshot_node = load_snapshot_node(input_snapshot_dir, node_id)
    node = snapshot_node.to_node()
    node_data = snapshot_node.data
    tokenization_column: str | None = None
    if search_mode == "tokens":
        if token_cache_path is None:
            raise ValueError(
                "tokens-mode concordance requires an explicit token cache path"
            )
        tokenization_column = node.find_tokenization_column(document_column)
        if tokenization_column is None:
            raise ValueError(
                f"No tokens column registered on node {node_id!r} for source column {document_column!r}"
            )
        from ..analysis.token_cache import hydrate_tokenization_lazyframe

        node_data = hydrate_tokenization_lazyframe(
            node=node,
            source_column=document_column,
            cache_path=token_cache_path,
        )

    schema_names = list(node_data.collect_schema().names())
    if include_all_metadata:
        metadata_columns = [
            column
            for column in schema_names
            if column != document_column and column != tokenization_column
        ]
    else:
        metadata_columns = list(extra_column_names or [])

    select_exprs = [pl.col(document_column)] + [pl.col(c) for c in metadata_columns]
    if tokenization_column is not None:
        select_exprs.append(pl.col(tokenization_column))

    corpus_df = (
        node_data.select(select_exprs)
        .filter(_source_text_filter(document_column))
        .collect()
    )
    node_corpus = [
        str(value) if value is not None else ""
        for value in corpus_df.get_column(document_column).to_list()
    ]

    extra_columns_data: dict[str, list] | None = None
    extra_columns_dtypes: dict[str, Any] | None = None
    if metadata_columns:
        extra_columns_data = {}
        extra_columns_dtypes = {}
        for column in metadata_columns:
            series = corpus_df.get_column(column)
            extra_columns_data[column] = series.to_list()
            extra_columns_dtypes[column] = series.dtype

    node_tokens = (
        corpus_df.get_column(tokenization_column).to_list()
        if tokenization_column is not None
        else None
    )
    return node_corpus, extra_columns_data, extra_columns_dtypes, node_tokens


def _build_concordance_response_from_snapshot(
    *,
    input_snapshot_dir: str,
    token_cache_path: str | None,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the initial concordance result from Analysis-owned snapshots.

    Used by:
    - ``run_concordance_analysis`` because submission must only create the
      snapshot, persist the Analysis, and return.

    Flow: load snapshot nodes, build the same source descriptors used by
    route-side pagination helpers, compute each requested node page inside the
    worker, and return the result payload for Analysis persistence.
    """

    from ..analysis.concordance_core import (
        _resolve_page_size,
        compute_node_concordance_page,
    )
    from .input_snapshots import load_snapshot_node

    request = ConcordanceAnalysisRequest.model_validate(request_payload)
    canonical_payload = request.model_dump(mode="json", exclude={"kind"})
    node_ids = [str(node_id) for node_id in request.node_ids]
    node_columns = {
        str(node_id): column for node_id, column in request.node_columns.items()
    }

    node_sources: dict[str, dict[str, Any]] = {}
    for node_id in node_ids:
        column = node_columns[node_id]
        snapshot_node = load_snapshot_node(input_snapshot_dir, node_id)
        node = snapshot_node.to_node()
        node_label = snapshot_node.name or node_id
        tokenization_column = node.find_tokenization_column(column)
        node_sources[node_id] = {
            "lf": snapshot_node.data,
            "column": column,
            "label": node_label,
            "tokenization_column": tokenization_column,
            "node": node,
            "token_cache_path": token_cache_path,
        }

    estimates = [
        _resolve_page_size(
            node_sources[node_id]["lf"],
            node_sources[node_id]["column"],
            canonical_payload,
            None,
            tokenization_column=node_sources[node_id]["tokenization_column"],
        )
        for node_id in node_ids
    ]
    page_size = max(estimates)

    sources: list[dict[str, Any]] = []
    for node_id in node_ids:
        src = node_sources[node_id]
        sources.append(
            {
                "node_id": node_id,
                "node_name": src["label"],
                "result": compute_node_concordance_page(
                    src,
                    canonical_payload,
                    page=1,
                    page_size=page_size,
                    sort_by=None,
                    descending=False,
                ),
            }
        )

    return {
        "state": "successful",
        "message": "Concordance analysis complete",
        "sources": sources,
    }


@process_entrypoint
def run_concordance_analysis(
    user_id: str,
    workspace_id: str,
    input_snapshot_dir: str,
    request_payload: dict[str, Any],
    token_cache_path: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Execute the primary concordance analysis in a worker process.

    Used by:
    - canonical concordance Analysis execution because submission must not
      collect source rows or build first-page results on the event loop.

    Flow: load snapshotted node plans, compute the
    requested first page, and return the persisted Analysis result payload.
    """
    try:
        if progress_callback:
            progress_callback(0.1, "Loading concordance input...")
        result = _build_concordance_response_from_snapshot(
            input_snapshot_dir=input_snapshot_dir,
            token_cache_path=token_cache_path,
            request_payload=request_payload,
        )
        return result
    except Exception:
        logger.exception(
            "Concordance Analysis failed for user=%s workspace=%s",
            user_id,
            workspace_id,
        )
        raise


def _build_concordance_occurrence_dataframe(
    node_corpus: list[str],
    document_column: str,
    search_word: str,
    num_left_tokens: int,
    num_right_tokens: int,
    regex: bool,
    whole_word: bool,
    case_sensitive: bool,
    include_document_column: bool,
    extra_columns_data: dict[str, list] | None,
    extra_columns_dtypes: dict[str, Any] | None = None,
):
    """Compute flattened occurrence rows for one corpus. Returns (df, output_columns).

    Called by:
    - concordance detachment child Analyses that recompute a complete published
      Data Block from their immutable request and snapshot.

    Flow: normalize aligned input rows, derive concordance occurrences, and
        return the frame and exact output-column order.
    """
    import polars as pl
    import polars_text as pt

    corpus = [str(v) if v is not None else "" for v in node_corpus]
    non_empty_mask = [bool(v.strip()) for v in corpus]
    corpus = [v for v, keep in zip(corpus, non_empty_mask) if keep]

    source_column_name = "__concordance_source__"
    data: dict[str, list] = {source_column_name: corpus}
    base_columns: list[pl.Expr] = []
    output_columns: list[str] = []

    if include_document_column:
        data[document_column] = corpus
        base_columns.append(pl.col(document_column))
        output_columns.append(document_column)

    if extra_columns_data:
        for col_name, col_values in extra_columns_data.items():
            filtered = [v for v, keep in zip(col_values, non_empty_mask) if keep]
            data[col_name] = filtered
            base_columns.append(pl.col(col_name))
            output_columns.append(col_name)

    df = pl.DataFrame(data)
    if extra_columns_dtypes:
        cast_exprs = [
            pl.col(col).cast(dtype)
            for col, dtype in extra_columns_dtypes.items()
            if col in df.columns and df.schema[col] != dtype
        ]
        if cast_exprs:
            df = df.with_columns(cast_exprs)
    search_pattern, use_regex = build_concordance_search_pattern(
        search_word,
        regex=regex,
        whole_word=whole_word,
    )
    result = (
        df.select(
            [
                pl.col(source_column_name).alias("__concordance_doc__"),
                *base_columns,
                pt.concordance(
                    pl.col(source_column_name),
                    search_pattern,
                    num_left_tokens=num_left_tokens,
                    num_right_tokens=num_right_tokens,
                    regex=use_regex,
                    case_sensitive=case_sensitive,
                ).alias("concordance"),
            ]
        )
        .explode("concordance")
        .select(
            [
                pl.exclude("concordance"),
                *concordance_struct_projection("concordance"),
            ]
        )
        .filter(pl.col(CONC_MATCHED_TEXT_COLUMN).is_not_null())
        .with_columns(concordance_extraction_expr("__concordance_doc__"))
        .drop("__concordance_doc__")
    )
    return result, output_columns + list(CORE_CONCORDANCE_COLUMNS) + [
        CONC_EXTRACTION_COLUMN
    ]


def _build_tokens_concordance_occurrence_dataframe(
    node_corpus: list[str],
    node_tokens: list[Any],
    document_column: str,
    search_word: str,
    num_left_tokens: int,
    num_right_tokens: int,
    case_sensitive: bool,
    include_document_column: bool,
    extra_columns_data: dict[str, list] | None,
    extra_columns_dtypes: dict[str, Any] | None = None,
):
    """Tokens-mode parallel of :func:`_build_concordance_occurrence_dataframe`.

    Output column shape is identical to the regex-mode build so paginated
    reads, detach, and dispersion bin fetches don't have to branch on the
    parquet's origin. Walks ``node_tokens`` (the dynamically hydrated token
    column for the registered source/model) for exact token equality with
    ``search_word``, then reuses
    :func:`build_token_hit` to construct each row.

    Called by:
    - token-mode concordance detachment child Analyses.

    Flow: align token hits with source rows and return the same occurrence
        shape as regex-mode concordance.
    """
    import polars as pl

    corpus = [str(v) if v is not None else "" for v in node_corpus]
    tokens_per_row = list(node_tokens or [])
    if len(tokens_per_row) != len(corpus):
        raise ValueError(
            "node_tokens length must equal node_corpus length "
            f"(got {len(tokens_per_row)} vs {len(corpus)})"
        )
    # Mirror the regex builder's empty-row filter so the document index
    # stays aligned with extra columns.
    keep_mask = [bool(text.strip()) for text in corpus]
    corpus = [text for text, keep in zip(corpus, keep_mask) if keep]
    tokens_per_row = [tokens for tokens, keep in zip(tokens_per_row, keep_mask) if keep]

    filtered_extras: dict[str, list] = {}
    if extra_columns_data:
        for col_name, col_values in extra_columns_data.items():
            filtered_extras[col_name] = [
                v for v, keep in zip(col_values, keep_mask) if keep
            ]

    hits: list[dict[str, Any]] = []
    for row_index, (raw_text, tokens) in enumerate(zip(corpus, tokens_per_row)):
        if not isinstance(tokens, list) or not tokens:
            continue
        # ``tokens`` may include None entries (polars struct nulls). The
        # helpers below tolerate that, so no extra filtering needed here.
        token_list = cast(list[Any], tokens)
        match_indices = find_token_matches(
            token_list, search_word, case_sensitive=case_sensitive
        )
        for match_index in match_indices:
            hit = build_token_hit(
                cast(list[dict[str, Any]], token_list),
                match_index,
                raw_text=raw_text,
                num_left=num_left_tokens,
                num_right=num_right_tokens,
            )
            full: dict[str, Any] = dict(hit)
            if include_document_column:
                full[document_column] = raw_text
            for col_name, values in filtered_extras.items():
                full[col_name] = values[row_index]
            hits.append(full)

    # Build the output columns list in the same order the regex builder
    # uses: [document_column?, *extras, *CORE_CONCORDANCE_COLUMNS,
    # CONC_extraction]. The DataFrame constructor will follow this order
    # because we pass dicts; force the column order explicitly via select
    # at the end so downstream consumers see byte-identical schema.
    output_columns: list[str] = []
    if include_document_column:
        output_columns.append(document_column)
    output_columns.extend(filtered_extras.keys())
    output_columns.extend(CORE_CONCORDANCE_COLUMNS)
    output_columns.append(CONC_EXTRACTION_COLUMN)

    if not hits:
        # Build an empty DataFrame with the right schema so the downstream
        # group_by joins don't error on an empty input.
        schema: dict[str, Any] = {}
        if include_document_column:
            schema[document_column] = pl.Utf8
        if extra_columns_dtypes:
            for col_name in filtered_extras:
                schema[col_name] = extra_columns_dtypes.get(col_name, pl.Utf8)
        else:
            for col_name in filtered_extras:
                schema[col_name] = pl.Utf8
        schema[CONC_LEFT_CONTEXT_COLUMN] = pl.Utf8
        schema[CONC_MATCHED_TEXT_COLUMN] = pl.Utf8
        schema[CONC_RIGHT_CONTEXT_COLUMN] = pl.Utf8
        schema[CONC_START_IDX_COLUMN] = pl.Int64
        schema[CONC_END_IDX_COLUMN] = pl.Int64
        schema[CONC_L1_COLUMN] = pl.Utf8
        schema[CONC_R1_COLUMN] = pl.Utf8
        schema[CONC_EXTRACTION_COLUMN] = pl.Utf8
        return pl.DataFrame(schema=schema), output_columns

    df = pl.DataFrame(hits)
    # Cast extras to the source dtypes if provided, mirroring the regex
    # builder's behaviour. CONC_* numeric columns come out as Int64 from
    # the build_token_hit dicts, which matches the regex side.
    if extra_columns_dtypes:
        cast_exprs = [
            pl.col(col).cast(dtype)
            for col, dtype in extra_columns_dtypes.items()
            if col in df.columns and df.schema[col] != dtype
        ]
        if cast_exprs:
            df = df.with_columns(cast_exprs)
    df = df.select(output_columns)
    return df, output_columns


@process_entrypoint
def run_concordance_detachment(
    workspace_dir: str,
    input_snapshot_dir: str,
    parent_node_id: str,
    document_column: str,
    search_word: str,
    num_left_tokens: int,
    num_right_tokens: int,
    regex: bool,
    whole_word: bool,
    case_sensitive: bool,
    new_node_name: str,
    include_document_column: bool = False,
    include_extraction: bool = False,
    selected_generated_columns: list[str] | None = None,
    extra_column_names: list[str] | None = None,
    search_mode: str = "regex",
    token_cache_path: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Recompute a complete concordance selection as one private Data Block."""
    try:
        if progress_callback:
            progress_callback(0.02, "Loading concordance libraries...")

        import os

        from ..infrastructure.storage.node_store import write_detached_frame

        logger.info("[Worker %d] Starting concordance detachment", os.getpid())

        if progress_callback:
            progress_callback(0.2, "Preparing text data...")

        (
            node_corpus,
            extra_columns_data,
            extra_columns_dtypes,
            node_tokens,
        ) = _collect_source_input_from_snapshot(
            input_snapshot_dir=input_snapshot_dir,
            node_id=parent_node_id,
            document_column=document_column,
            token_cache_path=token_cache_path,
            extra_column_names=extra_column_names,
            search_mode=search_mode,
        )

        if progress_callback:
            progress_callback(0.55, "Generating concordance matches...")

        if search_mode == "tokens":
            if node_tokens is None:
                raise ValueError("Token-mode concordance input is unavailable")
            result, output_columns = _build_tokens_concordance_occurrence_dataframe(
                node_corpus=node_corpus,
                node_tokens=node_tokens,
                document_column=document_column,
                search_word=search_word,
                num_left_tokens=num_left_tokens,
                num_right_tokens=num_right_tokens,
                case_sensitive=case_sensitive,
                include_document_column=include_document_column,
                extra_columns_data=extra_columns_data,
                extra_columns_dtypes=extra_columns_dtypes,
            )
        else:
            result, output_columns = _build_concordance_occurrence_dataframe(
                node_corpus=node_corpus,
                document_column=document_column,
                search_word=search_word,
                num_left_tokens=num_left_tokens,
                num_right_tokens=num_right_tokens,
                regex=regex,
                whole_word=whole_word,
                case_sensitive=case_sensitive,
                include_document_column=include_document_column,
                extra_columns_data=extra_columns_data,
                extra_columns_dtypes=extra_columns_dtypes,
            )

        # Decide which generated columns the user wants. Deselected generated
        # columns are dropped and, for frequency columns, never even computed.
        all_generated = set(DETACHABLE_CONCORDANCE_COLUMNS)
        wanted_generated = set(selected_generated_columns or [])
        need_freq = (
            CONC_L1_FREQ_COLUMN in wanted_generated
            or CONC_R1_FREQ_COLUMN in wanted_generated
        )

        # Compute frequency columns only when the user kept at least one of
        # them, skipping the group-by/join when neither was selected.
        if need_freq:
            l1_freq = (
                result.group_by(CONC_L1_COLUMN)
                .len()
                .rename({"len": CONC_L1_FREQ_COLUMN})
            )
            r1_freq = (
                result.group_by(CONC_R1_COLUMN)
                .len()
                .rename({"len": CONC_R1_FREQ_COLUMN})
            )
            result = result.join(l1_freq, on=CONC_L1_COLUMN, how="left").join(
                r1_freq, on=CONC_R1_COLUMN, how="left"
            )
            output_columns = output_columns + [CONC_L1_FREQ_COLUMN, CONC_R1_FREQ_COLUMN]

        # Final projection honoring the user's column choice. Generated columns
        # (core + freq) are kept only when ticked; CONC_extraction stays opt-in;
        # the document column and metadata columns pass through. Order follows
        # the columns produced above so the detached node keeps a stable shape.
        keep_columns: list[str] = []
        for col in output_columns:
            if col in all_generated:
                if col in wanted_generated:
                    keep_columns.append(col)
            elif col == CONC_EXTRACTION_COLUMN:
                if include_extraction:
                    keep_columns.append(col)
            else:
                keep_columns.append(col)
        if keep_columns:
            result = result.select(keep_columns)
            output_columns = keep_columns

        if progress_callback:
            progress_callback(0.82, "Serializing detached data block...")

        node_payload = write_detached_frame(
            result,
            base_dir=workspace_dir,
            name=new_node_name,
            provenance=DerivationProvenance(
                operation=ConcordanceDetachmentDerivation(),
                inputs=[
                    DerivationInput(
                        role="source",
                        value=node_reference(parent_node_id),
                    )
                ],
            ),
            document=document_column,
        )

        if progress_callback:
            progress_callback(0.95, "Publishing concordance Data Block...")

        logger.info(
            "[Worker %d] Concordance detachment completed successfully", os.getpid()
        )

        return {
            "state": "successful",
            "result": {
                **node_payload,
                "output_columns": output_columns,
                "record_count": len(result),
            },
            "message": "Concordance detach completed successfully",
        }
    except Exception:
        logger.exception("Concordance detachment failed")
        raise


def _aggregate_hits_per_document(
    hits_df,
    document_column: str,
    selected_bins,
    total_bins,
    include_document_column: bool = True,
    extra_metadata_columns: list[str] | None = None,
    selected_matched_texts: list[str] | None = None,
    match_case_insensitive: bool = False,
):
    """Group per-hit rows by source document and aggregate into list columns.

    Used by concordance dispersion detachment to produce the per-document output shape:
    one row per document, hits collected into `List<T>` columns plus a
    `CONC_extraction` string column rendering each hit's character slice as an
    asterisk-bulleted multi-line paragraph in document-flow order. The same
    `CONC_extraction` name is used for both per-hit rows and this per-document
    aggregate output: different aggregation level, same semantic.

    When `selected_bins` is provided, hits are filtered to those whose bin
    index (`start_idx / doc_length * total_bins`, floored) is in the selected
    set — the "in-range hits only" semantic from the dispersion chart.

    When `selected_matched_texts` is provided, hits are filtered to those
    whose ``CONC_matched_text`` is in the set — the "legend filter" semantic
    from the chart. Set ``match_case_insensitive=True`` to lowercase both
    sides before comparison (mirrors the chart's ``lowercaseMatches`` toggle).

    Called by:
    - concordance dispersion child Analyses.

    Flow: filter selected hits, group them by source document, and return the
        child Data Block frame and exact output-column order.
    """
    import polars as pl

    df = hits_df
    # The hits frame must carry start/end indices; CONC_left/right_context are
    # dropped per the dispersion-detach spec.
    if CONC_START_IDX_COLUMN not in df.columns or CONC_END_IDX_COLUMN not in df.columns:
        raise ValueError(
            "Per-document aggregation requires CONC_start_idx and CONC_end_idx columns"
        )

    if selected_matched_texts is not None:
        if not selected_matched_texts:
            # All legend entries hidden → empty result. Filter to no rows so
            # downstream aggregation produces a zero-row dataframe with the
            # right schema.
            df = df.filter(pl.lit(False))
        elif CONC_MATCHED_TEXT_COLUMN in df.columns:
            if match_case_insensitive:
                allowed = [str(t).lower() for t in selected_matched_texts]
                df = df.filter(
                    pl.col(CONC_MATCHED_TEXT_COLUMN)
                    .cast(pl.Utf8, strict=False)
                    .str.to_lowercase()
                    .is_in(allowed)
                )
            else:
                allowed = [str(t) for t in selected_matched_texts]
                df = df.filter(
                    pl.col(CONC_MATCHED_TEXT_COLUMN)
                    .cast(pl.Utf8, strict=False)
                    .is_in(allowed)
                )

    df = df.with_columns(
        pl.col(document_column)
        .cast(pl.Utf8, strict=False)
        .str.len_chars()
        .alias("__doc_len__"),
    )

    if selected_bins is not None and total_bins:
        allowed = sorted({int(b) for b in selected_bins})
        bin_idx = (
            pl.when(pl.col("__doc_len__") > 0)
            .then(
                (
                    pl.col(CONC_START_IDX_COLUMN).cast(pl.Float64)
                    / pl.col("__doc_len__").cast(pl.Float64)
                    * float(total_bins)
                )
                .floor()
                .clip(0, total_bins - 1)
                .cast(pl.Int64)
            )
            .otherwise(pl.lit(None, dtype=pl.Int64))
        )
        df = df.with_columns(bin_idx.alias("__bin_idx__"))
        df = df.filter(pl.col("__bin_idx__").is_in(allowed))

    # Ensure CONC_extraction is present. Newly-generated hits already carry
    # it (added in `_build_concordance_occurrence_dataframe`), but older
    # materialised parquets pre-dating that change need it computed lazily.
    if CONC_EXTRACTION_COLUMN not in df.columns:
        df = df.with_columns(concordance_extraction_expr(document_column))

    # Sort by document then by hit start so list aggregates land in
    # document-flow order, then group with `maintain_order=True` so the
    # outer row order stays stable too.
    df = df.sort([document_column, CONC_START_IDX_COLUMN])

    available_metadata = [
        c
        for c in (extra_metadata_columns or [])
        if c in df.columns and c != document_column
    ]

    agg_columns = [
        pl.col(CONC_EXTRACTION_COLUMN).alias("__extracts_list__"),
        pl.col(CONC_MATCHED_TEXT_COLUMN).alias(CONC_MATCHED_TEXT_COLUMN),
        pl.col(CONC_L1_COLUMN).alias(CONC_L1_COLUMN),
        pl.col(CONC_R1_COLUMN).alias(CONC_R1_COLUMN),
    ]
    if CONC_L1_FREQ_COLUMN in df.columns:
        agg_columns.append(pl.col(CONC_L1_FREQ_COLUMN).alias(CONC_L1_FREQ_COLUMN))
    if CONC_R1_FREQ_COLUMN in df.columns:
        agg_columns.append(pl.col(CONC_R1_FREQ_COLUMN).alias(CONC_R1_FREQ_COLUMN))
    # Source metadata is identical for every hit in a document, so take the
    # first value of each requested column. This yields one value per row in
    # the per-document output rather than a list aggregation.
    for col in available_metadata:
        agg_columns.append(pl.col(col).first().alias(col))

    grouped = df.group_by(document_column, maintain_order=True).agg(agg_columns)

    # Use polars-native list manipulation to prefix each extract with "- "
    # (Markdown bullet syntax) and join with newlines. The earlier
    # `map_elements` form passed each row value as a Series, which broke
    # the `items or []` truthiness check.
    # Embedded newlines inside an individual extract (the raw document
    # slice can span CR/LF) would otherwise split the bullet across
    # multiple lines and render unprefixed continuation lines — collapse
    # any internal whitespace runs to a single space per element first.
    grouped = grouped.with_columns(
        pl.col("__extracts_list__")
        .list.eval(
            pl.lit("- ") + pl.element().str.replace_all(r"\s+", " ").str.strip_chars()
        )
        .list.join("\n")
        .alias(DISPERSION_EXTRACTED_CONTENTS_COLUMN)
    ).drop("__extracts_list__")

    output_columns: list[str] = []
    if include_document_column:
        output_columns.append(document_column)
    output_columns.extend(
        [
            DISPERSION_EXTRACTED_CONTENTS_COLUMN,
            CONC_MATCHED_TEXT_COLUMN,
            CONC_L1_COLUMN,
            CONC_R1_COLUMN,
        ]
    )
    if CONC_L1_FREQ_COLUMN in grouped.columns:
        output_columns.append(CONC_L1_FREQ_COLUMN)
    if CONC_R1_FREQ_COLUMN in grouped.columns:
        output_columns.append(CONC_R1_FREQ_COLUMN)
    output_columns.extend(available_metadata)

    return grouped.select(output_columns), output_columns


@process_entrypoint
def run_concordance_dispersion_detachment(
    workspace_dir: str,
    input_snapshot_dir: str,
    parent_node_id: str,
    document_column: str,
    search_word: str,
    num_left_tokens: int,
    num_right_tokens: int,
    regex: bool,
    whole_word: bool,
    case_sensitive: bool,
    new_node_name: str,
    include_document_column: bool = True,
    selected_bins: list[int] | None = None,
    total_bins: int | None = None,
    selected_matched_texts: list[str] | None = None,
    match_case_insensitive: bool = False,
    extra_column_names: list[str] | None = None,
    search_mode: str = "regex",
    token_cache_path: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Recompute, aggregate, and publish one private dispersion Data Block."""
    try:
        if progress_callback:
            progress_callback(0.02, "Loading concordance libraries...")

        import os

        from ..infrastructure.storage.node_store import write_detached_frame

        logger.info(
            "[Worker %d] Starting concordance dispersion detachment", os.getpid()
        )

        metadata_column_names = list(extra_column_names or [])

        if progress_callback:
            progress_callback(0.25, "Generating concordance matches...")
        (
            node_corpus,
            extra_columns_data,
            extra_columns_dtypes,
            node_tokens,
        ) = _collect_source_input_from_snapshot(
            input_snapshot_dir=input_snapshot_dir,
            node_id=parent_node_id,
            document_column=document_column,
            token_cache_path=token_cache_path,
            extra_column_names=extra_column_names,
            search_mode=search_mode,
        )
        if search_mode == "tokens":
            if node_tokens is None:
                raise ValueError("Token-mode concordance input is unavailable")
            hits_df, _ = _build_tokens_concordance_occurrence_dataframe(
                node_corpus=node_corpus,
                node_tokens=node_tokens,
                document_column=document_column,
                search_word=search_word,
                num_left_tokens=num_left_tokens,
                num_right_tokens=num_right_tokens,
                case_sensitive=case_sensitive,
                include_document_column=True,
                extra_columns_data=extra_columns_data,
                extra_columns_dtypes=extra_columns_dtypes,
            )
        else:
            hits_df, _ = _build_concordance_occurrence_dataframe(
                node_corpus=node_corpus,
                document_column=document_column,
                search_word=search_word,
                num_left_tokens=num_left_tokens,
                num_right_tokens=num_right_tokens,
                regex=regex,
                whole_word=whole_word,
                case_sensitive=case_sensitive,
                # Aggregation groups hits by source document even when the
                # published child Data Block omits that column.
                include_document_column=True,
                extra_columns_data=extra_columns_data,
                extra_columns_dtypes=extra_columns_dtypes,
            )
        if (
            CONC_L1_COLUMN in hits_df.columns
            and CONC_L1_FREQ_COLUMN not in hits_df.columns
        ):
            l1_freq = (
                hits_df.group_by(CONC_L1_COLUMN)
                .len()
                .rename({"len": CONC_L1_FREQ_COLUMN})
            )
            hits_df = hits_df.join(l1_freq, on=CONC_L1_COLUMN, how="left")
        if (
            CONC_R1_COLUMN in hits_df.columns
            and CONC_R1_FREQ_COLUMN not in hits_df.columns
        ):
            r1_freq = (
                hits_df.group_by(CONC_R1_COLUMN)
                .len()
                .rename({"len": CONC_R1_FREQ_COLUMN})
            )
            hits_df = hits_df.join(r1_freq, on=CONC_R1_COLUMN, how="left")

        if progress_callback:
            progress_callback(0.65, "Aggregating hits per document...")

        aggregated, output_columns = _aggregate_hits_per_document(
            hits_df,
            document_column=document_column,
            selected_bins=selected_bins,
            total_bins=total_bins,
            include_document_column=include_document_column,
            extra_metadata_columns=metadata_column_names,
            selected_matched_texts=selected_matched_texts,
            match_case_insensitive=match_case_insensitive,
        )

        if progress_callback:
            progress_callback(0.85, "Serializing detached data block...")

        node_payload = write_detached_frame(
            aggregated,
            base_dir=workspace_dir,
            name=new_node_name,
            provenance=DerivationProvenance(
                operation=ConcordanceDispersionDetachmentDerivation(),
                inputs=[
                    DerivationInput(
                        role="source",
                        value=node_reference(parent_node_id),
                    )
                ],
            ),
            document=document_column,
        )

        if progress_callback:
            progress_callback(0.95, "Publishing dispersion Data Block...")

        return {
            "state": "successful",
            "result": {
                **node_payload,
                "output_columns": output_columns,
                "record_count": int(len(aggregated)),
            },
            "message": "Concordance dispersion detach completed successfully",
        }
    except Exception:
        logger.exception("Concordance dispersion detachment failed")
        raise
