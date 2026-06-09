"""Worker implementations for concordance background tasks.

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: load text or token inputs, derive concordance rows or dispersion bins, persist
    parquet artifacts, and return metadata that the manager can serialize.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

from ..api.workspaces.analyses.concordance_core import build_concordance_search_pattern
from ..api.workspaces.analyses.concordance_tokens_mode import (
    build_token_hit,
    find_token_matches,
)
from ..api.workspaces.analyses.generated_columns import (
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
    MATERIALIZED_CONCORDANCE_COLUMNS,
    concordance_extraction_expr,
    concordance_struct_projection,
)
from .analysis_cache import materialized_cache_path
from .worker_utils import worker_task

# The dispersion-detach output reuses `CONC_extraction` as the column name
# for the per-document multi-line joined string. It carries the same KWIC
# windows as the per-hit `CONC_extraction` column on the materialised
# parquet, just collapsed into one row per source document.
DISPERSION_EXTRACTED_CONTENTS_COLUMN = CONC_EXTRACTION_COLUMN

logger = logging.getLogger(__name__)


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
    - Local helpers, route handlers, or service methods in this module because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: load text or token inputs, derive concordance rows or dispersion bins, persist
        parquet artifacts, and return metadata that the manager can serialize.
    """
    import polars as pl
    import polars_text as pt

    corpus = [str(v) if v is not None else "" for v in (node_corpus or [])]
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
    - Local helpers, route handlers, or service methods in this module because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: load text or token inputs, derive concordance rows or dispersion bins, persist
        parquet artifacts, and return metadata that the manager can serialize.
    """
    import polars as pl

    corpus = [str(v) if v is not None else "" for v in (node_corpus or [])]
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


@worker_task
def run_concordance_detach_task(
    configure_worker_environment,
    workspace_dir: str,
    node_corpus: list[str],
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
    extra_columns_data: dict[str, list] | None = None,
    extra_columns_dtypes: dict[str, Any] | None = None,
    materialized_path: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run concordance detach and return a serialized detached node payload.

    Fast path: when `materialized_path` points to an existing parquet (previously
    written by `run_concordance_materialize_task`), skip extraction and wrap the
    parquet as the detached node. Otherwise perform full concordance extraction.

    Used by:
    - backend tests, core workspace and worker services because tests need the same
      observable contract that production routes and workers rely on.

    Flow: load text or token inputs, derive concordance rows or dispersion bins, persist
        parquet artifacts, and return metadata that the manager can serialize.
    """
    configure_worker_environment()

    try:
        if progress_callback:
            progress_callback(0.02, "Loading concordance libraries...")

        import os

        import polars as pl

        from docworkspace import Node

        logger.info("[Worker %d] Starting concordance detach task", os.getpid())

        if materialized_path and os.path.exists(materialized_path):
            if progress_callback:
                progress_callback(0.4, "Reusing materialized occurrences...")
            import uuid

            # Read the materialized parquet and select only the columns the
            # user requested.  The materialized parquet contains all source
            # metadata columns so the detach fast path can respect the user's
            # column selection.
            mat_df = pl.read_parquet(materialized_path)

            # Determine which columns to keep.
            keep_cols: list[str] = []
            if include_document_column and document_column in mat_df.columns:
                keep_cols.append(document_column)
            if extra_columns_data:
                for col_name in extra_columns_data:
                    if col_name in mat_df.columns and col_name not in keep_cols:
                        keep_cols.append(col_name)
            # Always keep CORE_CONCORDANCE_COLUMNS + freq columns when the
            # caller didn't send an explicit selection (backwards-compat).
            # Otherwise honor exactly the generated columns the user ticked —
            # deselected generated columns are dropped from the output.
            all_generated = set(MATERIALIZED_CONCORDANCE_COLUMNS)
            wanted_generated = (
                all_generated
                if selected_generated_columns is None
                else set(selected_generated_columns)
            )
            for col in mat_df.columns:
                if col in keep_cols:
                    continue
                if col in wanted_generated:
                    keep_cols.append(col)
                elif col == CONC_EXTRACTION_COLUMN and include_extraction:
                    keep_cols.append(col)
            mat_df = mat_df.select(keep_cols) if keep_cols else mat_df

            detach_data_dir = os.path.join(workspace_dir, "data")
            os.makedirs(detach_data_dir, exist_ok=True)
            detach_parquet_path = os.path.join(
                detach_data_dir,
                f"concordance_detach_{uuid.uuid4().hex}.parquet",
            )
            mat_df.write_parquet(detach_parquet_path)
            lazy = pl.scan_parquet(detach_parquet_path)
            schema_names = list(lazy.collect_schema().names())
            record_count = len(mat_df)
            output_columns = schema_names
            detached_node = Node(
                data=lazy,
                name=new_node_name,
                workspace=None,
                operation="concordance_detach",
                parents=[parent_node_id],
                document=document_column,
            )
            node_payload = detached_node.to_dict(base_dir=workspace_dir)
            if progress_callback:
                progress_callback(1.0, "Concordance detach completed")
            return {
                "state": "successful",
                "result": {
                    "node_payload": node_payload,
                    "output_columns": output_columns,
                    "record_count": record_count,
                },
                "message": "Concordance detach completed successfully",
            }

        if progress_callback:
            progress_callback(0.2, "Preparing text data...")

        if progress_callback:
            progress_callback(0.55, "Generating concordance matches...")

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

        # Decide which generated columns the user wants. None == backwards-
        # compatible "keep all core + freq"; otherwise honor the explicit
        # selection so deselected generated columns are dropped (and, for the
        # frequency columns, never even computed).
        all_generated = set(MATERIALIZED_CONCORDANCE_COLUMNS)
        wanted_generated = (
            all_generated
            if selected_generated_columns is None
            else set(selected_generated_columns)
        )
        need_freq = (
            CONC_L1_FREQ_COLUMN in wanted_generated
            or CONC_R1_FREQ_COLUMN in wanted_generated
        )

        # Compute frequency columns (same as materialize) only when the user
        # kept at least one of them — skipping the group-by/join work entirely
        # when neither frequency column was selected.
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

        detached_node = Node(
            data=result.lazy(),
            name=new_node_name,
            workspace=None,
            operation="concordance_detach",
            parents=[parent_node_id],
            document=document_column,
        )
        node_payload = detached_node.to_dict(base_dir=workspace_dir)

        if progress_callback:
            progress_callback(1.0, "Concordance detach completed")

        logger.info(
            "[Worker %d] Concordance detach task completed successfully", os.getpid()
        )

        return {
            "state": "successful",
            "result": {
                "node_payload": node_payload,
                "output_columns": output_columns,
                "record_count": len(result),
            },
            "message": "Concordance detach completed successfully",
        }
    except Exception as e:
        return {
            "state": "failed",
            "error": str(e),
            "message": f"Concordance detach task failed: {str(e)}",
        }


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

    Used by the dispersion-detach task to produce the per-document output shape:
    one row per document, hits collected into `List<T>` columns plus a
    `CONC_extraction` string column rendering each hit's character slice as an
    asterisk-bulleted multi-line paragraph in document-flow order. The same
    `CONC_extraction` name is used per-hit on the materialised parquet and
    per-document on this aggregate output — different aggregation level, same
    semantic.

    When `selected_bins` is provided, hits are filtered to those whose bin
    index (`start_idx / doc_length * total_bins`, floored) is in the selected
    set — the "in-range hits only" semantic from the dispersion chart.

    When `selected_matched_texts` is provided, hits are filtered to those
    whose ``CONC_matched_text`` is in the set — the "legend filter" semantic
    from the chart. Set ``match_case_insensitive=True`` to lowercase both
    sides before comparison (mirrors the chart's ``lowercaseMatches`` toggle).

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: load text or token inputs, derive concordance rows or dispersion bins, persist
        parquet artifacts, and return metadata that the manager can serialize.
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


@worker_task
def run_concordance_dispersion_detach_task(
    configure_worker_environment,
    workspace_dir: str,
    node_corpus: list[str],
    parent_node_id: str,
    document_column: str,
    search_word: str,
    num_left_tokens: int,
    num_right_tokens: int,
    regex: bool,
    whole_word: bool,
    case_sensitive: bool,
    new_node_name: str,
    child_task_id: str | None = None,
    parent_task_id: str | None = None,
    include_document_column: bool = True,
    extra_columns_data: dict[str, list] | None = None,
    extra_columns_dtypes: dict[str, Any] | None = None,
    materialized_path: str | None = None,
    selected_bins: list[int] | None = None,
    total_bins: int | None = None,
    selected_matched_texts: list[str] | None = None,
    match_case_insensitive: bool = False,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Aggregate concordance hits per document and detach as a workspace node.

    Fast path: reads the previously-materialised flat parquet when
    `materialized_path` is provided; otherwise computes hits from the full
    source corpus. Always drops left/right context and start/end indices —
    callers consume `CONC_extraction` plus the List<T> aggregates.

    Used by:
    - backend tests, core workspace and worker services because tests need the same
      observable contract that production routes and workers rely on.

    Flow: load text or token inputs, derive concordance rows or dispersion bins, persist
        parquet artifacts, and return metadata that the manager can serialize.
    """
    configure_worker_environment()

    try:
        if progress_callback:
            progress_callback(0.02, "Loading concordance libraries...")

        import os
        import uuid

        import polars as pl

        from docworkspace import Node

        logger.info(
            "[Worker %d] Starting concordance dispersion detach task", os.getpid()
        )

        # Track the materialised parquet path so we can publish an
        # `analysis_materialized` event back to the dispatcher after the
        # slow path runs — the user otherwise has to click "Process All"
        # separately even though we already did the full-corpus work.
        side_effect_materialized_path: str | None = None
        side_effect_summary: dict[str, Any] | None = None

        if materialized_path and os.path.exists(materialized_path):
            if progress_callback:
                progress_callback(0.4, "Reusing materialized occurrences...")
            hits_df = pl.read_parquet(materialized_path)
        else:
            if progress_callback:
                progress_callback(0.25, "Generating concordance matches...")
            hits_df, _ = _build_concordance_occurrence_dataframe(
                node_corpus=node_corpus,
                document_column=document_column,
                search_word=search_word,
                num_left_tokens=num_left_tokens,
                num_right_tokens=num_right_tokens,
                regex=regex,
                whole_word=whole_word,
                case_sensitive=case_sensitive,
                # Always materialise with the document column so the parquet
                # is reusable from the fast path next time, regardless of
                # whether this detach call includes it in its output.
                include_document_column=True,
                extra_columns_data=extra_columns_data,
                extra_columns_dtypes=extra_columns_dtypes,
            )
            # Match the materialize path: always include CONC_l1_freq /
            # CONC_r1_freq so the aggregated list aggregates have something
            # numeric to collect.
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

            # Persist the fully-computed hits as the materialised parquet so
            # the dispersion view can switch the "page above / whole data
            # block" dropdown to whole-data-block automatically, and so
            # subsequent bin-filtered detaches reuse the parquet via the fast
            # path. Only do this when we know the parent task id (otherwise
            # the dispatcher can't route the materialised event back to a
            # specific analysis task). The file itself is owned by the worker
            # task that produced it.
            if parent_task_id and child_task_id:
                cache_path = materialized_cache_path(
                    workspace_dir,
                    feature="concordance",
                    task_id=child_task_id,
                    node_id=parent_node_id,
                )
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                side_effect_materialized_path = str(cache_path)
                hits_df.write_parquet(side_effect_materialized_path)
                side_effect_summary = {
                    "record_count": int(len(hits_df)),
                    "unique_documents_with_hits": (
                        int(hits_df.select(pl.col(document_column).n_unique()).item())
                        if document_column in hits_df.columns
                        else 0
                    ),
                    "total_source_documents": len(node_corpus),
                }

        if progress_callback:
            progress_callback(0.65, "Aggregating hits per document...")

        extra_metadata_columns = (
            list(extra_columns_data.keys()) if extra_columns_data else []
        )
        aggregated, output_columns = _aggregate_hits_per_document(
            hits_df,
            document_column=document_column,
            selected_bins=selected_bins,
            total_bins=total_bins,
            include_document_column=include_document_column,
            extra_metadata_columns=extra_metadata_columns,
            selected_matched_texts=selected_matched_texts,
            match_case_insensitive=match_case_insensitive,
        )

        if progress_callback:
            progress_callback(0.85, "Serializing detached data block...")

        detach_data_dir = os.path.join(workspace_dir, "data")
        os.makedirs(detach_data_dir, exist_ok=True)
        detach_parquet_path = os.path.join(
            detach_data_dir,
            f"concordance_dispersion_detach_{uuid.uuid4().hex}.parquet",
        )
        aggregated.write_parquet(detach_parquet_path)

        lazy = pl.scan_parquet(detach_parquet_path)
        detached_node = Node(
            data=lazy,
            name=new_node_name,
            workspace=None,
            operation="concordance_dispersion_detach",
            parents=[parent_node_id],
            document=document_column,
        )
        node_payload = detached_node.to_dict(base_dir=workspace_dir)

        if progress_callback:
            progress_callback(1.0, "Concordance dispersion detach completed")

        result_payload: dict[str, Any] = {
            "node_payload": node_payload,
            "output_columns": output_columns,
            "record_count": int(len(aggregated)),
        }
        # Side-effect: if the slow path produced a materialised parquet,
        # include the info so the dispatcher can publish an
        # `analysis_materialized` event back to the frontend.
        if side_effect_materialized_path and parent_task_id:
            result_payload["materialized_path"] = side_effect_materialized_path
            result_payload["parent_task_id"] = parent_task_id
            result_payload["parent_node_id"] = parent_node_id
            if side_effect_summary:
                result_payload.update(side_effect_summary)

        return {
            "state": "successful",
            "result": result_payload,
            "message": "Concordance dispersion detach completed successfully",
        }
    except Exception as e:
        return {
            "state": "failed",
            "error": str(e),
            "message": f"Concordance dispersion detach task failed: {str(e)}",
        }


@worker_task
def run_concordance_materialize_task(
    configure_worker_environment,
    workspace_dir: str,
    node_corpus: list[str],
    child_task_id: str,
    parent_task_id: str,
    parent_node_id: str,
    document_column: str,
    search_word: str,
    num_left_tokens: int,
    num_right_tokens: int,
    regex: bool,
    whole_word: bool,
    case_sensitive: bool,
    extra_columns_data: dict[str, list] | None = None,
    extra_columns_dtypes: dict[str, Any] | None = None,
    search_mode: str = "regex",
    node_tokens: list[Any] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run full concordance extraction and persist the flattened parquet.

    Unlike detach, no Node is produced. The cached parquet is recorded on the
    parent analysis task so subsequent pagination and detach can reuse it.

    Used by:
    - backend tests, core workspace and worker services because tests need the same
      observable contract that production routes and workers rely on.

    Flow: load text or token inputs, derive concordance rows or dispersion bins, persist
        parquet artifacts, and return metadata that the manager can serialize.
    """
    configure_worker_environment()

    try:
        import os

        if progress_callback:
            progress_callback(0.02, "Loading concordance libraries...")

        logger.info("[Worker %d] Starting concordance materialize task", os.getpid())

        if progress_callback:
            progress_callback(0.25, "Generating concordance matches...")

        if search_mode == "tokens":
            if node_tokens is None:
                raise ValueError(
                    "search_mode='tokens' requires node_tokens from the route"
                )
            result, output_columns = _build_tokens_concordance_occurrence_dataframe(
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
            result, output_columns = _build_concordance_occurrence_dataframe(
                node_corpus=node_corpus,
                document_column=document_column,
                search_word=search_word,
                num_left_tokens=num_left_tokens,
                num_right_tokens=num_right_tokens,
                regex=regex,
                whole_word=whole_word,
                case_sensitive=case_sensitive,
                include_document_column=True,
                extra_columns_data=extra_columns_data,
                extra_columns_dtypes=extra_columns_dtypes,
            )

        import polars as pl

        l1_freq = (
            result.group_by(CONC_L1_COLUMN).len().rename({"len": CONC_L1_FREQ_COLUMN})
        )
        r1_freq = (
            result.group_by(CONC_R1_COLUMN).len().rename({"len": CONC_R1_FREQ_COLUMN})
        )
        result = result.join(l1_freq, on=CONC_L1_COLUMN, how="left").join(
            r1_freq, on=CONC_R1_COLUMN, how="left"
        )
        output_columns = output_columns + [CONC_L1_FREQ_COLUMN, CONC_R1_FREQ_COLUMN]

        if progress_callback:
            progress_callback(0.85, "Writing materialized parquet...")

        cache_path = materialized_cache_path(
            workspace_dir,
            feature="concordance",
            task_id=child_task_id,
            node_id=parent_node_id,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        materialized_path = str(cache_path)
        result.write_parquet(materialized_path)

        total_source_documents = len(node_corpus)
        unique_documents_with_hits = (
            int(result.select(pl.col(document_column).n_unique()).item())
            if document_column in result.columns
            else 0
        )

        if progress_callback:
            progress_callback(1.0, "Concordance materialize completed")

        return {
            "state": "successful",
            "result": {
                "materialized_path": materialized_path,
                "parent_task_id": parent_task_id,
                "parent_node_id": parent_node_id,
                "output_columns": output_columns,
                "record_count": int(len(result)),
                "unique_documents_with_hits": unique_documents_with_hits,
                "total_source_documents": total_source_documents,
            },
            "message": "Concordance materialize completed successfully",
        }
    except Exception as e:
        return {
            "state": "failed",
            "error": str(e),
            "message": f"Concordance materialize task failed: {str(e)}",
        }
