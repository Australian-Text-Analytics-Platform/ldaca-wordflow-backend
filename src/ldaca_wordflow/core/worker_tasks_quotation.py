"""Worker implementations for quotation background tasks.

Used by:
- Backend API routes, worker tasks, workspace services, and backend tests because they
  need a backend boundary that validates inputs before delegating to workspace or worker
  state.

Flow: normalize source text, run local or remote quotation extraction, preserve
    source-row mappings, and return grouped quotation artifacts for later
    materialization.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

from ..api.workspaces.analyses.generated_columns import (
    QUOTE_COLUMN_NAMES,
    QUOTE_EXTRACTION_COLUMN,
    QUOTE_QUOTE_COLUMN,
)
from .analysis_cache import materialized_cache_path
from .worker_utils import worker_task

logger = logging.getLogger(__name__)


def _collect_quotation_source_from_snapshot(
    *,
    input_snapshot_dir: str,
    node_id: str,
    document_column: str,
    extra_column_names: list[str] | None,
    include_all_metadata: bool = False,
) -> tuple[list[str], dict[str, list] | None, dict[str, Any] | None]:
    """Collect quotation source rows inside the worker process.

    Used by:
    - quotation detach and materialize workers when submit routes pass
      task-owned LazyFrame snapshots instead of full Python corpora.

    Flow: load the snapshotted node plan, select the document and requested
    metadata columns, filter blank documents, and return aligned lists for the
    existing quotation extraction builder.
    """

    import polars as pl

    from ..api.workspaces.analyses.generated_columns import is_tokenization_column_name
    from .worker_input_snapshots import load_snapshot_node

    snapshot_node = load_snapshot_node(input_snapshot_dir, node_id)
    node_data = snapshot_node.data
    schema_names = list(node_data.collect_schema().names())
    if include_all_metadata:
        metadata_columns = [
            column
            for column in schema_names
            if column != document_column and not is_tokenization_column_name(column)
        ]
    else:
        metadata_columns = list(extra_column_names or [])

    corpus_df = cast(
        pl.DataFrame,
        node_data.select([pl.col(document_column)] + [pl.col(c) for c in metadata_columns])
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
    if not metadata_columns:
        return node_corpus, None, None

    extra_columns_data: dict[str, list] = {}
    extra_columns_dtypes: dict[str, Any] = {}
    for column in metadata_columns:
        series = corpus_df.get_column(column)
        extra_columns_data[column] = series.to_list()
        extra_columns_dtypes[column] = series.dtype
    return node_corpus, extra_columns_data, extra_columns_dtypes


@worker_task
def run_quotation_analysis_task(
    configure_worker_environment,
    user_id: str,
    workspace_id: str,
    input_snapshot_dir: str,
    node_id: str,
    request_payload: dict[str, Any],
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Execute the primary quotation analysis in a worker process.

    Used by:
    - ``core.worker.quotation_task`` because the API submit endpoint must not
      run quotation extraction or page collection on the event loop.

    Flow: configure the worker runtime, load the snapshotted node plan, reuse
    the quotation page builder, and return the persisted task result payload.
    """

    configure_worker_environment()
    try:
        if progress_callback:
            progress_callback(0.1, "Loading quotation input...")

        import asyncio

        from ..api.workspaces.analyses.quotation_core import (
            DEFAULT_CONTEXT_LENGTH,
            compute_remote_on_demand_page,
        )
        from ..models import QuotationEngineConfig
        from .worker_input_snapshots import load_snapshot_node

        snapshot_node = load_snapshot_node(input_snapshot_dir, node_id)
        node = snapshot_node.to_node()
        engine_payload = request_payload.get("engine") or {}
        engine = QuotationEngineConfig.model_validate(engine_payload)
        page_payload = asyncio.run(
            compute_remote_on_demand_page(
                node,
                str(request_payload["column"]),
                engine,
                page=int(request_payload.get("page") or 1),
                page_size=request_payload.get("page_size"),
                sort_by=request_payload.get("sort_by"),
                descending=bool(request_payload.get("descending", False)),
                materialized_path=None,
            )
        )
        if progress_callback:
            progress_callback(1.0, "Quotation analysis completed")
        return {
            **page_payload,
            "preferences": {"context_length": DEFAULT_CONTEXT_LENGTH},
        }
    except Exception as exc:
        logger.exception(
            "Quotation analysis task failed for user=%s workspace=%s node=%s",
            user_id,
            workspace_id,
            node_id,
        )
        return {
            "state": "failed",
            "message": f"Quotation analysis task failed: {exc}",
            "data": [],
            "columns": [],
        }


def _build_quotation_occurrence_dataframe(
    node_corpus: list[str],
    document_column: str,
    include_document_column: bool,
    extra_columns_data: dict[str, list] | None,
    extra_columns_dtypes: dict[str, Any] | None = None,
):
    """Extract quotation occurrences from a corpus. Returns (df, output_columns).

    Called by:
    - Local helpers, route handlers, or service methods in this module because they need a
      backend boundary that validates inputs before delegating to workspace or worker state.

    Flow: normalize source text, run local or remote quotation extraction, preserve
        source-row mappings, and return grouped quotation artifacts for later
        materialization.
    """
    import polars as pl

    from ldaca_wordflow.api.workspaces.analyses.quotation_core import (
        flatten_grouped_quotation_dataframe,
        quotation_groups_via_quote_extractor,
    )

    corpus = [str(v) if v is not None else "" for v in (node_corpus or [])]
    non_empty_mask = [bool(value.strip()) for value in corpus]
    filtered_corpus = [value for value, keep in zip(corpus, non_empty_mask) if keep]

    source_column_name = "__quotation_source__"
    data: dict[str, list] = {source_column_name: filtered_corpus}
    # `QUOTE_extraction` is the per-quote-row copy of the raw source document
    # text — exposed under a canonical name so callers (table view, detach)
    # can refer to it without needing to know the user's source column name.
    # Always carried through extraction so the materialised parquet has it
    # for the fast-path detach. The detach worker drops it from the final
    # output when the user hasn't opted into the column.
    data[QUOTE_EXTRACTION_COLUMN] = filtered_corpus
    selected_columns: list[str] = [QUOTE_EXTRACTION_COLUMN]
    output_columns: list[str] = [QUOTE_EXTRACTION_COLUMN]

    if include_document_column:
        data[document_column] = filtered_corpus
        selected_columns.append(document_column)
        output_columns.append(document_column)

    if extra_columns_data:
        for col_name, col_values in extra_columns_data.items():
            filtered_values = [
                value for value, keep in zip(col_values, non_empty_mask) if keep
            ]
            data[col_name] = filtered_values
            selected_columns.append(col_name)
            output_columns.append(col_name)

    input_df = pl.DataFrame(data)
    if extra_columns_dtypes:
        cast_exprs = [
            pl.col(col).cast(dtype)
            for col, dtype in extra_columns_dtypes.items()
            if col in input_df.columns and input_df.schema[col] != dtype
        ]
        if cast_exprs:
            input_df = input_df.with_columns(cast_exprs)
    quote_df = quotation_groups_via_quote_extractor(input_df, source_column_name)
    quote_df = flatten_grouped_quotation_dataframe(quote_df)
    generated_columns = [
        column_name
        for column_name in QUOTE_COLUMN_NAMES
        if column_name in quote_df.columns
    ]
    quote_df = quote_df.select(selected_columns + generated_columns)

    if QUOTE_QUOTE_COLUMN in quote_df.columns:
        quote_df = quote_df.filter(pl.col(QUOTE_QUOTE_COLUMN).is_not_null())

    return quote_df, output_columns + generated_columns


@worker_task
def run_quotation_detach_task(
    configure_worker_environment,
    workspace_dir: str,
    node_corpus: list[str],
    parent_node_id: str,
    document_column: str,
    engine_config: dict[str, Any],
    new_node_name: str,
    include_document_column: bool = False,
    include_extraction: bool = False,
    selected_generated_columns: list[str] | None = None,
    extra_columns_data: dict[str, list] | None = None,
    extra_columns_dtypes: dict[str, Any] | None = None,
    materialized_path: str | None = None,
    input_snapshot_dir: str | None = None,
    extra_column_names: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run quotation detach and return a serialized detached node payload.

    Fast path: when `materialized_path` points to an existing parquet, wrap it
    directly as the detached node without re-extracting quotations.

    Used by:
    - backend tests, core workspace and worker services because tests need the same
      observable contract that production routes and workers rely on.

    Flow: normalize source text, run local or remote quotation extraction, preserve
        source-row mappings, and return grouped quotation artifacts for later
        materialization.
    """
    configure_worker_environment()

    try:
        if progress_callback:
            progress_callback(0.02, "Loading quotation extractor...")

        import os

        import polars as pl

        from docworkspace import Node

        logger.info("[Worker %d] Starting quotation detach task", os.getpid())

        if materialized_path and os.path.exists(materialized_path):
            if progress_callback:
                progress_callback(0.4, "Reusing materialized quotations...")
            # Copy the materialized parquet into a detach-owned file so the
            # parent analysis task's materialized artifact can be cleaned up
            # independently (e.g. when the user clears results) without
            # breaking this detached node's serialized query plan.
            import shutil
            import uuid

            detach_data_dir = os.path.join(workspace_dir, "data")
            os.makedirs(detach_data_dir, exist_ok=True)
            detach_parquet_path = os.path.join(
                detach_data_dir,
                f"quotation_detach_{uuid.uuid4().hex}.parquet",
            )
            # The materialised parquet always carries every generated quote
            # column plus QUOTE_extraction (so re-ticking is cheap), but the
            # detached node should respect the user's column picks: keep only
            # the generated columns they left ticked, opt-in QUOTE_extraction,
            # and pass through the document/metadata columns.
            mat_lazy = pl.scan_parquet(materialized_path)
            mat_columns = list(mat_lazy.collect_schema().names())
            generated_set = set(QUOTE_COLUMN_NAMES)
            wanted_generated = set(selected_generated_columns or [])
            keep_cols: list[str] = []
            for col in mat_columns:
                if col in generated_set:
                    if col in wanted_generated:
                        keep_cols.append(col)
                elif col == QUOTE_EXTRACTION_COLUMN:
                    if include_extraction:
                        keep_cols.append(col)
                else:
                    keep_cols.append(col)
            if keep_cols != mat_columns:
                mat_df = cast(
                    pl.DataFrame,
                    mat_lazy.select(keep_cols).collect(),
                )
                mat_df.write_parquet(detach_parquet_path)
            else:
                shutil.copy2(materialized_path, detach_parquet_path)
            lazy = pl.scan_parquet(detach_parquet_path)
            schema_names = list(lazy.collect_schema().names())
            record_count = int(
                cast(pl.DataFrame, lazy.select(pl.len()).collect()).item() or 0
            )
            detached_node = Node(
                data=lazy,
                name=new_node_name,
                workspace=None,
                operation="quotation_detach",
                parents=[parent_node_id],
                document=document_column,
            )
            node_payload = detached_node.to_dict(base_dir=workspace_dir)
            if progress_callback:
                progress_callback(1.0, "Quotation detach completed")
            return {
                "state": "successful",
                "result": {
                    "node_payload": node_payload,
                    "output_columns": schema_names,
                    "record_count": record_count,
                    "engine_config": engine_config,
                },
                "message": "Quotation detach completed successfully",
            }

        if progress_callback:
            progress_callback(0.2, "Preparing text data...")
        if input_snapshot_dir is not None:
            node_corpus, extra_columns_data, extra_columns_dtypes = (
                _collect_quotation_source_from_snapshot(
                    input_snapshot_dir=input_snapshot_dir,
                    node_id=parent_node_id,
                    document_column=document_column,
                    extra_column_names=extra_column_names,
                )
            )
        if progress_callback:
            progress_callback(0.6, "Extracting quotations...")

        quote_df, output_columns = _build_quotation_occurrence_dataframe(
            node_corpus=node_corpus,
            document_column=document_column,
            include_document_column=include_document_column,
            extra_columns_data=extra_columns_data,
            extra_columns_dtypes=extra_columns_dtypes,
        )

        # Final projection honoring the user's column choice. Generated quote
        # columns are kept only when ticked; QUOTE_extraction stays opt-in; the
        # document column and metadata columns pass through.
        generated_set = set(QUOTE_COLUMN_NAMES)
        wanted_generated = set(selected_generated_columns or [])
        keep_columns: list[str] = []
        for col in output_columns:
            if col in generated_set:
                if col in wanted_generated:
                    keep_columns.append(col)
            elif col == QUOTE_EXTRACTION_COLUMN:
                if include_extraction:
                    keep_columns.append(col)
            else:
                keep_columns.append(col)
        if keep_columns and keep_columns != output_columns:
            quote_df = quote_df.select(keep_columns)
            output_columns = keep_columns

        if progress_callback:
            progress_callback(0.82, "Serializing detached data block...")

        detached_node = Node(
            data=quote_df.lazy(),
            name=new_node_name,
            workspace=None,
            operation="quotation_detach",
            parents=[parent_node_id],
            document=document_column,
        )
        node_payload = detached_node.to_dict(base_dir=workspace_dir)

        if progress_callback:
            progress_callback(1.0, "Quotation detach completed")

        logger.info(
            "[Worker %d] Quotation detach task completed successfully", os.getpid()
        )

        return {
            "state": "successful",
            "result": {
                "node_payload": node_payload,
                "output_columns": output_columns,
                "record_count": int(quote_df.height),
                "engine_config": engine_config,
            },
            "message": "Quotation detach completed successfully",
        }
    except Exception as e:
        return {
            "state": "failed",
            "error": str(e),
            "message": f"Quotation detach task failed: {str(e)}",
        }


@worker_task
def run_quotation_materialize_task(
    configure_worker_environment,
    workspace_dir: str,
    node_corpus: list[str],
    child_task_id: str,
    parent_task_id: str,
    parent_node_id: str,
    document_column: str,
    engine_config: dict[str, Any],
    extra_columns_data: dict[str, list] | None = None,
    extra_columns_dtypes: dict[str, Any] | None = None,
    input_snapshot_dir: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Run full quotation extraction and persist the flattened parquet.

    Used by:
    - core workspace and worker services because background jobs need one lifecycle owner
      for submission, progress, cancellation, and artifact cleanup.

    Flow: normalize source text, run local or remote quotation extraction, preserve
        source-row mappings, and return grouped quotation artifacts for later
        materialization.
    """
    configure_worker_environment()

    try:
        import os

        if progress_callback:
            progress_callback(0.02, "Loading quotation extractor...")

        logger.info("[Worker %d] Starting quotation materialize task", os.getpid())

        if progress_callback:
            progress_callback(0.3, "Extracting quotations...")
        if input_snapshot_dir is not None:
            node_corpus, extra_columns_data, extra_columns_dtypes = (
                _collect_quotation_source_from_snapshot(
                    input_snapshot_dir=input_snapshot_dir,
                    node_id=parent_node_id,
                    document_column=document_column,
                    extra_column_names=None,
                    include_all_metadata=True,
                )
            )

        quote_df, output_columns = _build_quotation_occurrence_dataframe(
            node_corpus=node_corpus,
            document_column=document_column,
            include_document_column=True,
            extra_columns_data=extra_columns_data,
            extra_columns_dtypes=extra_columns_dtypes,
        )

        if progress_callback:
            progress_callback(0.85, "Writing materialized parquet...")

        cache_path = materialized_cache_path(
            workspace_dir,
            feature="quotation",
            task_id=child_task_id,
            node_id=parent_node_id,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        materialized_path = str(cache_path)
        quote_df.write_parquet(materialized_path)

        import polars as pl

        total_source_documents = len(node_corpus)
        unique_documents_with_hits = (
            int(quote_df.select(pl.col(document_column).n_unique()).item())
            if document_column in quote_df.columns
            else 0
        )

        if progress_callback:
            progress_callback(1.0, "Quotation materialize completed")

        return {
            "state": "successful",
            "result": {
                "materialized_path": materialized_path,
                "parent_task_id": parent_task_id,
                "parent_node_id": parent_node_id,
                "output_columns": output_columns,
                "record_count": int(quote_df.height),
                "unique_documents_with_hits": unique_documents_with_hits,
                "total_source_documents": total_source_documents,
                "engine_config": engine_config,
            },
            "message": "Quotation materialize completed successfully",
        }
    except Exception as e:
        return {
            "state": "failed",
            "error": str(e),
            "message": f"Quotation materialize task failed: {str(e)}",
        }
